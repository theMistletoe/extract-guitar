#!/usr/bin/env python
"""Train the learned mask refiner (stacking ensemble) on TRAINING data only.

The benchmark validation set (datasets/validation) is never touched here; early stopping
uses a held-out slice of the training clips. The resulting checkpoint is then evaluated by
scripts/benchmark.py like any other challenger.

    python scripts/train.py prepare --clips-per-song 6 --n-syn 90 --seconds 6
    python scripts/train.py candidates --pos xlance_gtr sw6 mega_acoustic --neg mega_violin mega_woodwind
    python scripts/train.py fit --pos xlance_gtr sw6 mega_acoustic --neg mega_violin mega_woodwind \
        --epochs 30 --out artifacts/refiner/r001
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from acoustic_separator.audio import load_audio, save_audio  # noqa: E402
from acoustic_separator.inference import InferenceParams  # noqa: E402
from acoustic_separator.mixing import (SCENARIOS, Source, load_song_groups, make_mixture,  # noqa: E402
                                       make_multitrack_clip)
from acoustic_separator.pipeline import ModelPool, PipelineRunner  # noqa: E402
from acoustic_separator.refiner import MaskRefiner, features, istft, refiner_loss  # noqa: E402

CLIPS = ROOT / "datasets" / "train_clips"
MANIFEST = ROOT / "datasets" / "manifest.csv"
# Training clips are 6 s; candidates are computed with a 6 s chunk so every clip costs exactly
# one forward pass per model (default 13-20 s chunks would mostly process zero padding).
PARAMS = {"num_overlap": 2, "precision": "bf16", "chunk_size": 264600}


def train_pool():
    pool, songs = {}, []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            p = ROOT / r["path"]
            if not p.exists():
                continue
            if r["type"] == "multitrack_song":
                songs.append(p)
            else:
                pool.setdefault(r["instrument"], []).append(
                    Source(id=r["id"], path=str(p), cls=r["instrument"], license=r["license"],
                           dataset=r["source"], split="train"))
    ag = pool.get("acoustic_guitar", [])
    nylon = [x for x in ag if x.id.startswith("gaps_")]
    steel = [x for x in ag if not x.id.startswith("gaps_")]
    if nylon and steel:  # balance nylon / steel string guitars (the target is nylon)
        pool["acoustic_guitar"] = steel + nylon * max(1, round(len(steel) / len(nylon)))
    return pool, songs


def cmd_prepare(a):
    pool, songs = train_pool()
    rng = np.random.default_rng(a.seed)
    CLIPS.mkdir(parents=True, exist_ok=True)
    n = 0
    for song in songs if a.clips_per_song > 0 else []:
        groups = load_song_groups(song)
        if "acoustic_guitar" not in groups:
            continue
        used = []
        for k in range(a.clips_per_song):
            clip = make_multitrack_clip(song, rng, a.seconds, groups=groups, exclude=used, strength=0.8)
            used.append((clip["start"], clip["start"] + int(a.seconds * 44100)))
            _write(CLIPS / f"ms_{song.name[:30].replace(' ', '_')}_{k}", clip)
            n += 1
    print({k: len(v) for k, v in pool.items()})
    scen = [s for s, d in SCENARIOS.items() if any(pool.get(c) for c in d["classes"])]
    weights = json.loads(Path(a.scenario_weights).read_text()) if a.scenario_weights else {}
    p = np.array([float(weights.get(sc, 1.0)) for sc in scen])
    p = p / p.sum()
    for k in range(a.n_syn):
        sc = scen[k % len(scen)] if not weights else str(rng.choice(scen, p=p))
        clip = make_mixture(pool, sc, rng, seconds=a.seconds)
        _write(CLIPS / f"syn{a.tag}_{sc}_{k:03d}", clip)
        n += 1
    print(f"{n} training clips in {CLIPS}")


def _write(d: Path, clip: dict):
    d.mkdir(parents=True, exist_ok=True)
    save_audio(d / "mixture.wav", clip["mixture"], 44100)
    save_audio(d / "acoustic_guitar.wav", clip["stems"]["acoustic_guitar"], 44100)
    for k, v in clip["stems"].items():
        if k != "acoustic_guitar":
            save_audio(d / "stems" / f"{k}.wav", v, 44100)
    (d / "meta.json").write_text(json.dumps(clip["info"], default=str))


def _spec(s: str):
    """'model' -> (model, None) ; 'model:stem' -> (model, stem)."""
    return tuple(s.split(":", 1)) if ":" in s else (s, None)


def _run(runner, spec, mix):
    """Candidate stem for a spec. Specs chain with '>': 'sw6:guitar>mega_multi:~electric-guitar'
    runs the second model on the first output; '~stem' means input minus that stem."""
    if ">" in spec:
        first, rest = spec.split(">", 1)
        return _run(runner, rest, _run(runner, first, mix))
    model, stem = _spec(spec)
    out, _, _ = runner._separate(model, mix, 44100, InferenceParams(**PARAMS))
    cat = runner.catalog[model]
    stem = stem or cat.target_stem or cat.stems[0]
    if stem.startswith("~"):
        return mix - out[stem[1:]]
    return out[stem]


def cmd_candidates(a):
    runner = PipelineRunner(ModelPool(capacity=1))
    clips = sorted(p for p in CLIPS.iterdir() if (p / "mixture.wav").exists())
    for spec in a.pos + a.neg:  # model-major order keeps a single model in memory
        t = time.time()
        for c in clips:
            mix, _ = load_audio(c / "mixture.wav")
            _run(runner, spec, mix)
        print(f"{spec}: {len(clips)} clips in {time.time() - t:.0f}s", flush=True)


def load_training_set(pos, neg):
    runner = PipelineRunner(ModelPool(capacity=1))
    data = []
    for c in sorted(p for p in CLIPS.iterdir() if (p / "mixture.wav").exists()):
        mix, _ = load_audio(c / "mixture.wav")
        ref, _ = load_audio(c / "acoustic_guitar.wav")
        data.append({"name": c.name, "mix": mix, "ref": ref,
                     "pos": [_run(runner, s, mix) for s in pos],
                     "neg": [_run(runner, s, mix) for s in neg]})
    return data


def batches(data, bs, crop, rng, weights=None):
    idx = list(range(len(data)))
    if weights:  # hard-example oversampling (sampling with replacement)
        w = [weights.get(d["name"], 1.0) for d in data]
        idx = rng.choices(idx, weights=w, k=len(idx))
    else:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        items = [data[j] for j in idx[i:i + bs]]
        T = min(d["mix"].shape[-1] for d in items)
        L = min(crop, T)
        out = {"mix": [], "ref": [], "pos": [], "neg": []}
        for d in items:
            s = rng.randint(0, T - L)
            g = rng.uniform(0.5, 1.5)
            out["mix"].append(d["mix"][:, s:s + L] * g)
            out["ref"].append(d["ref"][:, s:s + L] * g)
            out["pos"].append([p[:, s:s + L] * g for p in d["pos"]])
            out["neg"].append([n[:, s:s + L] * g for n in d["neg"]])
        t = lambda x: torch.from_numpy(np.stack(x).astype(np.float32))  # noqa: E731
        yield (t(out["mix"]), t(out["ref"]),
               [t([p[k] for p in out["pos"]]) for k in range(len(out["pos"][0]))],
               [t([n[k] for n in out["neg"]]) for k in range(len(out["neg"][0]))])


def evaluate(model, data, base_members=None):
    model.eval()
    sdrs = []
    with torch.no_grad():
        for d in data:
            t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
            feat, X, base = features(t(d["mix"]), [t(p) for p in d["pos"]], [t(n) for n in d["neg"]],
                                     base_members)
            y = istft(model.apply(feat, X, base), d["mix"].shape[-1])[0].numpy()
            b = istft(base if base_members is not None else X * base, d["mix"].shape[-1])[0].numpy()
            r = d["ref"]
            sd = lambda e: 10 * np.log10((r ** 2).sum() / (((r - e) ** 2).sum() + 1e-10) + 1e-10)  # noqa: E731
            sdrs.append((sd(y), sd(b)))
    a = np.array(sdrs)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


FAILURE_OF_CLASS = {"piano": "piano_leak", "keys": "piano_leak", "electric_clean": "electric_leak",
                    "electric_dist": "electric_leak", "electric_guitar": "electric_leak",
                    "vocals": "vocal_leak", "drums": "drum_leak", "percussion": "drum_leak",
                    "violin": "strings_winds_leak", "strings": "strings_winds_leak",
                    "winds": "strings_winds_leak", "brass": "strings_winds_leak",
                    "plucked_other": "plucked_leak", "bass": "bass_leak", "synth": "synth_leak"}
SCENARIOS_FOR_FAILURE = {"piano_leak": ["piano", "piano_band"], "electric_leak": ["clean_electric", "electric_band"],
                         "vocal_leak": ["female_vocal", "band_pop"], "drum_leak": ["cymbal_drums"],
                         "strings_winds_leak": ["strings", "winds", "chamber_frevo"],
                         "plucked_leak": ["plucked"], "guitar_removed": ["buried", "dense"],
                         "bass_leak": ["band_pop"], "synth_leak": ["dense"]}


def cmd_mine(a):
    """Hard-example mining on TRAINING clips: classify each clip's dominant failure."""
    from acoustic_separator.evaluation import evaluate_estimate

    data = load_training_set(a.pos, a.neg)
    model, bm = None, None
    if a.refiner:
        ck = torch.load(a.refiner, map_location="cpu", weights_only=False)
        model = MaskRefiner(**ck["model_kwargs"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        bm = ck.get("base_members")
    report = {}
    for d in data:
        t = lambda x: torch.from_numpy(x)[None]  # noqa: E731
        with torch.no_grad():
            feat, X, base = features(t(d["mix"]), [t(p) for p in d["pos"]], [t(n) for n in d["neg"]], bm)
            if model is not None:
                spec = model.apply(feat, X, base)
            else:
                spec = X * base
            est = istft(spec, d["mix"].shape[-1])[0].numpy()
        stems = {p.stem: load_audio(p)[0] for p in sorted((CLIPS / d["name"] / "stems").glob("*.wav"))}
        mm = evaluate_estimate(d["ref"], est, d["mix"], stems, with_bss=False)
        leaks = {k[5:-3]: v for k, v in mm.items() if k.startswith("leak_") and k.endswith("_db")}
        worst_cls = max(leaks, key=leaks.get) if leaks else None
        if mm["target_retention"] < a.retention_floor:
            failure = "guitar_removed"
        elif worst_cls:
            failure = FAILURE_OF_CLASS.get(worst_cls.rstrip("0123456789"), "other_leak")
        else:
            failure = "other"
        report[d["name"]] = {"sdr": mm["sdr"], "retention": mm["target_retention"],
                             "worst_leak_class": worst_cls, "worst_leak_db": leaks.get(worst_cls),
                             "failure": failure}
    sdrs = np.array([r["sdr"] for r in report.values()])
    cut = np.percentile(sdrs, a.hard_percentile)
    hard = {k: v for k, v in report.items() if v["sdr"] <= cut}
    counts = {}
    for v in hard.values():
        counts[v["failure"]] = counts.get(v["failure"], 0) + 1
    weights = {k: (a.hard_weight if k in hard else 1.0) for k in report}
    scen_w = {}
    for f, c in counts.items():
        for sc in SCENARIOS_FOR_FAILURE.get(f, []):
            scen_w[sc] = scen_w.get(sc, 1.0) + c
    out = Path(a.out)
    out.write_text(json.dumps({"clips": report, "hard_threshold_sdr": float(cut),
                               "hard_failure_counts": counts, "sample_weights": weights,
                               "scenario_weights": scen_w}, indent=1))
    (out.parent / (out.stem + "_scenario_weights.json")).write_text(json.dumps(scen_w, indent=1))
    print(f"mean SDR {sdrs.mean():.2f}; hard (<= {cut:.2f} dB) failures: {counts}")
    print(f"scenario weights for the next prepare: {scen_w}")


def cmd_fit(a):
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    data = load_training_set(a.pos, a.neg)
    rng.shuffle(data)
    n_hold = max(4, int(len(data) * 0.12))
    hold, train = data[:n_hold], data[n_hold:]
    bm = a.base_members if a.mode == "gain" else None
    if bm is None:
        in_ch = 2 + len(a.pos) + len(a.neg) + (1 if len(a.pos) > 1 else 0)
    else:
        in_ch = 3 + len(a.pos) + len(a.neg)
    kwargs = {"in_ch": in_ch, "width": a.width, "depth": a.depth, "mode": a.mode}
    model = MaskRefiner(**kwargs)
    print(f"{len(train)} train / {len(hold)} held-out clips; params "
          f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f}k", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h_sdr, b_sdr = evaluate(model, hold, bm)
    log = [{"epoch": 0, "hold_sdr": h_sdr, "hold_base_sdr": b_sdr}]
    print(f"epoch 0 hold-out SDR {h_sdr:.3f} (base {b_sdr:.3f})", flush=True)
    best = h_sdr
    torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs, "pos": a.pos, "neg": a.neg,
                "base_members": bm, "epoch": 0, "hold_sdr": h_sdr}, out / "model.pt")
    crop = int(a.crop_s * 44100)
    sample_w = json.loads(Path(a.weights).read_text())["sample_weights"] if a.weights else None
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for mix, ref, pos, neg in batches(train, a.batch, crop, rng, sample_w):
            feat, X, base = features(mix, pos, neg, bm)
            est = istft(model.apply(feat, X, base), mix.shape[-1])
            loss = refiner_loss(est, ref)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        h_sdr, _ = evaluate(model, hold, bm)
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "hold_sdr": h_sdr})
        flag = ""
        if h_sdr > best:
            best = h_sdr
            flag = " *"
            torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs, "pos": a.pos,
                        "neg": a.neg, "base_members": bm, "epoch": ep, "hold_sdr": h_sdr}, out / "model.pt")
        print(f"epoch {ep} loss {np.mean(losses):.4f} hold-out SDR {h_sdr:.3f}{flag} "
              f"({time.time() - t0:.0f}s)", flush=True)
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log,
                                                    "train_clips": [d["name"] for d in train],
                                                    "holdout_clips": [d["name"] for d in hold]}, indent=1))
    print(f"best hold-out SDR {best:.3f} (base {b_sdr:.3f}) -> {out / 'model.pt'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--clips-per-song", type=int, default=6)
    p.add_argument("--n-syn", type=int, default=90)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--scenario-weights", default=None, help="JSON {scenario: weight} (from mine)")
    p.add_argument("--tag", default="", help="suffix for synthetic clip names (new batches)")
    for name in ("candidates", "fit", "mine"):
        q = sub.add_parser(name)
        q.add_argument("--pos", nargs="+", required=True)
        q.add_argument("--neg", nargs="*", default=[])
        q.add_argument("--seed", type=int, default=0)
        if name == "fit":
            q.add_argument("--epochs", type=int, default=30)
            q.add_argument("--batch", type=int, default=2)
            q.add_argument("--crop-s", type=float, default=3.0)
            q.add_argument("--lr", type=float, default=1e-3)
            q.add_argument("--width", type=int, default=24)
            q.add_argument("--depth", type=int, default=6)
            q.add_argument("--out", required=True)
            q.add_argument("--weights", default=None, help="mining JSON with sample_weights")
            q.add_argument("--mode", choices=["mask", "gain"], default="mask",
                           help="mask: re-mask the mixture; gain: 0..2 TF gain on the plain ensemble")
            q.add_argument("--base-members", nargs="*", type=int, default=[0, 1],
                           help="gain mode: indices into --pos averaged as the base ensemble")
        if name == "mine":
            q.add_argument("--refiner", default=None)
            q.add_argument("--retention-floor", type=float, default=0.5)
            q.add_argument("--hard-percentile", type=float, default=25)
            q.add_argument("--hard-weight", type=float, default=3.0)
            q.add_argument("--out", default=str(CLIPS / "mining.json"))
    a = ap.parse_args()
    {"prepare": cmd_prepare, "candidates": cmd_candidates, "fit": cmd_fit, "mine": cmd_mine}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
