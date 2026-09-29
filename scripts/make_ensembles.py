#!/usr/bin/env python
"""Generate Strategy D (ensemble) pipelines + a queue from member pipelines.

Members are existing pipelines (e.g. A_xlance A_sw6); their steps are copied with
prefixed ids so cached separator outputs are re-used (same model + params + input).

    python scripts/make_ensembles.py --members A_xlance A_sw6 A_mega_guitar \
        --methods wave_mean mag_mean mask_mean mag_median --tag top3 --wiener \
        --queue configs/queues/phase5_ensembles.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.pipeline import PIPELINE_DIR, load_pipeline  # noqa: E402


def member_steps(name: str, prefix: str) -> tuple[list[dict], str]:
    p = load_pipeline(name)
    ids = {s["id"]: f"{prefix}_{s['id']}" for s in p["steps"]}
    steps = []
    for s in p["steps"]:
        s = copy.deepcopy(s)
        s["id"] = ids[s["id"]]
        for key in ("input", "a", "b", "mix_ref"):
            if key in s and s[key] in ids:
                s[key] = ids[s[key]]
        for key in ("ensemble", "inputs", "positives", "negatives"):
            if key in s:
                s[key] = [ids.get(r, r) for r in s[key]]
        steps.append(s)
    return steps, ids[p.get("output", p["steps"][-1]["id"])]


def build(members: list[str], method: str, name: str, weights=None, extra=None, wiener=False) -> dict:
    steps, outs = [], []
    for i, m in enumerate(members):
        st, out = member_steps(m, f"m{i}")
        steps += st
        outs.append(out)
    ens = {"id": "ens", "ensemble": outs, "method": method}
    if weights:
        ens["weights"] = weights
    ens.update(extra or {})
    steps.append(ens)
    output = "ens"
    if wiener:
        steps.append({"id": "wf", "post": "wiener", "input": "ens", "iterations": 1, "power": 2.0})
        output = "wf"
    return {"name": name,
            "description": f"Strategy D: {method} of {', '.join(members)}" + (" + Wiener" if wiener else ""),
            "steps": steps, "output": output}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["wave_mean", "mag_mean", "mask_mean"])
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--wiener", action="store_true", help="also emit a Wiener-refined variant of each")
    ap.add_argument("--queue", default=None, help="append entries to this queue YAML")
    a = ap.parse_args()
    entries = []
    for method in a.methods:
        variants = [False, True] if a.wiener else [False]
        for wf in variants:
            name = f"D_{a.tag}_{method}" + ("_wiener" if wf else "")
            pipe = build(a.members, method, name, a.weights, wiener=wf)
            (PIPELINE_DIR / f"{name}.yaml").write_text(yaml.safe_dump(pipe, sort_keys=False))
            entries.append({"pipeline": name, "slug": name, "strategy": "D",
                            "hypothesis": f"Combining {', '.join(a.members)} with {method}"
                                          f"{' and a Wiener refinement' if wf else ''} averages out "
                                          "model-specific errors and beats the best single member.",
                            "change": f"ensemble method={method}, weights={a.weights or 'equal'}"
                                      f"{', wiener post-filter' if wf else ''}",
                            "next": "Compare methods; keep the best as Champion candidate."})
            print("wrote", name)
    if a.queue:
        q = Path(a.queue)
        old = yaml.safe_load(q.read_text()) if q.exists() else []
        q.write_text(yaml.safe_dump((old or []) + entries, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
