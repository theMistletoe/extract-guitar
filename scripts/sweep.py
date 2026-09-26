#!/usr/bin/env python
"""Grid search over pipeline parameters (PRD §26). Every point is a full experiment
(validation metrics, results.csv row, notes.md), so nothing is lost.

A sweep file (configs/sweeps/*.yaml):

    base: A_xlance                 # pipeline to start from
    slug: sweep_xlance_overlap
    strategy: "A+tuning"
    target: best                   # run the target song for: none | best | all
    grid:
      guitar.params.num_overlap: [2, 4, 8]
      guitar.params.tta: [[], [swap, invert]]
    # keys are <step id>.<field>[.<subfield>]; a list of values per key

    python scripts/sweep.py configs/sweeps/xlance_overlap.yaml
"""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.pipeline import load_pipeline  # noqa: E402

GEN = ROOT / "configs" / "pipelines" / "generated"


def set_path(pipe: dict, key: str, value) -> None:
    sid, *path = key.split(".")
    step = next(s for s in pipe["steps"] if s["id"] == sid)
    node = step
    for p in path[:-1]:
        node = node.setdefault(p, {})
    node[path[-1]] = value


def short(v) -> str:
    if isinstance(v, list):
        return "-".join(map(str, v)) or "none"
    return str(v).replace(".", "p")


def main(sweep_file: str) -> int:
    sw = yaml.safe_load(Path(sweep_file).read_text())
    base = load_pipeline(sw["base"])
    keys = list(sw["grid"])
    GEN.mkdir(parents=True, exist_ok=True)
    done = []
    for combo in itertools.product(*[sw["grid"][k] for k in keys]):
        pipe = copy.deepcopy(base)
        tag = "_".join(f"{k.split('.')[-1]}{short(v)}" for k, v in zip(keys, combo))
        pipe["name"] = f"{sw['slug']}__{tag}"
        pipe["description"] = f"{base.get('description', base['name'])} | sweep {dict(zip(keys, combo))}"
        for k, v in zip(keys, combo):
            set_path(pipe, k, v)
        path = GEN / f"{pipe['name']}.yaml"
        path.write_text(yaml.safe_dump(pipe, sort_keys=False))
        exp_id = tracking.next_experiment_id(f"{sw['slug']}_{tag}"[:60])
        args = [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--pipeline", str(path),
                "--slug", "x", "--exp-id", exp_id, "--strategy", sw.get("strategy", "tuning"),
                "--hypothesis", sw.get("hypothesis", f"parameter sweep over {keys}"),
                "--change", json.dumps(dict(zip(keys, combo))), "--next", sw.get("next", "")]
        if sw.get("target", "best") != "all":
            args.append("--no-target")
        if sw.get("no_bss"):
            args.append("--no-bss")
        print(">>>", exp_id, flush=True)
        subprocess.run(args, cwd=ROOT, check=False)
        done.append((exp_id, path))
    if sw.get("target", "best") == "best" and done:
        scores = {}
        for exp_id, path in done:
            m = json.loads((ROOT / "experiments" / exp_id / "metrics.json").read_text())
            scores[exp_id] = (m.get("validation", {}).get("sdr_mean", -99), path)
        best = max(scores, key=lambda e: scores[e][0])
        print(f"best point {best} ({scores[best][0]:.3f} dB) -> rendering the target song")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "target_only.py"), best], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
