#!/usr/bin/env python
"""Run a queue of experiments (configs/queues/*.yaml) sequentially through benchmark.py.

    python scripts/run_queue.py configs/queues/phase2_baselines.yaml
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(queue: str, *extra: str) -> int:
    items = yaml.safe_load(Path(queue).read_text())
    for it in items:
        cmd = [sys.executable, str(ROOT / "scripts" / "benchmark.py"),
               "--pipeline", it["pipeline"], "--slug", it["slug"],
               "--strategy", it.get("strategy", ""), "--hypothesis", it.get("hypothesis", ""),
               "--change", it.get("change", ""), "--next", it.get("next", ""), *extra,
               *it.get("args", [])]
        print(">>>", it["slug"], flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f"!!! {it['slug']} failed with {r.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
