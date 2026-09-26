#!/usr/bin/env python
"""Write the CLI quality presets from the experiment results.

* max      = current Champion pipeline, every model step at fp32 (final-render precision)
* standard = best single-model (Strategy A) pipeline, bf16
* fast     = the cheapest pipeline within 1 dB of the best Strategy-A SDR (HTDemucs class)

    python scripts/update_presets.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402
from acoustic_separator.pipeline import PIPELINE_DIR  # noqa: E402


def exp_pipeline(exp: str) -> dict:
    return yaml.safe_load((ROOT / "experiments" / exp / "pipeline.yaml").read_text())


def with_precision(pipe: dict, precision: str) -> dict:
    pipe = copy.deepcopy(pipe)
    for s in pipe["steps"]:
        if "model" in s:
            s.setdefault("params", {})["precision"] = precision
    return pipe


def write(name: str, pipe: dict, note: str) -> None:
    pipe = copy.deepcopy(pipe)
    pipe["description"] = f"[preset {name}] {note} | {pipe.get('description', pipe['name'])}"
    pipe["name"] = name
    (PIPELINE_DIR / f"{name}.yaml").write_text(yaml.safe_dump(pipe, sort_keys=False))
    print(f"{name}: {note}")


def main() -> int:
    champ = tracking.current_champion()
    rows = [r for r in tracking.read_results() if r.get("sdr")]
    if not champ or not rows:
        raise SystemExit("need results and a champion")
    write("max", with_precision(champ["pipeline"], "fp32"),
          f"Champion {champ['experiment']} (val SDR {champ['metrics']['sdr_mean']:.2f} dB), fp32")
    single = [r for r in rows if r["strategy"] == "A"]
    best = max(single, key=lambda r: float(r["sdr"]))
    write("standard", exp_pipeline(best["experiment"]),
          f"best single model {best['experiment']} (val SDR {float(best['sdr']):.2f} dB)")

    def cost(r):
        try:
            return float(r["runtime_target_s"])
        except (TypeError, ValueError):
            return 1e9

    ok = [r for r in single if float(r["sdr"]) >= float(best["sdr"]) - 1.0] or [best]
    fast = min(ok, key=cost)
    if cost(fast) > 0.5 * cost(best):  # nothing close enough is cheap: fall back to the cheapest
        fast = min(single, key=cost)
    write("fast", exp_pipeline(fast["experiment"]),
          f"fast preset {fast['experiment']} (val SDR {float(fast['sdr']):.2f} dB, "
          f"{cost(fast):.0f} s per 132 s song on the reference CPU)")
    (PIPELINE_DIR / "presets.json").write_text(json.dumps(
        {"max": champ["experiment"], "standard": best["experiment"], "fast": fast["experiment"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
