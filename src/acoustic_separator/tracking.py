"""Experiment records, results.csv and the Champion / Challenger registry."""
from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
RESULTS_CSV = EXPERIMENTS / "results.csv"
CHAMPION_DIR = REPO_ROOT / "artifacts" / "champion"

# Primary selection metric on the ground-truth validation set.
PRIMARY_METRIC = "sdr_mean"
CSV_FIELDS = [
    "experiment", "timestamp", "strategy", "pipeline", "models", "checkpoints", "chunk",
    "overlap", "tta", "sdr", "si_sdr", "sdri", "sir", "sar", "target_retention", "leakage",
    "mrstft", "hard_sdr", "sdr_ms", "sdr_syn", "target_guitar_prob", "target_leak_prob", "runtime_val_s",
    "runtime_target_s", "is_champion", "git_commit",
]


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                           text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_info() -> dict:
    import numpy
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": str(torch.__version__),
        "numpy": str(numpy.__version__),
        "cuda": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import distributions

        info["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
    except Exception:  # noqa: BLE001
        pass
    return info


def next_experiment_id(slug: str) -> str:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[3:6]) for p in EXPERIMENTS.glob("exp[0-9][0-9][0-9]_*") if p.is_dir()]
    return f"exp{(max(nums) + 1) if nums else 1:03d}_{slug}"


def write_experiment(exp_id: str, config: dict, metrics: dict, notes: str,
                     command: str | None = None) -> Path:
    d = EXPERIMENTS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["reproducibility"] = {
        "git_commit": git_commit(),
        "command": command or " ".join(sys.argv),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": config.get("seed", 0),
        "hardware": {k: v for k, v in environment_info().items() if k != "packages"},
    }
    config = json.loads(json.dumps(config, default=str))  # plain types only (yaml.safe_dump)
    (d / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))
    (d / "notes.md").write_text(notes)
    env = environment_info()
    (d / "environment.txt").write_text("\n".join(env.pop("packages", [])) + "\n")
    return d


def append_results(row: dict) -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    rows = read_results()
    if rows and list(rows[0].keys()) != CSV_FIELDS:  # schema changed: rewrite with new header
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def read_results() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def current_champion() -> dict | None:
    p = CHAMPION_DIR / "champion.json"
    return json.loads(p.read_text()) if p.exists() else None


def challenge(exp_id: str, pipeline: dict, metrics: dict, min_gain_db: float = 0.05) -> bool:
    """Promote ``exp_id`` if it beats the Champion on the validation primary metric.

    Past champions are appended to history.jsonl and never deleted.
    """
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    champ = current_champion()
    score = metrics.get(PRIMARY_METRIC)
    if score is None:
        return False
    if champ is not None and score < champ["metrics"][PRIMARY_METRIC] + min_gain_db:
        return False
    record = {"experiment": exp_id, "pipeline": pipeline, "metrics": metrics,
              "promoted_at": dt.datetime.now().isoformat(timespec="seconds"),
              "git_commit": git_commit(),
              "previous": champ["experiment"] if champ else None}
    (CHAMPION_DIR / "champion.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (CHAMPION_DIR / "pipeline.yaml").write_text(yaml.safe_dump(pipeline, sort_keys=False))
    with open(CHAMPION_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def copy_champion_audio(src_dir: Path) -> None:
    for name in ("acoustic_guitar.wav", "non_acoustic_guitar.wav"):
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, CHAMPION_DIR / name)
