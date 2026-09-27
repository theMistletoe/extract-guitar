#!/usr/bin/env python
"""Assemble reports/final_report.md from the Champion, results.csv and the hand-written
conclusions (reports/conclusions.md, sections: What worked / What failed / Remaining
artifacts / Known failure modes / Why the final model was selected). The chronological
lab notebook stays in reports/analysis.md and is linked, not inlined."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator import tracking  # noqa: E402


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    champ = tracking.current_champion()
    rows = tracking.read_results()
    if not champ:
        raise SystemExit("no champion yet")
    exp = champ["experiment"]
    m = json.loads((ROOT / "experiments" / exp / "metrics.json").read_text())
    v = m["validation"]
    t = m.get("target", {})
    cfg = (ROOT / "experiments" / exp / "config.yaml").read_text()
    import yaml

    c = yaml.safe_load(cfg)
    models = c.get("models", {})
    hw = c.get("reproducibility", {}).get("hardware", {})
    steps = champ["pipeline"]["steps"]
    pipe_txt = "\n".join(f"  {i + 1}. {json.dumps(s, ensure_ascii=False)}" for i, s in enumerate(steps))
    base = [r for r in rows if r["strategy"] == "A"]
    best_base = max(base, key=lambda r: float(r["sdr"] or "nan")) if base else None
    conclusions = ROOT / "reports" / "conclusions.md"
    lines = [
        "# Final report — acoustic guitar extraction", "",
        "```text",
        f"Best architecture:   {champ['pipeline'].get('description', champ['pipeline']['name'])}",
        f"Best checkpoint(s):  " + "; ".join(f"{k} ({v.get('repo')}:{v.get('checkpoint')}, sha256 {(v.get('sha256') or '')[:16]})"
                                             for k, v in models.items()),
        f"Pipeline:            {champ['pipeline']['name']} (experiment {exp})",
        f"Dataset:             datasets/validation — 41 clips x 12 s (19 real multitrack + 22 scenario), exact GT",
        f"Validation SDR:      {f(v.get('sdr_mean'))} dB mean / {f(v.get('sdr_median'))} dB median "
        f"(real multitracks {f(v.get('sdr_ms_mean'))}, synthetic {f(v.get('sdr_syn_mean'))})",
        f"Validation SI-SDR:   {f(v.get('si_sdr_mean'))} dB (SDRi {f(v.get('sdri_mean'))} dB)",
        f"SIR / SAR (BSS):     {f(v.get('sir_bss_mean'))} / {f(v.get('sar_bss_mean'))} dB",
        f"Leakage:             {f(v.get('leakage_db_mean'))} dB excess energy; target retention "
        f"{f(v.get('target_retention_mean'), 3)}",
        f"Runtime:             target song {f(t.get('runtime_s'), 0)} s for 132 s of audio (search setting); "
        f"validation {f(v.get('runtime_s'), 0)} s",
        f"Hardware:            {hw.get('processor', platform.machine())}, {hw.get('cpu_count')} cores, "
        f"torch {hw.get('torch')}, CUDA={hw.get('cuda')}",
        f"Number of experiments: {len(rows)}",
        "```", "",
    ]
    if best_base:
        lines += [f"Best single pretrained model (Strategy A): `{best_base['experiment']}` — "
                  f"validation SDR {f(best_base['sdr'])} dB; Champion improves on it by "
                  f"{float(v.get('sdr_mean')) - float(best_base['sdr']):+.2f} dB.", ""]
    lines += ["## Champion pipeline", "", "```text", pipe_txt, "```", "",
              "## All experiments", "", "See `docs/experiments.md` (full table and per-experiment "
              "Hypothesis / Change / Result / Conclusion / Next), `experiments/results.csv`, and the "
              "chronological lab notebook `reports/analysis.md`.", ""]
    if conclusions.exists():
        lines += [conclusions.read_text().strip(), ""]
    out = ROOT / "reports" / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
