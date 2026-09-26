#!/usr/bin/env python
"""Download (once) and checksum every model in configs/models.yaml.

    python scripts/download_models.py            # all models
    python scripts/download_models.py sw6 mega_acoustic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acoustic_separator.models.loader import cache_dir, load_catalog, load_separator, resolve_files  # noqa: E402


def main(names: list[str]) -> int:
    cat = load_catalog()
    names = names or list(cat)
    ok = True
    for name in names:
        spec = cat[name]
        try:
            if spec.arch == "demucs_pkg":
                load_separator(name)
            else:
                resolve_files(spec)
            print(f"{name:20s} OK  sha256={spec.extra.get('resolved_sha256', '')[:16]}  licence: {spec.license}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{name:20s} FAILED: {e}")
    sums = cache_dir() / "checksums.json"
    if sums.exists():
        print(f"checksums: {sums} ({len(json.loads(sums.read_text()))} files)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
