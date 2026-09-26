"""Model catalog (configs/models.yaml), weight cache and architecture instantiation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from ..audio import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "configs" / "models.yaml"
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "model_configs"


def cache_dir() -> Path:
    d = Path(os.environ.get("ACOUSTIC_SEPARATOR_CACHE", Path.home() / ".cache" / "acoustic-separator"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ModelSpec:
    name: str
    arch: str  # MSST model_type
    repo: str | None  # Hugging Face repo id
    checkpoint: str  # file in repo, or local path
    config: str  # file in repo (hf:<file>) or file in configs/model_configs/
    stems: list[str]
    samplerate: int = 44100
    target_stem: str | None = None  # stem that carries (acoustic) guitar
    license: str = "unknown"
    notes: str = ""
    sha256: str | None = None
    revision: str | None = None
    extra: dict = field(default_factory=dict)


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, ModelSpec]:
    data = yaml.safe_load(Path(path).read_text())
    out = {}
    for name, d in data["models"].items():
        known = {k: v for k, v in d.items() if k in ModelSpec.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in ModelSpec.__dataclass_fields__}
        out[name] = ModelSpec(name=name, extra=extra, **known)
    return out


def _hf_download(repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                cache_dir=str(cache_dir() / "hf")))


def resolve_files(spec: ModelSpec) -> tuple[Path, Path]:
    """Download (once) checkpoint + config, verify/record checksum."""
    if spec.repo:
        ckpt = _hf_download(spec.repo, spec.checkpoint, spec.revision)
    else:
        ckpt = Path(spec.checkpoint)
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
    if spec.config.startswith("hf:"):
        cfg = _hf_download(spec.repo, spec.config[3:], spec.revision)
    else:
        cfg = MODEL_CONFIG_DIR / spec.config
    sums_path = cache_dir() / "checksums.json"
    sums = json.loads(sums_path.read_text()) if sums_path.exists() else {}
    key = str(ckpt)
    if key not in sums or sums[key]["mtime"] != ckpt.stat().st_mtime:
        sums[key] = {"sha256": file_sha256(ckpt), "mtime": ckpt.stat().st_mtime,
                     "model": spec.name, "size": ckpt.stat().st_size}
        sums_path.write_text(json.dumps(sums, indent=1))
    digest = sums[key]["sha256"]
    if spec.sha256 and spec.sha256 != digest:
        raise RuntimeError(f"checksum mismatch for {spec.name}: {digest} != {spec.sha256}")
    spec.extra["resolved_sha256"] = digest
    spec.extra["resolved_checkpoint"] = str(ckpt)
    spec.extra["resolved_config"] = str(cfg)
    return ckpt, cfg


def load_config(arch: str, cfg_path: Path):
    if arch == "htdemucs":
        from omegaconf import OmegaConf

        return OmegaConf.load(str(cfg_path))
    from ml_collections import ConfigDict

    return ConfigDict(yaml.load(Path(cfg_path).read_text(), Loader=yaml.FullLoader))


def build_model(arch: str, config):
    if arch == "mdx23c":
        from .msst.mdx23c_tfc_tdf_v3 import TFC_TDF_net

        return TFC_TDF_net(config)
    if arch == "htdemucs":
        from .msst.demucs4ht import get_model

        return get_model(config)
    if arch == "mel_band_roformer":
        from .msst.bs_roformer import MelBandRoformer

        return MelBandRoformer(**dict(config.model))
    if arch == "bs_roformer":
        from .msst.bs_roformer import BSRoformer

        return BSRoformer(**dict(config.model))
    if arch == "scnet":
        from .msst.scnet import SCNet

        return SCNet(**config.model)
    if arch == "bandit_v2":
        from .msst.bandit_v2.bandit import Bandit

        return Bandit(**config.kwargs)
    raise ValueError(f"unsupported arch {arch}")


def _state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "state", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def load_separator(name_or_spec, device=None):
    from ..inference import Separator, pick_device

    spec = name_or_spec if isinstance(name_or_spec, ModelSpec) else load_catalog()[name_or_spec]
    device = device or pick_device()
    ckpt, cfg_path = resolve_files(spec)
    config = load_config(spec.arch, cfg_path)
    for k, v in spec.extra.get("config_overrides", {}).items():
        node = config
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    model = build_model(spec.arch, config)
    sd = _state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{spec.name}: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    return Separator(spec, model, config, device)
