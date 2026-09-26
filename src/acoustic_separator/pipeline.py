"""Declarative separation pipelines (direct, two-stage, cascade, ensemble).

A pipeline YAML looks like::

    name: two_stage_example
    steps:
      - {id: g,   model: some_guitar_model, input: mix}            # Strategy A / stage 1
      - {id: ag,  model: acoustic_vs_electric, input: g}           # Strategy B stage 2
      - {id: ens, ensemble: [g, ag], method: mag_mean, weights: [1, 1]}   # Strategy D
      - {id: out, post: wiener, input: ens, iterations: 1}
      - {id: rest, op: sub, a: mix, b: vocals_step.vocals}          # cascade residual
    output: out

References: ``mix`` is the input; ``<id>`` is a step's primary output (for a separator:
its ``target_stem``); ``<id>.<stem>`` selects another stem of a separator step.
Separator results are cached on disk keyed by (checkpoint hash, params, input hash).
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

from . import ensemble as ens
from .inference import InferenceParams
from .models.loader import cache_dir, load_catalog, load_separator, resolve_files

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "configs" / "pipelines"


def load_pipeline(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = PIPELINE_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("name", p.stem)
    return cfg


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


class ModelPool:
    """Keeps at most ``capacity`` models in memory."""

    def __init__(self, capacity: int = 2, device=None):
        self.capacity = capacity
        self.device = device
        self.models: OrderedDict = OrderedDict()

    def get(self, name: str):
        if name in self.models:
            self.models.move_to_end(name)
            return self.models[name]
        while len(self.models) >= self.capacity:
            self.models.popitem(last=False)
        sep = load_separator(name, device=self.device)
        self.models[name] = sep
        return sep


class PipelineRunner:
    def __init__(self, pool: ModelPool | None = None, use_cache: bool = True, progress: bool = False):
        self.pool = pool or ModelPool()
        self.catalog = load_catalog()
        self.use_cache = use_cache
        self.progress = progress
        self.stem_cache = cache_dir() / "stems"
        self.stem_cache.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}

    # -- separator step with disk cache -------------------------------------------------
    def _separate(self, model: str, audio: np.ndarray, sr: int, params: InferenceParams):
        spec = self.catalog[model]
        resolve_files(spec)
        key_src = json.dumps({"model": model, "sha": spec.extra["resolved_sha256"],
                              "params": params.to_dict(), "sr": sr,
                              "overrides": spec.extra.get("config_overrides", {}),
                              "input": _hash_array(audio)}, sort_keys=True)
        key = hashlib.sha1(key_src.encode()).hexdigest()[:20]
        folder = self.stem_cache / key
        if self.use_cache and (folder / "done").exists():
            return {p.stem: np.load(p) for p in folder.glob("*.npy")}, 0.0, True
        sep = self.pool.get(model)
        t = time.perf_counter()
        out = sep.separate(audio, sr, params, progress=self.progress)
        dt = time.perf_counter() - t
        if self.use_cache:
            folder.mkdir(parents=True, exist_ok=True)
            for k, v in out.items():
                np.save(folder / f"{k}.npy", v)
            (folder / "meta.json").write_text(key_src)
            (folder / "done").write_text(str(dt))
        return out, dt, False

    def run(self, pipeline: dict, mix: np.ndarray, sr: int) -> dict:
        """Returns {"output": acoustic, "values": all step outputs, "runtime": seconds, ...}."""
        vals: dict[str, np.ndarray] = {"mix": mix}
        total = 0.0
        step_times = {}

        def ref(name: str) -> np.ndarray:
            if name not in vals:
                raise KeyError(f"unknown reference {name!r}; have {sorted(vals)}")
            return vals[name]

        for step in pipeline["steps"]:
            sid = step["id"]
            if "model" in step:
                params = InferenceParams(**step.get("params", {}))
                out, dt, cached = self._separate(step["model"], ref(step.get("input", "mix")), sr, params)
                spec = self.catalog[step["model"]]
                primary = step.get("take", spec.target_stem or spec.stems[0])
                for k, v in out.items():
                    vals[f"{sid}.{k}"] = v
                if primary.startswith("~"):  # complement: input minus the named stem
                    vals[sid] = ref(step.get("input", "mix")) - out[primary[1:]]
                elif "+" in primary:
                    vals[sid] = sum(out[p] for p in primary.split("+"))
                else:
                    vals[sid] = out[primary]
                step_times[sid] = {"seconds": dt, "cached": cached}
                total += dt
            elif "ensemble" in step:
                fn = ens.METHODS[step["method"]]
                kw = {k: v for k, v in step.items() if k not in ("id", "ensemble", "method")}
                vals[sid] = fn([ref(r) for r in step["ensemble"]], mix=mix, sr=sr, **kw)
            elif "post" in step:
                fn = ens.POSTPROCESS[step["post"]]
                kw = {k: v for k, v in step.items() if k not in ("id", "post", "input", "mix_ref")}
                vals[sid] = fn(ref(step["input"]), ref(step.get("mix_ref", "mix")), **kw)
            elif step.get("op") == "sub":
                vals[sid] = ref(step["a"]) - ref(step["b"])
            elif step.get("op") == "sum":
                vals[sid] = np.sum([ref(r) for r in step["inputs"]], axis=0)
            elif step.get("op") == "gain":
                vals[sid] = ref(step["input"]) * float(step["gain"])
            else:
                raise ValueError(f"cannot interpret step {step}")
        out = vals[pipeline.get("output", pipeline["steps"][-1]["id"])].astype(np.float32)
        return {"output": out, "residual": (mix - out).astype(np.float32), "values": vals,
                "runtime": total, "step_times": step_times}


def describe(pipeline: dict) -> dict:
    """Checkpoint provenance for every model used in a pipeline."""
    cat = load_catalog()
    models = {}
    for s in pipeline["steps"]:
        if "model" in s:
            spec = cat[s["model"]]
            try:
                resolve_files(spec)
            except Exception as e:  # noqa: BLE001
                spec.extra["resolve_error"] = str(e)
            models[s["model"]] = {"repo": spec.repo, "checkpoint": spec.checkpoint,
                                  "sha256": spec.extra.get("resolved_sha256"), "arch": spec.arch,
                                  "license": spec.license}
    return {"pipeline": copy.deepcopy(pipeline), "models": models}
