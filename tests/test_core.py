import numpy as np
import pytest

from acoustic_separator import augment, ensemble
from acoustic_separator.evaluation import evaluate_estimate, sdr, si_sdr

SR = 44100


def _sig(seed=0, seconds=3.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    g = np.stack([np.sin(2 * np.pi * 196 * t) * np.exp(-(t % 0.5) * 6)] * 2).astype(np.float32) * 0.3
    v = np.stack([np.sin(2 * np.pi * 880 * t)] * 2).astype(np.float32) * 0.2
    d = (rng.standard_normal((2, len(t))) * 0.05).astype(np.float32)
    return g, v, d


def test_sdr_ordering():
    g, v, d = _sig()
    mix = g + v + d
    assert sdr(g, g + 0.05 * v) > sdr(g, g + 0.3 * v) > sdr(g, mix)
    assert si_sdr(g, 0.5 * g) > 100  # scale invariant


def test_metrics_leakage_and_retention():
    g, v, d = _sig()
    mix = g + v + d
    m_clean = evaluate_estimate(g, g + 0.05 * v, mix, {"violin": v, "drums": d}, with_bss=False)
    m_dirty = evaluate_estimate(g, g + 0.5 * v, mix, {"violin": v, "drums": d}, with_bss=False)
    assert m_dirty["leak_violin_db"] > m_clean["leak_violin_db"]
    m_lossy = evaluate_estimate(g, 0.3 * g, mix, {"violin": v}, with_bss=False)
    assert m_lossy["target_retention"] < 0.2


@pytest.mark.parametrize("method", ["wave_mean", "wave_median", "mag_mean", "mag_median", "mask_mean"])
def test_ensemble_of_identical_estimates_is_identity(method):
    g, v, d = _sig()
    mix = g + v + d
    out = ensemble.METHODS[method]([g, g, g], mix=mix)
    assert sdr(g, out) > 25


def test_mastering_keeps_mixture_identity():
    g, v, d = _sig()
    rng = np.random.default_rng(1)
    stems = augment.mastering({"g": g, "v": v, "d": d}, SR, rng)
    mix = sum(stems.values())
    assert np.abs(mix).max() <= 1.0
    # gain is shared, so ratios between stems are preserved sample-wise
    ratio = stems["g"][0, 1000:2000] / (g[0, 1000:2000] + 1e-12)
    ratio_v = stems["v"][0, 1000:2000] / (v[0, 1000:2000] + 1e-12)
    np.testing.assert_allclose(ratio[np.abs(g[0, 1000:2000]) > 1e-3],
                               ratio_v[np.abs(g[0, 1000:2000]) > 1e-3], rtol=1e-3)


def test_wiener_refine_is_mix_consistent():
    g, v, d = _sig()
    mix = g + v + d
    est = ensemble.wiener_refine(g + 0.2 * v, mix)
    assert est.shape == mix.shape
    assert sdr(g, est) > sdr(g, mix)


def test_untrained_refiner_equals_mask_mean():
    import torch

    from acoustic_separator.refiner import MaskRefiner, features, istft

    g, v, d = _sig(seconds=2.0)
    mix = g + v + d
    pos = [g + 0.1 * v, g + 0.2 * d]
    t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
    feat, X, base = features(t(mix), [t(p) for p in pos], [t(v)])
    model = MaskRefiner(in_ch=feat.shape[1])
    m = model(feat, base).reshape(1, 2, *X.shape[-2:])
    np.testing.assert_allclose(m.detach().numpy(), base.clamp(1e-4, 1 - 1e-4).numpy(), atol=1e-5)
    y = istft(X * m, mix.shape[-1])
    assert y.shape == (1, 2, mix.shape[-1])


def test_pipeline_cascade_ops_with_fake_models(monkeypatch, tmp_path):
    from acoustic_separator import pipeline as P
    from acoustic_separator.models.loader import ModelSpec

    g, v, d = _sig(seconds=1.0)
    mix = g + v + d
    catalog = {
        "fake_gtr": ModelSpec("fake_gtr", "x", None, "c", "none", ["guitar"], target_stem="guitar"),
        "fake_multi": ModelSpec("fake_multi", "x", None, "c", "none", ["guitar", "violin", "other"],
                                target_stem="guitar"),
    }

    def fake_separate(self, model, audio, sr, params):
        # a perfect oracle that answers from the known components
        if model == "fake_gtr":
            return {"guitar": g * (np.abs(audio).sum() > 0)}, 0.0, False
        return {"guitar": g, "violin": v, "other": audio - g - v}, 0.0, False

    monkeypatch.setattr(P, "load_catalog", lambda: catalog)
    monkeypatch.setattr(P, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(P.PipelineRunner, "_separate", fake_separate)
    runner = P.PipelineRunner(use_cache=False)
    pipe = {"name": "t", "steps": [
        {"id": "a", "model": "fake_multi", "input": "mix", "take": "guitar+other"},
        {"id": "b", "model": "fake_multi", "input": "a", "take": "~violin"},
        {"id": "c", "model": "fake_gtr", "input": "b"},
        {"id": "e", "ensemble": ["c", "a.guitar"], "method": "wave_mean"},
        {"id": "r", "op": "sub", "a": "mix", "b": "e"},
    ], "output": "e"}
    res = runner.run(pipe, mix, 44100)
    np.testing.assert_allclose(res["values"]["a"], g + d, atol=1e-6)
    np.testing.assert_allclose(res["values"]["b"], g + d - v, atol=1e-6)  # input minus violin estimate
    np.testing.assert_allclose(res["output"], g, atol=1e-6)
    np.testing.assert_allclose(res["output"] + res["residual"], mix, atol=1e-6)


def test_fresh_and_cached_separation_are_bit_identical(monkeypatch, tmp_path):
    """Downstream cascade steps key their cache on the hash of the upstream output, so a
    freshly computed result must equal the later cache hit exactly (FLAC round trip)."""
    from acoustic_separator import pipeline as P
    from acoustic_separator.models.loader import ModelSpec

    g, v, d = _sig(seconds=1.0)
    spec = ModelSpec("fake", "x", None, "c", "none", ["guitar"], target_stem="guitar")
    spec.extra["resolved_sha256"] = "0"

    class FakeSep:
        def separate(self, audio, sr, params, progress=False):
            return {"guitar": audio * 0.3141592}

    monkeypatch.setattr(P, "load_catalog", lambda: {"fake": spec})
    monkeypatch.setattr(P, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(P, "resolve_files", lambda s: None)
    runner = P.PipelineRunner()
    monkeypatch.setattr(runner.pool, "get", lambda m: FakeSep())
    mix = (g + v + d).astype(np.float32)
    fresh, _, cached1 = runner._separate("fake", mix, 44100, P.InferenceParams())
    again, _, cached2 = runner._separate("fake", mix, 44100, P.InferenceParams())
    assert (cached1, cached2) == (False, True)
    assert P._hash_array(fresh["guitar"]) == P._hash_array(again["guitar"])
