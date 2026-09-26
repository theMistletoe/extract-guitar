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
