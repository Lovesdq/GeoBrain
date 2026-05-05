import numpy as np

from research_pipelines.stress_conditioned_softdata.stress_features import compute_stress_features


def test_stress_features_values_and_warnings():
    props = {
        "SH1": np.array([[[30.0, 20.0]]], dtype=np.float32),
        "SH2": np.array([[[20.0, 25.0]]], dtype=np.float32),
    }
    config = {"stress": {"unit": "MPa", "normalize": "zscore"}}
    features, warnings = compute_stress_features(props, config)
    assert np.allclose(features["differential_stress"][0, 0, 0], 10.0)
    assert np.allclose(features["stress_ratio"][0, 0, 0], 1.5)
    assert np.allclose(features["mean_stress_proxy"][0, 0, 0], 25.0)
    assert any("SH1 < SH2" in warning for warning in warnings)
