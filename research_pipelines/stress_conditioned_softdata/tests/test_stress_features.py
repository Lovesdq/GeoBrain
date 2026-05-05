import numpy as np

from research_pipelines.stress_conditioned_softdata.stress_features import compute_geomechanics_features, compute_stress_features


def test_stress_features_values_and_warnings():
    props = {
        "stress": np.array([[[24.0, 23.0]]], dtype=np.float32),
        "SH1": np.array([[[30.0, 20.0]]], dtype=np.float32),
        "SH2": np.array([[[20.0, 25.0]]], dtype=np.float32),
    }
    config = {"stress": {"unit": "MPa", "normalize": "zscore"}, "schema": {"property_units": {"stress": "MPa"}}}
    features, warnings = compute_stress_features(props, config)
    assert np.allclose(features["differential_stress"][0, 0, 0], 10.0)
    assert np.allclose(features["stress_ratio"][0, 0, 0], 1.5)
    assert np.allclose(features["mean_stress_proxy"][0, 0, 0], 25.0)
    assert np.allclose(features["stress_MPa"][0, 0, 0], 24.0)
    assert any("SH1 < SH2" in warning for warning in warnings)


class _Grid:
    z = np.array([100.0, 200.0], dtype=np.float32)


def test_geomechanics_features_use_depth_gradient_when_columns_missing():
    props = {"porosity": np.ones((1, 1, 2), dtype=np.float32)}
    config = {
        "geomechanics": {
            "pore_pressure_gradient_mpa_per_m": 0.01,
            "vertical_stress_gradient_mpa_per_m": 0.025,
            "reference_datum_m": 0.0,
            "effective_pressure_clip_mpa": [2.0, 80.0],
        }
    }
    features, warnings, metadata = compute_geomechanics_features(_Grid(), props, config)
    assert np.allclose(features["pore_pressure_MPa"][0, 0], [1.0, 2.0])
    assert np.allclose(features["vertical_stress_Sv_MPa"][0, 0], [2.5, 5.0])
    assert np.allclose(features["effective_pressure_physical_MPa"][0, 0], [2.0, 3.0])
    assert metadata["pore_pressure_source"] == "hydrostatic_gradient"
    assert warnings


def test_geomechanics_features_prefer_columns_when_present():
    props = {
        "porosity": np.ones((1, 1, 2), dtype=np.float32),
        "pore_pressure": np.array([[[3.0, 4.0]]], dtype=np.float32),
        "vertical_stress": np.array([[[8.0, 10.0]]], dtype=np.float32),
    }
    config = {
        "schema": {"property_units": {"pore_pressure": "MPa", "vertical_stress": "MPa"}},
        "geomechanics": {"effective_pressure_clip_mpa": [2.0, 80.0]},
    }
    features, warnings, metadata = compute_geomechanics_features(_Grid(), props, config)
    assert np.allclose(features["effective_pressure_physical_MPa"][0, 0], [5.0, 6.0])
    assert metadata["pore_pressure_source"] == "column"
    assert metadata["vertical_stress_source"] == "column"
    assert warnings == []
