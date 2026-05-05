import numpy as np

from research_pipelines.stress_conditioned_softdata.rock_physics_stress import generate_elastic_properties
from research_pipelines.stress_conditioned_softdata.stress_features import compute_stress_features


def _config():
    return {
        "project": {"device": "cpu", "dtype": "float32"},
        "schema": {
            "property_units": {"porosity": "percent", "oil_saturation": "percent", "brittleness_index": "percent"},
            "facies_map": {0: "mudstone", 1: "oil_layer"},
        },
        "stress": {
            "unit": "MPa",
            "normalize": "zscore",
            "effective_pressure": {"no_stress_mpa": 20.0, "stress_to_pressure_scale": 0.6, "min_mpa": 2.0, "max_mpa": 80.0},
            "weak_correction": {},
        },
        "rock_physics": {
            "critical_porosity": 0.40,
            "coordination_number": 7.0,
            "friction_coefficient": 1.0,
            "porosity_clip": [0.01, 0.40],
            "oil_saturation_clip": [0.0, 1.0],
            "minerals": {"oil_layer": {"quartz": 0.8, "clay": 0.2}, "mudstone": {"quartz": 0.4, "clay": 0.6}, "default": {"quartz": 0.7, "clay": 0.3}},
            "fluids": {"brine": "brine", "hydrocarbon": "oil_medium", "fluid_mixing": "reuss"},
            "facies_parameters": {},
            "elastic_impedance_angles": [12],
        },
    }


def test_generate_elastic_properties_positive_outputs():
    shape = (3, 2, 5)
    props = {
        "porosity": np.full(shape, 18.0, dtype=np.float32),
        "oil_saturation": np.full(shape, 55.0, dtype=np.float32),
        "brittleness_index": np.full(shape, 50.0, dtype=np.float32),
        "stress": np.full(shape, 30.0, dtype=np.float32),
        "SH1": np.full(shape, 34.0, dtype=np.float32),
        "SH2": np.full(shape, 27.0, dtype=np.float32),
        "facies": np.ones(shape, dtype=np.int16),
    }
    stress, _ = compute_stress_features(props, _config())
    result = generate_elastic_properties(props, _config(), stress_features=stress, mode="strong")
    assert result.tensors["Vp"].shape == shape
    assert float(result.tensors["Vp"].min()) > 0.0
    assert float(result.tensors["Vs"].min()) > 0.0
    assert "EI_12" in result.tensors
