import numpy as np
import pandas as pd

from research_pipelines.stress_conditioned_softdata.data_schema import canonicalize_dataframe, resolve_schema
from research_pipelines.stress_conditioned_softdata.grid_builder import build_regular_grid


def _config():
    return {
        "columns": {
            "x": ["x"],
            "y": ["y"],
            "z": ["z"],
            "porosity": ["PORO"],
            "oil_saturation": ["SOG"],
            "brittleness_index": ["BI"],
            "SH1": ["sh1"],
            "SH2": ["sh2"],
            "facies": ["OilPhase"],
        },
        "schema": {
            "required_properties": ["porosity", "oil_saturation", "brittleness_index", "SH1", "SH2", "facies"],
            "optional_properties": [],
            "facies_map": {0: "mudstone", 1: "oil_layer"},
        },
        "grid": {"tolerance": 1.0e-6, "duplicate_policy": "first"},
    }


def test_build_regular_grid_detects_missing_and_duplicates():
    x, y, z = np.arange(2), np.arange(3), np.arange(4)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    df = pd.DataFrame(
        {
            "x": X.ravel(),
            "y": Y.ravel(),
            "z": Z.ravel(),
            "PORO": np.full(X.size, 20.0),
            "SOG": np.full(X.size, 60.0),
            "BI": np.full(X.size, 50.0),
            "sh1": np.full(X.size, 35.0),
            "sh2": np.full(X.size, 28.0),
            "OilPhase": np.ones(X.size, dtype=int),
        }
    )
    df = pd.concat([df.iloc[:-1], df.iloc[[0]]], ignore_index=True)
    schema = resolve_schema(df, _config())
    grid = build_regular_grid(canonicalize_dataframe(df, schema), _config())
    assert grid.shape == (2, 3, 4)
    assert grid.metadata["missing_count"] == 1
    assert grid.metadata["duplicate_count"] == 1
    assert grid.properties["porosity"].shape == (2, 3, 4)
