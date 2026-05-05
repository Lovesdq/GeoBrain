import torch

from research_pipelines.stress_conditioned_softdata.seismic_softdata import generate_seismic_softdata


def test_generate_seismic_softdata_shapes():
    nx, ny, nz = 4, 3, 8
    z = torch.linspace(0.0, 1.0, nz).reshape(1, 1, nz)
    elastic = {
        "Vp": 2600.0 + 500.0 * z.expand(nx, ny, nz),
        "Vs": 1400.0 + 250.0 * z.expand(nx, ny, nz),
        "density": 2.15 + 0.15 * z.expand(nx, ny, nz),
    }
    config = {
        "seismic": {
            "angles": [12, 24, 36],
            "wavelet_frequencies_hz": [30, 25, 20],
            "dt_s": 0.001,
            "reflectivity_method": "shuey",
            "chunk_size_inline": 2,
            "noise": {"enabled": True, "gaussian_std_fraction": 0.01, "seed": 1},
        }
    }
    result = generate_seismic_softdata(elastic, config)
    assert result.tensors["reflectivity"].shape == (3, nx, ny, nz - 1)
    assert result.tensors["prestack_seismic"].shape == (3, nx, ny, nz - 1)
    assert result.tensors["poststack_seismic"].shape == (nx, ny, nz - 1)
    assert result.tensors["AVO_gradient"].shape == (nx, ny, nz - 1)
    assert "prestack_seismic_noisy" in result.tensors
