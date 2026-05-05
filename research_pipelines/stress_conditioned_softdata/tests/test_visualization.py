import numpy as np

from research_pipelines.stress_conditioned_softdata.visualization import save_crossplot, save_slice_panel, save_volume_orthoslices


def test_save_slice_panel_accepts_2d_maps(tmp_path):
    output = tmp_path / "horizon.png"
    save_slice_panel({"top": np.ones((4, 5), dtype=np.float32)}, output, dpi=50)
    assert output.exists()


def test_save_crossplot_crops_mismatched_lengths(tmp_path):
    output = tmp_path / "crossplot.png"
    save_crossplot(
        np.arange(12, dtype=np.float32),
        np.arange(10, dtype=np.float32),
        np.ones(11, dtype=np.int16),
        output,
        "x",
        "y",
        dpi=50,
    )
    assert output.exists()


def test_save_volume_orthoslices_accepts_3d_volumes(tmp_path):
    output = tmp_path / "orthoslices.png"
    save_volume_orthoslices({"AI": np.ones((4, 5, 6), dtype=np.float32)}, output, dpi=50)
    assert output.exists()
