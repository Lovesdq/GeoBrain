"""Export helpers for NumPy, Torch, VTK and optional SEG-Y outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data_schema import save_json
from .grid_builder import GridData


LOGGER = logging.getLogger(__name__)


def save_npz(path: str | Path, tensors: Mapping[str, Any]) -> None:
    """Save tensors to compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: to_numpy(v) for k, v in tensors.items()})


def save_torch(path: str | Path, tensors: Mapping[str, Any]) -> None:
    """Save tensors to a PyTorch ``.pt`` file on CPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: to_torch_cpu(v) for k, v in tensors.items()}, path)


def export_all(
    grid: GridData,
    groups: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    """Export all requested output groups."""
    output_dir = Path(output_dir)
    export_cfg = config.get("export", {})
    for group, tensors in groups.items():
        if bool(export_cfg.get("save_npz", True)):
            save_npz(output_dir / f"{group}.npz", tensors)
        if bool(export_cfg.get("save_torch", True)):
            save_torch(output_dir / f"{group}.pt", tensors)
        if bool(export_cfg.get("save_vtk", False)):
            for name, value in tensors.items():
                arr = to_numpy(value)
                if arr.ndim == 3:
                    write_vtk_structured_points(output_dir / "vtk" / f"{group}_{name}.vtk", arr, grid)
    save_json({"grid": grid.metadata, "groups": list(groups.keys())}, output_dir / "export_manifest.json")


def write_vtk_structured_points(path: str | Path, volume: np.ndarray, grid: GridData, scalar_name: str = "value") -> None:
    """Write a legacy ASCII VTK structured-points scalar volume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(volume, dtype=np.float32)
    if arr.shape != grid.shape:
        raise ValueError(f"VTK volume shape {arr.shape} does not match grid shape {grid.shape}")
    dx, dy, dz = grid.spacing
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("GeoBrain stress-conditioned soft data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {grid.shape[0]} {grid.shape[1]} {grid.shape[2]}\n")
        f.write(f"ORIGIN {float(grid.x[0])} {float(grid.y[0])} {float(grid.z[0])}\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"POINT_DATA {arr.size}\n")
        f.write(f"SCALARS {scalar_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        flat = arr.ravel(order="C")
        for start in range(0, flat.size, 8):
            f.write(" ".join(f"{v:.7g}" for v in flat[start:start + 8]) + "\n")


def try_write_segy(path: str | Path, volume: Any, dt_microseconds: float = 1000.0) -> bool:
    """Write a 3D seismic cube as SEG-Y when segyio is installed."""
    try:
        from geobrain.io import write_segy
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("GeoBrain SEG-Y writer unavailable: %s", exc)
        return False
    try:
        write_segy(str(path), to_numpy(volume), dt=dt_microseconds)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SEG-Y export failed; NPZ remains the stable fallback: %s", exc)
        return False
    return True


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch or array-like values to NumPy."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_torch_cpu(value: Any) -> torch.Tensor:
    """Convert values to CPU torch tensors."""
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.as_tensor(value)
