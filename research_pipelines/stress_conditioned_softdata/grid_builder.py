"""Point-table to regular 3D grid reconstruction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class GridData:
    """Regular 3D grid and property tensors."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    properties: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return grid tensor shape as ``(nx, ny, nz)``."""
        return (len(self.x), len(self.y), len(self.z))

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Return representative grid spacing as ``(dx, dy, dz)``."""
        return (
            float(self.metadata.get("dx", np.nan)),
            float(self.metadata.get("dy", np.nan)),
            float(self.metadata.get("dz", np.nan)),
        )


def build_regular_grid(df, config: Mapping[str, Any]) -> GridData:
    """Recover a regular ``[nx, ny, nz]`` grid from canonical point-table columns.

    Args:
        df: DataFrame with canonical ``x``, ``y``, ``z`` and property columns.
        config: Pipeline configuration.

    Returns:
        ``GridData`` with NumPy tensors shaped ``[nx, ny, nz]``.

    Raises:
        ValueError: If coordinate columns are missing or no valid grid exists.
    """
    for coord in ("x", "y", "z"):
        if coord not in df.columns:
            raise ValueError(f"Canonical coordinate column '{coord}' is missing")

    grid_cfg = config.get("grid", {})
    tol = float(grid_cfg.get("tolerance", 1.0e-6))
    duplicate_policy = str(grid_cfg.get("duplicate_policy", "first")).lower()

    x_raw = df["x"].to_numpy(dtype=np.float64, copy=False)
    y_raw = df["y"].to_numpy(dtype=np.float64, copy=False)
    z_raw = df["z"].to_numpy(dtype=np.float64, copy=False)
    x = np.unique(x_raw)
    y = np.unique(y_raw)
    z = np.unique(z_raw)

    nx, ny, nz = len(x), len(y), len(z)
    expected = nx * ny * nz
    if expected <= 0:
        raise ValueError("Empty coordinate grid inferred from input table")

    ix = _coord_to_index(x_raw, x, "x", tol)
    iy = _coord_to_index(y_raw, y, "y", tol)
    iz = _coord_to_index(z_raw, z, "z", tol)
    flat = np.ravel_multi_index((ix, iy, iz), dims=(nx, ny, nz))
    unique_flat, first_idx, counts = np.unique(flat, return_index=True, return_counts=True)

    duplicate_count = int(np.sum(counts - 1))
    missing_count = int(expected - len(unique_flat))
    if duplicate_count:
        LOGGER.warning("Detected %d duplicate grid rows; policy=%s", duplicate_count, duplicate_policy)
    if missing_count:
        LOGGER.warning("Detected %d missing grid nodes", missing_count)

    properties: Dict[str, np.ndarray] = {}
    for column in df.columns:
        if column in {"x", "y", "z"}:
            continue
        values = df[column].to_numpy(copy=False)
        if column == "facies":
            fill = -1
            arr = np.full(expected, fill, dtype=np.int16)
            arr[unique_flat] = values[first_idx].astype(np.int16, copy=False)
        else:
            arr = np.full(expected, np.nan, dtype=np.float32)
            numeric = values.astype(np.float32, copy=False)
            if duplicate_policy == "mean" and duplicate_count:
                sums = np.bincount(flat, weights=np.nan_to_num(numeric, nan=0.0), minlength=expected)
                valid_counts = np.bincount(flat, weights=np.isfinite(numeric).astype(np.float32), minlength=expected)
                valid = valid_counts > 0
                arr[valid] = (sums[valid] / valid_counts[valid]).astype(np.float32)
            else:
                arr[unique_flat] = numeric[first_idx]
        properties[column] = arr.reshape((nx, ny, nz))

    dx, regular_x = _spacing(x, tol)
    dy, regular_y = _spacing(y, tol)
    dz, regular_z = _spacing(z, tol)
    metadata = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_points": int(len(df)),
        "expected_grid_points": int(expected),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "regular_x": regular_x,
        "regular_y": regular_y,
        "regular_z": regular_z,
        "is_complete": missing_count == 0,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "coordinate_units": grid_cfg.get("coordinate_units", "unknown"),
        "z_positive_down": bool(grid_cfg.get("z_positive_down", True)),
        "input_order_monotonic": {
            "x": _is_monotonic(x_raw),
            "y": _is_monotonic(y_raw),
            "z": _is_monotonic(z_raw),
        },
    }
    LOGGER.info(
        "Recovered grid shape=(%d, %d, %d), spacing=(%.6g, %.6g, %.6g)",
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
    )
    return GridData(x=x, y=y, z=z, properties=properties, metadata=metadata)


def grid_coordinates(grid: GridData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate tensors ``X, Y, Z`` shaped like grid properties."""
    return np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")


def _coord_to_index(values: np.ndarray, unique: np.ndarray, name: str, tol: float) -> np.ndarray:
    idx = np.searchsorted(unique, values)
    idx = np.clip(idx, 0, len(unique) - 1)
    mismatch = np.abs(unique[idx] - values) > tol
    if np.any(mismatch):
        raise ValueError(f"Coordinate '{name}' contains values outside inferred grid tolerance")
    return idx.astype(np.int64, copy=False)


def _spacing(unique: np.ndarray, tol: float) -> tuple[float, bool]:
    if len(unique) < 2:
        return 0.0, True
    diffs = np.diff(unique)
    dx = float(np.median(diffs))
    regular = bool(np.all(np.abs(diffs - dx) <= max(tol, abs(dx) * tol)))
    return dx, regular


def _is_monotonic(values: np.ndarray) -> str:
    if len(values) < 2:
        return "trivial"
    diff = np.diff(values)
    if np.all(diff >= 0):
        return "increasing"
    if np.all(diff <= 0):
        return "decreasing"
    return "not_monotonic"
