"""Publication-oriented static figures for 3D soft-data volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def save_slice_panel(
    volumes: Mapping[str, Any],
    output_path: str | Path,
    axis: int = 0,
    index: int | None = None,
    dpi: int = 300,
    max_items: int = 9,
) -> None:
    """Save a multi-volume central-slice panel."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    items = list(volumes.items())[:max_items]
    if not items:
        return
    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (name, volume) in zip(axes.ravel(), items):
        arr = _to_numpy(volume)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        slc = _slice(arr, axis=axis, index=index)
        im = ax.imshow(slc.T, origin="lower", aspect="auto", cmap=_cmap(name))
        ax.set_title(name)
        ax.set_xlabel("trace")
        ax.set_ylabel("sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_difference_panel(
    baseline: Mapping[str, Any],
    stress: Mapping[str, Any],
    keys: Sequence[str],
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """Save stress-aware minus baseline central-slice differences."""
    diff = {}
    for key in keys:
        if key in baseline and key in stress:
            diff[f"{key}_stress_minus_baseline"] = _to_numpy(stress[key]) - _to_numpy(baseline[key])
    save_slice_panel(diff, output_path, dpi=dpi)


def save_crossplot(
    x,
    y,
    facies,
    output_path: str | Path,
    xlabel: str,
    ylabel: str,
    max_points: int = 20000,
    dpi: int = 300,
) -> None:
    """Save a facies-colored crossplot for soft-data interpretation."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = _to_numpy(x).ravel()
    y = _to_numpy(y).ravel()
    facies = _to_numpy(facies).ravel()
    n = min(x.size, y.size, facies.size)
    x, y, facies = x[:n], y[:n], facies[:n]
    valid = np.isfinite(x) & np.isfinite(y) & (facies >= 0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return
    if idx.size > max_points:
        rng = np.random.default_rng(20260505)
        idx = rng.choice(idx, size=max_points, replace=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(x[idx], y[idx], c=facies[idx], s=4, alpha=0.35, cmap="tab10", edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.colorbar(sc, ax=ax, label="facies code")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _slice(arr: np.ndarray, axis: int, index: int | None) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    axis = int(axis)
    if index is None:
        index = arr.shape[axis] // 2
    if axis == 0:
        return arr[index, :, :]
    if axis == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cmap(name: str) -> str:
    lname = name.lower()
    if "facies" in lname or "mask" in lname:
        return "tab10"
    if "seismic" in lname or "reflectivity" in lname or "gradient" in lname:
        return "seismic"
    if "difference" in lname or "minus" in lname:
        return "coolwarm"
    return "viridis"
