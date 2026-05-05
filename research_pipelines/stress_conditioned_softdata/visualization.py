"""Publication-oriented static figures for 3D soft-data volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def apply_publication_style(dpi: int = 600) -> None:
    """Apply a compact Nature-style matplotlib theme."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


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

    apply_publication_style(dpi)
    items = list(volumes.items())[:max_items]
    if not items:
        return
    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.05 * nrows), squeeze=False)
    for ax, (name, volume) in zip(axes.ravel(), items):
        arr = _to_numpy(volume)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        slc = _slice(arr, axis=axis, index=index)
        cmap = _colormap(name)
        vmin, vmax = _robust_limits(slc, name)
        im = ax.imshow(slc.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        _format_image_axis(ax, name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.tight_layout(pad=0.2, w_pad=0.4, h_pad=0.6)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_volume_orthoslices(
    volumes: Mapping[str, Any],
    output_path: str | Path,
    dpi: int = 600,
    max_items: int = 6,
) -> None:
    """Save a high-resolution orthoslice plate for selected 3D volumes."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    apply_publication_style(dpi)
    items = list(volumes.items())[:max_items]
    if not items:
        return
    fig, axes = plt.subplots(len(items), 3, figsize=(7.1, 1.85 * len(items)), squeeze=False)
    for row, (name, volume) in enumerate(items):
        arr = _to_numpy(volume)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim <= 2:
            panels = [(arr, "map"), (None, ""), (None, "")]
        else:
            panels = [
                (_slice(arr, 0, None), "inline"),
                (_slice(arr, 1, None), "crossline"),
                (_slice(arr, 2, None), "depth slice"),
            ]
        cmap = _colormap(name)
        valid = arr[np.isfinite(arr)] if np.asarray(arr).size else np.array([])
        vmin, vmax = _robust_limits(valid, name)
        last_im = None
        for col, (panel, label) in enumerate(panels):
            ax = axes[row, col]
            if panel is None:
                ax.axis("off")
                continue
            last_im = ax.imshow(panel.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            title = f"{name} - {label}" if col == 0 or arr.ndim <= 2 else label
            _format_image_axis(ax, title)
        if last_im is not None:
            fig.colorbar(last_im, ax=axes[row, :], fraction=0.018, pad=0.01)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.98, bottom=0.04, wspace=0.22, hspace=0.45)
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

    apply_publication_style(dpi)
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
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    cmap = plt.get_cmap("tab10")
    codes = np.unique(facies[idx]).astype(int)
    for color_idx, code in enumerate(codes):
        code_idx = idx[facies[idx] == code]
        ax.scatter(
            x[code_idx],
            y[code_idx],
            s=3.0,
            alpha=0.28,
            color=cmap(color_idx % 10),
            edgecolors="none",
            rasterized=True,
            label=f"facies {code}",
        )
        ax.scatter(
            np.nanmedian(x[code_idx]),
            np.nanmedian(y[code_idx]),
            s=26,
            marker="D",
            color=cmap(color_idx % 10),
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )
    corr = _corrcoef(x[idx], y[idx])
    if np.isfinite(corr):
        ax.text(0.02, 0.98, f"r = {corr:.2f}", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.18, linestyle="-", linewidth=0.4)
    ax.legend(loc="best", frameon=False, markerscale=2.0, handletextpad=0.2)
    _despine(ax)
    fig.tight_layout(pad=0.25)
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


def _colormap(name: str):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(_cmap(name)).copy()
    cmap.set_bad("#d9d9d9")
    return cmap


def _cmap(name: str) -> str:
    lname = name.lower()
    if "facies" in lname or "mask" in lname:
        return "tab10"
    if "seismic" in lname or "reflectivity" in lname or "gradient" in lname:
        return "seismic"
    if "difference" in lname or "minus" in lname:
        return "coolwarm"
    if "distance" in lname or "sdf" in lname:
        return "coolwarm"
    if "probability" in lname:
        return "magma"
    return "viridis"


def _robust_limits(arr: np.ndarray, name: str) -> tuple[float | None, float | None]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or "facies" in name.lower() or "mask" in name.lower():
        return None, None
    lname = name.lower()
    if "seismic" in lname or "reflectivity" in lname or "gradient" in lname or "minus" in lname or "distance" in lname:
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        return -vmax, vmax
    lo, hi = np.nanpercentile(finite, [2.0, 98.0])
    if float(lo) == float(hi):
        return None, None
    return float(lo), float(hi)


def _format_image_axis(ax, title: str) -> None:
    ax.set_title(title, pad=2)
    ax.set_xlabel("trace")
    ax.set_ylabel("sample")
    ax.tick_params(direction="out")
    _despine(ax)


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])
