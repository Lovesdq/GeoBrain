"""Quality-control statistics and figures for grid properties."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .data_schema import save_json
from .grid_builder import GridData


LOGGER = logging.getLogger(__name__)


def run_qc(grid: GridData, config: Mapping[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    """Compute QC statistics, warnings, and optional figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_cfg = config.get("qc", {})
    percentiles = qc_cfg.get("percentiles", [1, 5, 50, 95, 99])
    ranges = qc_cfg.get("physical_ranges", {})
    units = config.get("schema", {}).get("property_units", {})

    report: Dict[str, Any] = {"grid": grid.metadata, "properties": {}, "warnings": []}
    for name, arr in grid.properties.items():
        if name == "facies":
            values, counts = np.unique(arr[arr >= 0], return_counts=True)
            report["properties"][name] = {
                "classes": {str(int(v)): int(c) for v, c in zip(values, counts)}
            }
            continue
        stats = property_stats(arr, percentiles)
        warnings = property_warnings(name, arr, ranges.get(name), units.get(name))
        report["properties"][name] = {**stats, "warnings": warnings}
        report["warnings"].extend(warnings)
        for warning in warnings:
            LOGGER.warning(warning)

    save_json(report, output_dir / "qc_report.json")
    if bool(qc_cfg.get("figures", True)):
        save_qc_figures(grid, output_dir / "figures", dpi=int(config.get("export", {}).get("figures_dpi", 300)))
    return report


def property_stats(arr: np.ndarray, percentiles) -> Dict[str, Any]:
    """Return finite-value summary statistics for one property tensor."""
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    values = arr[finite]
    if values.size == 0:
        return {
            "finite_count": 0,
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
        }
    pct = np.percentile(values, percentiles)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    outlier_mask = (values < q1 - 3.0 * iqr) | (values > q3 + 3.0 * iqr)
    return {
        "finite_count": int(values.size),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {str(p): float(v) for p, v in zip(percentiles, pct)},
        "iqr_outlier_count": int(np.sum(outlier_mask)),
    }


def property_warnings(
    name: str,
    arr: np.ndarray,
    physical_range,
    unit: str | None = None,
) -> list[str]:
    """Return NaN/Inf and configured physical-range warnings."""
    warnings: list[str] = []
    if np.isnan(arr).any():
        warnings.append(f"{name}: contains {int(np.isnan(arr).sum())} NaN values")
    if np.isinf(arr).any():
        warnings.append(f"{name}: contains {int(np.isinf(arr).sum())} Inf values")
    if physical_range is not None:
        values = _range_units(np.asarray(arr, dtype=np.float64), unit)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = float(physical_range[0]), float(physical_range[1])
            n_bad = int(np.sum((finite < lo) | (finite > hi)))
            if n_bad:
                warnings.append(
                    f"{name}: {n_bad} finite values outside configured range [{lo}, {hi}] after unit normalization"
                )
    return warnings


def save_qc_figures(grid: GridData, output_dir: str | Path, dpi: int = 300) -> None:
    """Save compact histogram and central-slice figures for each property."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from .visualization import apply_publication_style

    apply_publication_style(dpi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ix = max(0, grid.shape[0] // 2)
    for name, arr in grid.properties.items():
        if name == "facies":
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.25))
        axes[0].hist(finite.ravel(), bins=70, color="#3b6ea8", alpha=0.88, linewidth=0)
        axes[0].set_title(f"{name} histogram")
        axes[0].set_xlabel(name)
        axes[0].set_ylabel("count")
        im = axes[1].imshow(arr[ix].T, origin="lower", aspect="auto", cmap="viridis")
        axes[1].set_title(f"{name} inline {ix}")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.025)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.25)
        fig.savefig(output_dir / f"{name}_qc.png", dpi=dpi)
        plt.close(fig)


def _range_units(values: np.ndarray, unit: str | None) -> np.ndarray:
    if unit and unit.lower() in {"percent", "%"}:
        return values / 100.0
    return values
