"""Paper-oriented ablation experiments and metrics."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .data_schema import save_json
from .grid_builder import GridData
from .horizon_softdata import generate_horizon_softdata
from .rock_physics_stress import generate_elastic_properties
from .seismic_softdata import generate_seismic_softdata
from .uncertainty import run_uncertainty
from .visualization import save_crossplot, save_difference_panel


LOGGER = logging.getLogger(__name__)


def run_ablation(
    grid: GridData,
    config: Mapping[str, Any],
    stress_features: Mapping[str, np.ndarray],
    output_dir: str | Path,
    precomputed: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Run Exp-0 through Exp-6 and save metrics/figures/log metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    start = time.perf_counter()
    precomputed = precomputed or {}

    rows.extend(_exp0_hard_data_stats(grid))
    hard_with_stress = {**grid.properties, **stress_features}

    t0 = time.perf_counter()
    no_stress = precomputed.get("elastic_no_stress") or generate_elastic_properties(
        grid.properties, config, stress_features=stress_features, mode="no_stress"
    )
    rows.extend(_elastic_metrics("Exp-1_no_stress_rock_physics", no_stress.tensors, hard_with_stress))
    rows.append(_runtime_row("Exp-1_no_stress_rock_physics", time.perf_counter() - t0))

    t0 = time.perf_counter()
    stress = precomputed.get("elastic_stress") or generate_elastic_properties(
        grid.properties, config, stress_features=stress_features, mode=config.get("rock_physics", {}).get("mode", "strong")
    )
    rows.extend(_elastic_metrics("Exp-2_stress_aware_rock_physics", stress.tensors, hard_with_stress))
    rows.extend(_difference_metrics("Exp-2_stress_minus_baseline", no_stress.tensors, stress.tensors, ["Vp", "Vs", "density", "AI", "VpVs"]))
    rows.append(_runtime_row("Exp-2_stress_aware_rock_physics", time.perf_counter() - t0))

    t0 = time.perf_counter()
    seis_no = precomputed.get("seismic_no_stress") or generate_seismic_softdata(no_stress.tensors, config)
    rows.extend(_seismic_metrics("Exp-3_no_stress_seismic_soft_data", seis_no.tensors, hard_with_stress))
    rows.append(_runtime_row("Exp-3_no_stress_seismic_soft_data", time.perf_counter() - t0))

    t0 = time.perf_counter()
    seis_st = precomputed.get("seismic_stress") or generate_seismic_softdata(stress.tensors, config)
    rows.extend(_seismic_metrics("Exp-4_stress_aware_seismic_soft_data", seis_st.tensors, hard_with_stress))
    rows.extend(_difference_metrics("Exp-4_stress_seismic_minus_baseline", seis_no.tensors, seis_st.tensors, ["poststack_seismic", "AVO_gradient"]))
    rows.append(_runtime_row("Exp-4_stress_aware_seismic_soft_data", time.perf_counter() - t0))

    if bool(config.get("uncertainty", {}).get("enabled", False)):
        t0 = time.perf_counter()
        unc = run_uncertainty(grid.properties, config, stress_features, include_seismic=False)
        for prop, stats in unc.items():
            rows.append({"experiment": "Exp-5_noise_uncertainty_sensitivity", "property": prop, "metric": "uncertainty_std_mean", "value": float(torch.mean(stats["std"]))})
            rows.append({"experiment": "Exp-5_noise_uncertainty_sensitivity", "property": prop, "metric": "uncertainty_p90_p10_mean_width", "value": float(torch.mean(stats["P90"] - stats["P10"]))})
        rows.append(_runtime_row("Exp-5_noise_uncertainty_sensitivity", time.perf_counter() - t0))
    else:
        rows.append({"experiment": "Exp-5_noise_uncertainty_sensitivity", "property": "all", "metric": "status", "value": "disabled_in_config"})

    t0 = time.perf_counter()
    horizon = precomputed.get("horizon") or generate_horizon_softdata({**grid.properties, **_torch_to_numpy_dict(stress.tensors)}, config)
    rows.append({"experiment": "Exp-6_with_without_horizon_constraints", "property": "horizon_probability", "metric": "mean", "value": float(np.nanmean(horizon.tensors["horizon_probability"]))})
    rows.append({"experiment": "Exp-6_with_without_horizon_constraints", "property": "reservoir_mask", "metric": "cell_count", "value": int(np.sum(horizon.tensors["reservoir_mask"]))})
    rows.append(_runtime_row("Exp-6_with_without_horizon_constraints", time.perf_counter() - t0))

    _write_metrics(output_dir / "metrics.csv", rows)
    save_json({"config": config, "elapsed_s": time.perf_counter() - start, "gpu": _gpu_info()}, output_dir / "ablation_log.json")
    _save_ablation_figures(grid, no_stress.tensors, stress.tensors, seis_no.tensors, seis_st.tensors, config, output_dir)
    return rows


def _exp0_hard_data_stats(grid: GridData) -> list[Dict[str, Any]]:
    rows = []
    for name, arr in grid.properties.items():
        if name == "facies":
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            rows.extend([
                {"experiment": "Exp-0_hard_data_statistics", "property": name, "metric": "mean", "value": float(np.mean(finite))},
                {"experiment": "Exp-0_hard_data_statistics", "property": name, "metric": "std", "value": float(np.std(finite))},
            ])
    rows.append({"experiment": "Exp-0_hard_data_statistics", "property": "grid", "metric": "missing_count", "value": grid.metadata["missing_count"]})
    rows.append({"experiment": "Exp-0_hard_data_statistics", "property": "grid", "metric": "duplicate_count", "value": grid.metadata["duplicate_count"]})
    return rows


def _elastic_metrics(experiment: str, tensors: Mapping[str, Any], hard: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    rows = []
    facies = hard.get("facies")
    for key in ("Vp", "Vs", "density", "AI", "VpVs"):
        arr = _to_numpy(tensors[key])
        rows.extend(_stats_rows(experiment, key, arr))
        if facies is not None:
            rows.extend(_facies_rows(experiment, key, arr, facies))
            rows.append({"experiment": experiment, "property": key, "metric": "facies_separability_fisher", "value": fisher_separability(arr, facies)})
    for soft_key in ("AI", "VpVs"):
        for hard_key in _correlation_keys():
            if hard_key in hard:
                rows.append({"experiment": experiment, "property": soft_key, "metric": f"corr_with_{hard_key}", "value": pearson_corr(_to_numpy(tensors[soft_key]), hard[hard_key])})
    return rows


def _seismic_metrics(experiment: str, tensors: Mapping[str, Any], hard: Mapping[str, np.ndarray]) -> list[Dict[str, Any]]:
    rows = []
    facies = hard.get("facies")
    for key in ("poststack_seismic", "AVO_intercept", "AVO_gradient"):
        arr = _to_numpy(tensors[key])
        rows.extend(_stats_rows(experiment, key, arr))
        if facies is not None:
            rows.append({"experiment": experiment, "property": key, "metric": "facies_separability_fisher", "value": fisher_separability(arr, facies)})
        for hard_key in _correlation_keys():
            if hard_key in hard:
                rows.append({"experiment": experiment, "property": key, "metric": f"corr_with_{hard_key}", "value": pearson_corr(arr, hard[hard_key])})
    return rows


def _difference_metrics(experiment: str, baseline: Mapping[str, Any], stress: Mapping[str, Any], keys) -> list[Dict[str, Any]]:
    rows = []
    for key in keys:
        if key not in baseline or key not in stress:
            continue
        diff = _crop_pair(_to_numpy(stress[key]), _to_numpy(baseline[key]))
        rows.append({"experiment": experiment, "property": key, "metric": "mean_abs_difference", "value": float(np.nanmean(np.abs(diff[0] - diff[1])))})
        rows.append({"experiment": experiment, "property": key, "metric": "rms_difference", "value": float(np.sqrt(np.nanmean((diff[0] - diff[1]) ** 2)))})
    return rows


def _stats_rows(experiment: str, key: str, arr: np.ndarray) -> list[Dict[str, Any]]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    return [
        {"experiment": experiment, "property": key, "metric": "mean", "value": float(np.mean(finite))},
        {"experiment": experiment, "property": key, "metric": "std", "value": float(np.std(finite))},
        {"experiment": experiment, "property": key, "metric": "p10", "value": float(np.percentile(finite, 10))},
        {"experiment": experiment, "property": key, "metric": "p50", "value": float(np.percentile(finite, 50))},
        {"experiment": experiment, "property": key, "metric": "p90", "value": float(np.percentile(finite, 90))},
    ]


def _facies_rows(experiment: str, key: str, arr: np.ndarray, facies: np.ndarray) -> list[Dict[str, Any]]:
    arr, fac = _crop_pair(arr, facies)
    rows = []
    for code in np.unique(fac[fac >= 0]):
        vals = arr[fac == code]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            rows.append({"experiment": experiment, "property": key, "metric": f"facies_{int(code)}_mean", "value": float(np.mean(vals))})
            rows.append({"experiment": experiment, "property": key, "metric": f"facies_{int(code)}_std", "value": float(np.std(vals))})
    return rows


def fisher_separability(values: np.ndarray, facies: np.ndarray) -> float:
    """Compute simple between-/within-facies variance ratio."""
    values, facies = _crop_pair(values, facies)
    valid = np.isfinite(values) & (facies >= 0)
    if not np.any(valid):
        return float("nan")
    y = values[valid]
    f = facies[valid]
    overall = float(np.mean(y))
    between = 0.0
    within = 0.0
    for code in np.unique(f):
        vals = y[f == code]
        between += vals.size * (float(np.mean(vals)) - overall) ** 2
        within += float(np.sum((vals - np.mean(vals)) ** 2))
    return float(between / max(within, 1.0e-12))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation after finite filtering and shape cropping."""
    a, b = _crop_pair(a, b)
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _crop_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)
    ndim = min(a.ndim, b.ndim)
    if a.ndim != ndim:
        a = a.reshape(a.shape[-ndim:])
    if b.ndim != ndim:
        b = b.reshape(b.shape[-ndim:])
    shape = tuple(min(a.shape[i], b.shape[i]) for i in range(ndim))
    slices = tuple(slice(0, s) for s in shape)
    return a[slices], b[slices]


def _write_metrics(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "property", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _save_ablation_figures(grid, no_stress, stress, seis_no, seis_st, config, output_dir: Path) -> None:
    dpi = int(config.get("export", {}).get("figures_dpi", 300))
    save_difference_panel(no_stress, stress, ["Vp", "Vs", "density", "AI", "VpVs"], output_dir / "figures" / "stress_minus_baseline_elastic.png", dpi=dpi)
    save_difference_panel(seis_no, seis_st, ["poststack_seismic", "AVO_gradient"], output_dir / "figures" / "stress_minus_baseline_seismic.png", dpi=dpi)
    if "facies" in grid.properties:
        save_crossplot(grid.properties["porosity"], _to_numpy(stress["AI"]), grid.properties["facies"], output_dir / "figures" / "porosity_ai_by_facies.png", "porosity", "AI", max_points=int(config.get("ablation", {}).get("max_crossplot_points", 20000)), dpi=dpi)


def _runtime_row(experiment: str, elapsed: float) -> Dict[str, Any]:
    return {"experiment": experiment, "property": "runtime", "metric": "seconds", "value": float(elapsed)}


def _correlation_keys() -> tuple[str, ...]:
    return (
        "porosity",
        "oil_saturation",
        "brittleness_index",
        "permeability",
        "gamma",
        "stress",
        "SH1",
        "SH2",
        "stress_MPa",
        "mean_stress_proxy",
        "differential_stress",
        "stress_ratio",
        "stress_anisotropy_index",
    )


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(idx),
        "allocated_mb": float(torch.cuda.memory_allocated(idx) / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved(idx) / 1024**2),
    }


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _torch_to_numpy_dict(tensors: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {k: _to_numpy(v) for k, v in tensors.items()}
