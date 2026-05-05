"""Stress-derived proxy features for stress-conditioned soft data."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)
EPS = 1.0e-8


def compute_stress_features(
    properties: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[Dict[str, np.ndarray], list[str]]:
    """Compute stress proxy features from SH1 and SH2.

    This is deliberately named a proxy calculation: without pore pressure,
    vertical stress and stress orientation, these features should be described
    as stress-informed/stress-conditioned rather than full geomechanical state.
    """
    if "SH1" not in properties or "SH2" not in properties:
        raise ValueError("Both SH1 and SH2 are required for stress feature generation")
    unit = config.get("stress", {}).get("unit") or config.get("schema", {}).get("property_units", {}).get("SH1", "MPa")
    sh1 = convert_stress_to_mpa(properties["SH1"].astype(np.float32, copy=False), unit)
    sh2 = convert_stress_to_mpa(properties["SH2"].astype(np.float32, copy=False), unit)

    warnings = []
    if np.any(sh1 < sh2):
        n = int(np.sum(sh1 < sh2))
        warnings.append(f"SH1 < SH2 at {n} grid cells; check stress convention or column mapping")
    if np.any(sh2 <= 0):
        n = int(np.sum(sh2 <= 0))
        warnings.append(f"SH2 <= 0 at {n} grid cells; stress_ratio is clipped with EPS")
    for warning in warnings:
        LOGGER.warning(warning)

    differential = sh1 - sh2
    ratio = sh1 / np.maximum(sh2, EPS)
    anisotropy = differential / np.maximum(sh1 + sh2, EPS)
    mean_proxy = 0.5 * (sh1 + sh2)

    features: Dict[str, np.ndarray] = {
        "SH1_MPa": sh1.astype(np.float32, copy=False),
        "SH2_MPa": sh2.astype(np.float32, copy=False),
        "differential_stress": differential.astype(np.float32, copy=False),
        "stress_ratio": ratio.astype(np.float32, copy=False),
        "stress_anisotropy_index": anisotropy.astype(np.float32, copy=False),
        "mean_stress_proxy": mean_proxy.astype(np.float32, copy=False),
        "stress_mean_proxy": mean_proxy.astype(np.float32, copy=False),
    }

    if "stress" in properties:
        stress_unit = config.get("schema", {}).get("property_units", {}).get("stress", unit)
        stress = convert_stress_to_mpa(properties["stress"].astype(np.float32, copy=False), stress_unit)
        features["stress_MPa"] = stress.astype(np.float32, copy=False)

    normalize = str(config.get("stress", {}).get("normalize", "zscore")).lower()
    for name in ("differential_stress", "stress_ratio", "stress_anisotropy_index", "mean_stress_proxy"):
        features[f"{name}_norm"] = normalize_array(features[name], method=normalize)
    if "stress_MPa" in features:
        features["stress_MPa_norm"] = normalize_array(features["stress_MPa"], method=normalize)
    return features, warnings


def compute_stress_brittleness_proxies(
    properties: Mapping[str, np.ndarray],
    stress_features: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute sweet-spot and fracture-risk proxy volumes."""
    weights = config.get("rock_physics", {}).get("stress_brittleness_weights", {})
    w_bi = float(weights.get("brittleness", 0.45))
    w_so = float(weights.get("oil_saturation", 0.25))
    w_an = float(weights.get("stress_anisotropy", 0.20))
    w_ds = float(weights.get("differential_stress", 0.10))

    brittleness = normalize_array(_fraction(properties.get("brittleness_index", 0.0)), "minmax")
    oil_sat = normalize_array(_fraction(properties.get("oil_saturation", 0.0)), "minmax")
    anis = normalize_array(np.abs(stress_features["stress_anisotropy_index"]), "minmax")
    diff = normalize_array(stress_features["differential_stress"], "minmax")
    sweet = w_bi * brittleness + w_so * oil_sat + w_an * anis + w_ds * diff
    fracability = normalize_array(0.60 * brittleness + 0.25 * diff + 0.15 * anis, "minmax")
    return {
        "stress_brittleness_sweet_spot": sweet.astype(np.float32),
        "fracability_proxy": fracability.astype(np.float32),
        "fracture_risk_proxy": fracability.astype(np.float32),
    }


def convert_stress_to_mpa(values: np.ndarray, unit: str = "MPa") -> np.ndarray:
    """Convert stress values to MPa."""
    unit_l = str(unit).lower()
    if unit_l == "mpa":
        scale = 1.0
    elif unit_l == "gpa":
        scale = 1000.0
    elif unit_l == "kpa":
        scale = 0.001
    elif unit_l == "pa":
        scale = 1.0e-6
    elif unit_l in {"psi", "psia"}:
        scale = 0.00689476
    else:
        LOGGER.warning("Unknown stress unit '%s'; assuming MPa", unit)
        scale = 1.0
    return values.astype(np.float32, copy=False) * np.float32(scale)


def normalize_array(values, method: str = "zscore") -> np.ndarray:
    """Normalize finite values using z-score or min-max scaling."""
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    vals = arr[finite]
    method = method.lower()
    if method == "minmax":
        lo, hi = float(vals.min()), float(vals.max())
        out[finite] = (vals - lo) / max(hi - lo, EPS)
    elif method == "none":
        out[finite] = vals
    else:
        out[finite] = (vals - float(vals.mean())) / max(float(vals.std()), EPS)
    return out


def _fraction(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) > 1.5:
        return arr / 100.0
    return arr
