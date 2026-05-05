"""Horizon, interface probability and signed-distance soft constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """Container for horizon-derived soft-data volumes."""

    tensors: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def generate_horizon_softdata(
    properties: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> HorizonResult:
    """Extract reservoir top/base and build probability/SDF volumes."""
    horizon_cfg = config.get("horizon", {})
    source = str(horizon_cfg.get("source", "facies")).lower()
    if source == "facies" and "facies" in properties:
        mask = _reservoir_mask(properties["facies"], config)
    else:
        prop_name = horizon_cfg.get("gradient_property", "AI")
        if prop_name not in properties:
            raise ValueError(f"Cannot build horizon soft data; property '{prop_name}' not available")
        mask = _gradient_mask(properties[prop_name])

    top, base = extract_top_base(mask)
    sigma = float(horizon_cfg.get("probability_sigma_samples", 2.0))
    prob = interface_probability(mask.shape, top, base, sigma=sigma)
    sdf = signed_distance_field(mask)

    metadata = {
        "source": source,
        "reservoir_cell_count": int(mask.sum()),
        "has_geobrain_implicit_module": has_geobrain_implicit(),
        "implicit_usage": (
            "GeoBrain implicit modeling is available for point/orientation constrained models; "
            "this pipeline uses a direct facies/gradient fallback for full-grid soft constraints."
        ),
    }
    return HorizonResult(
        tensors={
            "reservoir_mask": mask.astype(np.uint8),
            "reservoir_top_index": top.astype(np.int32),
            "reservoir_base_index": base.astype(np.int32),
            "horizon_probability": prob.astype(np.float32),
            "signed_distance_field": sdf.astype(np.float32),
        },
        metadata=metadata,
    )


def extract_top_base(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return top and base z-indices of True intervals for each ``x,y`` trace."""
    mask = np.asarray(mask, dtype=bool)
    has = mask.any(axis=2)
    top = np.full(mask.shape[:2], -1, dtype=np.int32)
    base = np.full(mask.shape[:2], -1, dtype=np.int32)
    top[has] = np.argmax(mask, axis=2)[has]
    base[has] = mask.shape[2] - 1 - np.argmax(mask[:, :, ::-1], axis=2)[has]
    return top, base


def interface_probability(
    shape: tuple[int, int, int],
    top: np.ndarray,
    base: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Create a Gaussian probability halo around top/base interfaces."""
    z = np.arange(shape[2], dtype=np.float32)[None, None, :]
    valid = top >= 0
    top3 = top[:, :, None].astype(np.float32)
    base3 = base[:, :, None].astype(np.float32)
    dist = np.minimum(np.abs(z - top3), np.abs(z - base3))
    prob = np.exp(-0.5 * (dist / max(sigma, 1.0e-6)) ** 2)
    prob[~valid, :] = 0.0
    return prob


def signed_distance_field(mask: np.ndarray) -> np.ndarray:
    """Compute SDF with negative values inside the reservoir mask."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for signed-distance horizon soft data") from exc
    mask = np.asarray(mask, dtype=bool)
    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask)
    return outside - inside


def has_geobrain_implicit() -> bool:
    """Return whether GeoBrain implicit geological modeling can be imported."""
    try:
        import geobrain.geomodel.implicit  # noqa: F401
    except Exception:
        return False
    return True


def _reservoir_mask(facies: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    code_to_name = {int(k): str(v) for k, v in config.get("schema", {}).get("facies_map", {}).items()}
    reservoir_names = set(config.get("schema", {}).get("reservoir_facies", ["oil_layer", "poor_oil_layer"]))
    reservoir_codes = {code for code, name in code_to_name.items() if name in reservoir_names}
    if not reservoir_codes:
        LOGGER.warning("No reservoir facies codes resolved; defaulting to facies > 0")
        return np.asarray(facies) > 0
    return np.isin(facies, list(reservoir_codes))


def _gradient_mask(volume: np.ndarray) -> np.ndarray:
    grad = np.abs(np.gradient(np.asarray(volume, dtype=np.float32), axis=2))
    threshold = np.nanpercentile(grad, 90)
    return grad >= threshold
