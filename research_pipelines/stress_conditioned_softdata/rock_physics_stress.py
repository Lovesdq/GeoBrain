"""Stress-conditioned rock physics for 3D geological grids."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from geobrain.physics.rock import (
    DensityModel,
    Gassmann,
    SoftSand,
    get_fluid,
    mix_minerals_vrh,
    v_from_moduli,
)


LOGGER = logging.getLogger(__name__)
EPS = 1.0e-8


@dataclass
class ElasticResult:
    """Container for elastic soft-data tensors and provenance metadata."""

    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


def select_device(config: Mapping[str, Any]) -> torch.device:
    """Select CPU/GPU device, preferring configured ``cuda:4`` when available."""
    requested = str(config.get("project", {}).get("device", "auto"))
    if requested == "auto":
        requested = "cuda:4" if torch.cuda.is_available() and torch.cuda.device_count() > 4 else "cuda:0"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        if ":" in requested:
            idx = int(requested.split(":", 1)[1])
            if idx >= torch.cuda.device_count():
                LOGGER.warning("Requested %s but only %d CUDA devices exist; falling back to CPU", requested, torch.cuda.device_count())
                return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cpu")


def generate_elastic_properties(
    properties: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    stress_features: Optional[Mapping[str, np.ndarray]] = None,
    mode: str = "strong",
    device: Optional[torch.device | str] = None,
) -> ElasticResult:
    """Generate Vp, Vs, density, impedance and EI volumes.

    Modes:
        ``no_stress`` uses constant effective pressure.
        ``weak`` uses constant-pressure SoftSand/Gassmann plus empirical stress correction.
        ``strong`` passes a spatial pressure proxy into GeoBrain SoftSand.
    """
    device = torch.device(device) if device is not None else select_device(config)
    dtype = torch.float32 if str(config.get("project", {}).get("dtype", "float32")) == "float32" else torch.float64
    rock_cfg = config.get("rock_physics", {})

    phi = _to_fraction(properties["porosity"], config, "porosity")
    phi_clip = rock_cfg.get("porosity_clip", [0.01, 0.42])
    phi_t = torch.as_tensor(np.clip(phi, phi_clip[0], phi_clip[1]), dtype=dtype, device=device)

    so = _to_fraction(properties.get("oil_saturation", np.zeros_like(phi)), config, "oil_saturation")
    so_clip = rock_cfg.get("oil_saturation_clip", [0.0, 1.0])
    so_t = torch.as_tensor(np.clip(so, so_clip[0], so_clip[1]), dtype=dtype, device=device)
    facies = properties.get("facies")
    facies_np = np.asarray(facies, dtype=np.int16) if facies is not None else np.full(phi.shape, -1, dtype=np.int16)

    K_m, G_m, rho_m, phi_c, cn, pressure_multiplier = _facies_parameter_tensors(
        facies_np, config, device=device, dtype=dtype
    )
    phi_t = torch.minimum(phi_t, torch.clamp(phi_c * 0.98, min=0.02))
    K_fl, rho_fl = _fluid_tensors(so_t, config)

    pressure = _effective_pressure(properties, config, stress_features, mode, device, dtype)
    pressure = pressure * pressure_multiplier

    soft_sand = SoftSand()
    gassmann = Gassmann()
    density_model = DensityModel()

    k_dry, g_dry = soft_sand(
        K_m,
        G_m,
        phi_t,
        phi_c,
        cn,
        pressure,
        float(rock_cfg.get("friction_coefficient", 1.0)),
    )
    k_sat, g_sat = gassmann(k_dry, g_dry, K_m, K_fl, phi_t)
    rho = density_model(phi_t, rho_m, rho_fl)
    vp, vs = v_from_moduli(k_sat, g_sat, rho)

    if mode == "weak" and stress_features is not None:
        vp, vs, rho = _apply_weak_stress_correction(vp, vs, rho, stress_features, config, device, dtype)

    vp = torch.clamp(vp, min=300.0, max=8000.0)
    vs = torch.clamp(vs, min=50.0, max=5000.0)
    rho = torch.clamp(rho, min=0.8, max=3.2)

    ai = vp * rho
    si = vs * rho
    out: Dict[str, torch.Tensor] = {
        "Vp": vp,
        "Vs": vs,
        "density": rho,
        "VpVs": vp / torch.clamp(vs, min=EPS),
        "AI": ai,
        "SI": si,
        "effective_pressure_proxy_MPa": pressure,
        "K_sat_GPa": k_sat,
        "G_sat_GPa": g_sat,
    }
    for angle in rock_cfg.get("elastic_impedance_angles", [12, 24, 36]):
        out[f"EI_{int(angle)}"] = elastic_impedance(vp, vs, rho, float(angle))

    metadata = {
        "mode": mode,
        "device": str(device),
        "model": "GeoBrain SoftSand + Gassmann + DensityModel + v_from_moduli",
        "stress_interpretation": (
            "stress-informed effective-pressure proxy; not a full 3D geomechanical forward model"
        ),
    }
    return ElasticResult(tensors=out, metadata=metadata)


def elastic_impedance(vp: torch.Tensor, vs: torch.Tensor, rho: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Compute Connolly-style normalized elastic impedance for one angle."""
    theta = torch.as_tensor(np.deg2rad(angle_deg), dtype=vp.dtype, device=vp.device)
    vp0 = torch.nanmean(vp)
    vs0 = torch.nanmean(vs)
    rho0 = torch.nanmean(rho)
    k = torch.clamp((vs0 / torch.clamp(vp0, min=EPS)) ** 2, min=0.01, max=0.60)
    sin2 = torch.sin(theta) ** 2
    tan2 = torch.tan(theta) ** 2
    a = 1.0 + tan2
    b = -8.0 * k * sin2
    c = 1.0 - 4.0 * k * sin2
    vp_n = torch.clamp(vp / torch.clamp(vp0, min=EPS), min=EPS)
    vs_n = torch.clamp(vs / torch.clamp(vs0, min=EPS), min=EPS)
    rho_n = torch.clamp(rho / torch.clamp(rho0, min=EPS), min=EPS)
    return vp0 * rho0 * (vp_n ** a) * (vs_n ** b) * (rho_n ** c)


def _facies_parameter_tensors(
    facies: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rock_cfg = config.get("rock_physics", {})
    code_to_name = {int(k): v for k, v in config.get("schema", {}).get("facies_map", {}).items()}
    default_comp = rock_cfg.get("minerals", {}).get("default", {"quartz": 0.75, "clay": 0.25})
    default_params = {
        "critical_porosity": float(rock_cfg.get("critical_porosity", 0.40)),
        "coordination_number": float(rock_cfg.get("coordination_number", 7.0)),
        "pressure_multiplier": 1.0,
    }
    shape = facies.shape
    K_m = torch.empty(shape, dtype=dtype, device=device)
    G_m = torch.empty(shape, dtype=dtype, device=device)
    rho_m = torch.empty(shape, dtype=dtype, device=device)
    phi_c = torch.empty(shape, dtype=dtype, device=device)
    cn = torch.empty(shape, dtype=dtype, device=device)
    pm = torch.empty(shape, dtype=dtype, device=device)

    unique_codes = np.unique(facies)
    if unique_codes.size == 0:
        unique_codes = np.array([-1], dtype=np.int16)
    for code in unique_codes:
        name = code_to_name.get(int(code), "default")
        comp = rock_cfg.get("minerals", {}).get(name, default_comp)
        K, G, rho = mix_minerals_vrh(comp)
        params = {**default_params, **rock_cfg.get("facies_parameters", {}).get(name, {})}
        mask = torch.as_tensor(facies == code, dtype=torch.bool, device=device)
        K_m[mask] = K.to(device=device, dtype=dtype)
        G_m[mask] = G.to(device=device, dtype=dtype)
        rho_m[mask] = rho.to(device=device, dtype=dtype)
        phi_c[mask] = float(params["critical_porosity"])
        cn[mask] = float(params["coordination_number"])
        pm[mask] = float(params["pressure_multiplier"])
    return K_m, G_m, rho_m, phi_c, cn, pm


def _fluid_tensors(so_t: torch.Tensor, config: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    fluids_cfg = config.get("rock_physics", {}).get("fluids", {})
    brine = get_fluid(fluids_cfg.get("brine", "brine"))
    hydrocarbon = get_fluid(fluids_cfg.get("hydrocarbon", "oil_medium"))
    sw = torch.clamp(1.0 - so_t, 0.0, 1.0)
    so = torch.clamp(so_t, 0.0, 1.0)
    k_w = torch.as_tensor(brine.K * float(fluids_cfg.get("brine_K_multiplier", 1.0)), dtype=so_t.dtype, device=so_t.device)
    k_h = torch.as_tensor(hydrocarbon.K * float(fluids_cfg.get("hydrocarbon_K_multiplier", 1.0)), dtype=so_t.dtype, device=so_t.device)
    rho_w = torch.as_tensor(brine.rho * float(fluids_cfg.get("brine_rho_multiplier", 1.0)), dtype=so_t.dtype, device=so_t.device)
    rho_h = torch.as_tensor(hydrocarbon.rho * float(fluids_cfg.get("hydrocarbon_rho_multiplier", 1.0)), dtype=so_t.dtype, device=so_t.device)
    if str(fluids_cfg.get("fluid_mixing", "reuss")).lower() == "voigt":
        k_fl = sw * k_w + so * k_h
    else:
        k_fl = 1.0 / (sw / torch.clamp(k_w, min=EPS) + so / torch.clamp(k_h, min=EPS))
    rho_fl = sw * rho_w + so * rho_h
    return k_fl, rho_fl


def _effective_pressure(
    properties: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    stress_features: Optional[Mapping[str, np.ndarray]],
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pressure_cfg = config.get("stress", {}).get("effective_pressure", {})
    const = float(pressure_cfg.get("no_stress_mpa", 20.0))
    shape = np.asarray(properties["porosity"]).shape
    if mode in {"no_stress", "weak"} or stress_features is None:
        return torch.full(shape, const, dtype=dtype, device=device)
    mean_proxy = np.asarray(stress_features.get("mean_stress_proxy"), dtype=np.float32)
    scale = float(pressure_cfg.get("stress_to_pressure_scale", 0.60))
    offset = float(pressure_cfg.get("offset_mpa", 0.0))
    p = mean_proxy * scale + offset
    p = np.nan_to_num(p, nan=const, posinf=const, neginf=const)
    p = np.clip(p, float(pressure_cfg.get("min_mpa", 2.0)), float(pressure_cfg.get("max_mpa", 80.0)))
    return torch.as_tensor(p, dtype=dtype, device=device)


def _apply_weak_stress_correction(
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    stress_features: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corr = config.get("stress", {}).get("weak_correction", {})
    z = torch.as_tensor(stress_features["mean_stress_proxy_norm"], dtype=dtype, device=device)
    anis = torch.as_tensor(stress_features["stress_anisotropy_index"], dtype=dtype, device=device)
    vp_factor = 1.0 + float(corr.get("vp_zscore_coeff", 0.018)) * z + float(corr.get("anisotropy_vp_coeff", 0.030)) * anis
    vs_factor = 1.0 + float(corr.get("vs_zscore_coeff", 0.010)) * z + float(corr.get("anisotropy_vs_coeff", 0.020)) * anis
    rho_factor = 1.0 + float(corr.get("rho_zscore_coeff", 0.003)) * z
    return vp * torch.clamp(vp_factor, 0.80, 1.25), vs * torch.clamp(vs_factor, 0.80, 1.25), rho * torch.clamp(rho_factor, 0.95, 1.05)


def _to_fraction(values, config: Mapping[str, Any], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    unit = str(config.get("schema", {}).get("property_units", {}).get(name, "")).lower()
    if unit in {"percent", "%"}:
        return arr / 100.0
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanmax(finite)) > 1.5 and name in {"porosity", "oil_saturation", "brittleness_index"}:
        LOGGER.warning("%s appears to be percent-valued; converting to fraction by /100", name)
        return arr / 100.0
    return arr
