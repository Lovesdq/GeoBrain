"""Seismic soft-data generation from elastic property volumes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from geobrain.physics.wave import RickerWavelet, Shuey, compute_reflectivity, create_conv_matrix


LOGGER = logging.getLogger(__name__)


@dataclass
class SeismicResult:
    """Container for seismic soft-data tensors and metadata."""

    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


def generate_seismic_softdata(
    elastic: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> SeismicResult:
    """Generate reflectivity, angle stacks, post-stack and AVO attributes."""
    seis_cfg = config.get("seismic", {})
    angles = [float(a) for a in seis_cfg.get("angles", [12, 24, 36])]
    freqs = _expand_frequencies(seis_cfg.get("wavelet_frequencies_hz", [25.0]), len(angles))
    dt = float(seis_cfg.get("dt_s", 0.001))
    method = str(seis_cfg.get("reflectivity_method", "shuey"))

    vp = elastic["Vp"]
    vs = elastic["Vs"]
    rho = elastic["density"]
    vp1, vs1, rho1 = vp[:, :, :-1], vs[:, :, :-1], rho[:, :, :-1]
    vp2, vs2, rho2 = vp[:, :, 1:], vs[:, :, 1:], rho[:, :, 1:]

    reflectivity = compute_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, theta=angles, method=method)
    post_reflectivity = compute_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, theta=[0.0], method=method)[0]
    intercept, gradient, curvature = Shuey().avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2)

    prestack = _convolve_reflectivity(reflectivity, freqs, dt, config)
    post_wavelet, _ = RickerWavelet()(f0=float(freqs[min(1, len(freqs) - 1)]), dt=dt, device=str(vp.device))
    poststack = _apply_conv_matrix(post_reflectivity, post_wavelet)

    tensors: Dict[str, torch.Tensor] = {
        "reflectivity": reflectivity,
        "post_reflectivity": post_reflectivity,
        "prestack_seismic": prestack,
        "poststack_seismic": poststack,
        "AVO_intercept": intercept,
        "AVO_gradient": gradient,
        "AVO_curvature": curvature,
    }
    for i, angle in enumerate(angles):
        tensors[f"seismic_angle_{int(angle)}"] = prestack[i]

    if seis_cfg.get("bandpass_hz"):
        tensors["prestack_seismic_bandpass"] = bandpass_volume(
            prestack,
            dt=dt,
            band=seis_cfg["bandpass_hz"],
        )

    noise_cfg = seis_cfg.get("noise", {})
    if bool(noise_cfg.get("enabled", False)):
        noisy = add_gaussian_noise(
            prestack,
            std_fraction=float(noise_cfg.get("gaussian_std_fraction", 0.05)),
            seed=noise_cfg.get("seed"),
        )
        tensors["prestack_seismic_noisy"] = noisy

    metadata = {
        "angles": angles,
        "wavelet_frequencies_hz": freqs,
        "dt_s": dt,
        "reflectivity_method": method,
        "reused_geobrain": ["compute_reflectivity", "RickerWavelet", "create_conv_matrix", "Shuey.avo_attributes"],
    }
    return SeismicResult(tensors=tensors, metadata=metadata)


def add_gaussian_noise(signal: torch.Tensor, std_fraction: float, seed=None) -> torch.Tensor:
    """Add zero-mean Gaussian noise scaled by signal standard deviation."""
    std = torch.std(signal)
    if not torch.isfinite(std) or std <= 0:
        return signal.clone()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=signal.device)
        generator.manual_seed(int(seed))
    noise = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=generator)
    return signal + noise * std * float(std_fraction)


def bandpass_volume(volume: torch.Tensor, dt: float, band) -> torch.Tensor:
    """Apply a Butterworth band-pass filter along the last axis."""
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for band-pass filtering") from exc
    low, high = float(band[0]), float(band[1])
    fs = 1.0 / float(dt)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    data = volume.detach().cpu().numpy()
    filtered = sosfiltfilt(sos, data, axis=-1).copy()
    return torch.as_tensor(filtered, dtype=volume.dtype, device=volume.device)


def _convolve_reflectivity(
    reflectivity: torch.Tensor,
    freqs: list[float],
    dt: float,
    config: Mapping[str, Any],
) -> torch.Tensor:
    n_angles = reflectivity.shape[0]
    out = torch.empty_like(reflectivity)
    for i in range(n_angles):
        wavelet, _ = RickerWavelet()(f0=float(freqs[i]), dt=dt, device=str(reflectivity.device))
        out[i] = _apply_conv_matrix(reflectivity[i], wavelet, chunk_size=int(config.get("seismic", {}).get("chunk_size_inline", 64)))
    return out


def _apply_conv_matrix(reflectivity: torch.Tensor, wavelet: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    nt = reflectivity.shape[-1]
    W = create_conv_matrix(wavelet.to(reflectivity.device), nt, mode="same")
    out = torch.empty_like(reflectivity)
    nx = reflectivity.shape[0]
    for start in range(0, nx, max(1, chunk_size)):
        end = min(nx, start + max(1, chunk_size))
        out[start:end] = torch.einsum("...i,oi->...o", reflectivity[start:end], W)
    return out


def _expand_frequencies(freqs, n_angles: int) -> list[float]:
    freqs = [float(f) for f in freqs]
    if len(freqs) == n_angles:
        return freqs
    if len(freqs) == 1:
        return freqs * n_angles
    if len(freqs) < n_angles:
        return freqs + [freqs[-1]] * (n_angles - len(freqs))
    return freqs[:n_angles]
