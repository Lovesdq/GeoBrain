"""Monte Carlo uncertainty support for soft-data simulation."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from .rock_physics_stress import generate_elastic_properties
from .seismic_softdata import generate_seismic_softdata


LOGGER = logging.getLogger(__name__)


def run_uncertainty(
    properties: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    stress_features: Optional[Mapping[str, np.ndarray]] = None,
    output_keys: Iterable[str] = ("Vp", "Vs", "density", "AI", "VpVs"),
    include_seismic: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run Monte Carlo perturbations and summarize selected output volumes."""
    unc_cfg = config.get("uncertainty", {})
    n_samples = int(unc_cfg.get("n_samples", 8))
    rng = np.random.default_rng(int(config.get("project", {}).get("random_seed", 20260505)))
    samples: Dict[str, list[torch.Tensor]] = {key: [] for key in output_keys}
    if include_seismic:
        samples["poststack_seismic"] = []

    for i in range(n_samples):
        cfg_i = perturb_config(config, rng)
        elastic = generate_elastic_properties(properties, cfg_i, stress_features=stress_features, mode=cfg_i.get("rock_physics", {}).get("mode", "strong"))
        for key in output_keys:
            samples[key].append(elastic.tensors[key].detach().cpu())
        if include_seismic:
            seismic = generate_seismic_softdata(elastic.tensors, cfg_i)
            samples["poststack_seismic"].append(seismic.tensors["poststack_seismic"].detach().cpu())
        LOGGER.info("Monte Carlo sample %d/%d complete", i + 1, n_samples)

    return {key: summarize_samples(value) for key, value in samples.items() if value}


def perturb_config(config: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Return a deep-copied config with one Monte Carlo parameter draw."""
    cfg = copy.deepcopy(config)
    pert = cfg.get("uncertainty", {}).get("perturbations", {})

    mineral_std = float(pert.get("mineral_modulus_rel_std", 0.03))
    minerals = cfg.get("rock_physics", {}).get("minerals", {})
    for facies, composition in list(minerals.items()):
        if not isinstance(composition, dict):
            continue
        jittered = {k: max(1.0e-6, float(v) * float(rng.lognormal(0.0, mineral_std))) for k, v in composition.items()}
        total = sum(jittered.values())
        minerals[facies] = {k: v / total for k, v in jittered.items()}

    fluid_std = float(pert.get("fluid_modulus_rel_std", 0.05))
    fluids = cfg.get("rock_physics", {}).setdefault("fluids", {})
    fluids["brine_K_multiplier"] = float(rng.lognormal(0.0, fluid_std))
    fluids["hydrocarbon_K_multiplier"] = float(rng.lognormal(0.0, fluid_std))

    stress_std = float(pert.get("stress_correction_rel_std", 0.15))
    pressure = cfg.get("stress", {}).setdefault("effective_pressure", {})
    pressure["stress_to_pressure_scale"] = float(pressure.get("stress_to_pressure_scale", 0.60)) * float(rng.lognormal(0.0, stress_std))

    freq_std = float(pert.get("wavelet_frequency_rel_std", 0.05))
    seismic = cfg.get("seismic", {})
    freqs = seismic.get("wavelet_frequencies_hz", [25.0])
    seismic["wavelet_frequencies_hz"] = [float(f) * float(rng.lognormal(0.0, freq_std)) for f in freqs]

    noise_range = pert.get("noise_std_fraction_range", [0.02, 0.10])
    seismic.setdefault("noise", {})["gaussian_std_fraction"] = float(rng.uniform(noise_range[0], noise_range[1]))
    seismic["noise"]["seed"] = int(rng.integers(0, 2**31 - 1))
    return cfg


def summarize_samples(samples: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return mean, std, P10, P50 and P90 for a list of tensors."""
    stack = torch.stack([s.to(torch.float32) for s in samples], dim=0)
    return {
        "mean": torch.mean(stack, dim=0),
        "std": torch.std(stack, dim=0, unbiased=False),
        "P10": torch.quantile(stack, 0.10, dim=0),
        "P50": torch.quantile(stack, 0.50, dim=0),
        "P90": torch.quantile(stack, 0.90, dim=0),
    }
