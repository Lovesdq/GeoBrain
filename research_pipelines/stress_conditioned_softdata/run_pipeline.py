"""CLI entry point for the stress-conditioned soft-data pipeline."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

try:
    from .ablation import run_ablation
    from .data_schema import canonicalize_dataframe, load_config, load_table, resolve_schema, save_json, write_metadata
    from .export_io import export_all, save_npz
    from .grid_builder import build_regular_grid
    from .horizon_softdata import generate_horizon_softdata
    from .qc import run_qc
    from .rock_physics_stress import generate_elastic_properties
    from .seismic_softdata import generate_seismic_softdata
    from .stress_features import compute_geomechanics_features, compute_stress_brittleness_proxies, compute_stress_features
    from .uncertainty import run_uncertainty
    from .visualization import save_crossplot, save_difference_panel, save_slice_panel, save_volume_orthoslices
except ImportError:  # Allows: python research_pipelines/.../run_pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research_pipelines.stress_conditioned_softdata.ablation import run_ablation
    from research_pipelines.stress_conditioned_softdata.data_schema import canonicalize_dataframe, load_config, load_table, resolve_schema, save_json, write_metadata
    from research_pipelines.stress_conditioned_softdata.export_io import export_all, save_npz
    from research_pipelines.stress_conditioned_softdata.grid_builder import build_regular_grid
    from research_pipelines.stress_conditioned_softdata.horizon_softdata import generate_horizon_softdata
    from research_pipelines.stress_conditioned_softdata.qc import run_qc
    from research_pipelines.stress_conditioned_softdata.rock_physics_stress import generate_elastic_properties
    from research_pipelines.stress_conditioned_softdata.seismic_softdata import generate_seismic_softdata
    from research_pipelines.stress_conditioned_softdata.stress_features import compute_geomechanics_features, compute_stress_brittleness_proxies, compute_stress_features
    from research_pipelines.stress_conditioned_softdata.uncertainty import run_uncertainty
    from research_pipelines.stress_conditioned_softdata.visualization import save_crossplot, save_difference_panel, save_slice_panel, save_volume_orthoslices


LOGGER = logging.getLogger("stress_conditioned_softdata")


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from CLI arguments."""
    args = parse_args(argv)
    config = load_config(args.config)
    if args.input:
        config.setdefault("input", {})["path"] = args.input
    if args.output:
        config.setdefault("project", {})["output_dir"] = args.output
    if args.device:
        config.setdefault("project", {})["device"] = args.device
    if args.no_ablation:
        config.setdefault("ablation", {})["enabled"] = False

    output_dir = Path(config.get("project", {}).get("output_dir", "outputs/stress_conditioned_softdata"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "pipeline.log")
    shutil.copyfile(args.config, output_dir / "config_used.yaml")

    LOGGER.info("Starting stress-conditioned soft-data pipeline")
    start = time.perf_counter()
    df = synthetic_dataframe(config) if args.demo else load_table(config.get("input", {}).get("path", "grid.csv"))
    schema = resolve_schema(df, config)
    canonical = canonicalize_dataframe(df, schema)
    write_metadata(schema, output_dir / "metadata.json", extra={"demo": bool(args.demo)})

    grid = build_regular_grid(canonical, config)
    run_qc(grid, config, output_dir / "qc")

    stress_features, stress_warnings = compute_stress_features(grid.properties, config)
    geomech_features, geomech_warnings, geomech_metadata = compute_geomechanics_features(grid, grid.properties, config)
    stress_features.update(geomech_features)
    stress_extra = compute_stress_brittleness_proxies(grid.properties, stress_features, config)
    stress_features.update(stress_extra)
    save_json(
        {
            "geomechanics": geomech_metadata,
            "warnings": geomech_warnings,
            "interpretation": "Pp/Sv are column-priority values with deterministic gradient proxies when columns are absent or incomplete.",
        },
        output_dir / "geomechanics_metadata.json",
    )

    elastic_no = generate_elastic_properties(grid.properties, config, stress_features=stress_features, mode="no_stress")
    elastic_stress = generate_elastic_properties(
        grid.properties,
        config,
        stress_features=stress_features,
        mode=config.get("rock_physics", {}).get("mode", "strong"),
    )
    elastic_physical = generate_elastic_properties(
        grid.properties,
        config,
        stress_features=stress_features,
        mode="sv_minus_pore_pressure",
    )
    seismic_no = generate_seismic_softdata(elastic_no.tensors, config)
    seismic_stress = generate_seismic_softdata(elastic_stress.tensors, config)
    seismic_physical = generate_seismic_softdata(elastic_physical.tensors, config)

    horizon_input = {**grid.properties, **_torch_to_numpy_dict(elastic_stress.tensors)}
    horizon = generate_horizon_softdata(horizon_input, config)

    groups = {
        "grid_properties": grid.properties,
        "stress_features": stress_features,
        "elastic_no_stress": elastic_no.tensors,
        "elastic_stress": elastic_stress.tensors,
        "elastic_physical_pressure": elastic_physical.tensors,
        "seismic_no_stress": seismic_no.tensors,
        "seismic_stress": seismic_stress.tensors,
        "seismic_physical_pressure": seismic_physical.tensors,
        "horizon_softdata": horizon.tensors,
    }
    export_all(grid, groups, config, output_dir / "exports")

    if bool(config.get("uncertainty", {}).get("enabled", False)):
        uncertainty = run_uncertainty(grid.properties, config, stress_features)
        for key, stats in uncertainty.items():
            save_npz(output_dir / "exports" / f"uncertainty_{key}.npz", stats)

    save_figures(grid, stress_features, elastic_no.tensors, elastic_stress.tensors, elastic_physical.tensors, seismic_stress.tensors, seismic_physical.tensors, horizon.tensors, config, output_dir)

    if bool(config.get("ablation", {}).get("enabled", True)):
        run_ablation(
            grid,
            config,
            stress_features,
            output_dir / "ablation",
            precomputed={
                "elastic_no_stress": elastic_no,
                "elastic_stress": elastic_stress,
                "elastic_physical_pressure": elastic_physical,
                "seismic_no_stress": seismic_no,
                "seismic_stress": seismic_stress,
                "seismic_physical_pressure": seismic_physical,
                "horizon": horizon,
            },
        )

    elapsed = time.perf_counter() - start
    LOGGER.info("Pipeline complete in %.2f s. Stress warnings=%s Geomechanics warnings=%s", elapsed, stress_warnings, geomech_warnings)
    print(f"Pipeline complete: {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="Path to config.yaml")
    parser.add_argument("--input", default=None, help="CSV/Parquet input path. Overrides config input.path")
    parser.add_argument("--output", default=None, help="Output directory. Overrides config project.output_dir")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda:4")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic small-grid demo")
    parser.add_argument("--no-ablation", action="store_true", help="Skip ablation experiments")
    return parser.parse_args(argv)


def synthetic_dataframe(config: Mapping[str, Any]):
    """Create a deterministic synthetic point table for smoke tests and demos."""
    import pandas as pd

    rng = np.random.default_rng(int(config.get("project", {}).get("random_seed", 20260505)))
    nx, ny, nz = 12, 10, 18
    x = np.arange(nx, dtype=np.float32) * 25.0
    y = np.arange(ny, dtype=np.float32) * 25.0
    z = np.arange(nz, dtype=np.float32) * 2.0 + 1200.0
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    zn = (Z - Z.min()) / (Z.max() - Z.min())
    facies = np.zeros((nx, ny, nz), dtype=np.int16)
    facies[(zn > 0.32) & (zn < 0.55)] = 1
    facies[(zn >= 0.55) & (zn < 0.68)] = 2
    facies[zn >= 0.68] = 3
    poro = 8.0 + 14.0 * (facies == 1) + 8.0 * (facies == 2) + rng.normal(0, 1.0, facies.shape)
    sog = 18.0 + 52.0 * (facies == 1) + 30.0 * (facies == 2) + rng.normal(0, 3.0, facies.shape)
    gr = 145.0 - 65.0 * (facies == 1) - 35.0 * (facies == 2) + rng.normal(0, 4.0, facies.shape)
    bi = 35.0 + 28.0 * (facies == 1) + 16.0 * (facies == 2) + rng.normal(0, 2.0, facies.shape)
    sh1 = 30.0 + 0.015 * (Z - Z.min()) + 2.0 * (facies == 3) + rng.normal(0, 0.4, facies.shape)
    sh2 = sh1 - (4.0 + 1.5 * (facies == 1) + rng.normal(0, 0.3, facies.shape))
    perm = np.exp(1.5 + 0.12 * poro + rng.normal(0, 0.25, facies.shape))
    return pd.DataFrame(
        {
            "x": X.ravel(),
            "y": Y.ravel(),
            "z": Z.ravel(),
            "PORO": poro.ravel(),
            "GR": gr.ravel(),
            "SOG": np.clip(sog, 0, 100).ravel(),
            "SH": (0.5 * (sh1 + sh2)).ravel(),
            "BI": np.clip(bi, 0, 100).ravel(),
            "sh1": sh1.ravel(),
            "sh2": sh2.ravel(),
            "PR": perm.ravel(),
            "OilPhase": facies.ravel(),
        }
    )


def save_figures(
    grid,
    stress_features: Mapping[str, Any],
    elastic_no: Mapping[str, Any],
    elastic_stress: Mapping[str, Any],
    elastic_physical: Mapping[str, Any],
    seismic: Mapping[str, Any],
    seismic_physical: Mapping[str, Any],
    horizon: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Save representative paper figures."""
    dpi = int(config.get("export", {}).get("figures_dpi", 300))
    fig_dir = output_dir / "figures"
    save_slice_panel({k: elastic_stress[k] for k in ["Vp", "Vs", "density", "VpVs", "AI", "SI"] if k in elastic_stress}, fig_dir / "elastic_softdata_slices.png", dpi=dpi)
    save_slice_panel({k: seismic[k] for k in ["poststack_seismic", "AVO_intercept", "AVO_gradient"] if k in seismic}, fig_dir / "seismic_softdata_slices.png", dpi=dpi)
    save_slice_panel(horizon, fig_dir / "horizon_softdata_slices.png", dpi=dpi)
    save_difference_panel(elastic_no, elastic_stress, ["Vp", "Vs", "density", "AI", "VpVs"], fig_dir / "stress_aware_minus_baseline.png", dpi=dpi)
    context = {k: grid.properties[k] for k in ["facies", "porosity", "oil_saturation", "permeability", "gamma", "stress", "SH1", "SH2"] if k in grid.properties}
    context.update({k: stress_features[k] for k in ["mean_stress_proxy", "differential_stress", "stress_anisotropy_index", "pore_pressure_MPa", "vertical_stress_Sv_MPa", "effective_pressure_physical_MPa"] if k in stress_features})
    save_volume_orthoslices(context, fig_dir / "nature_harddata_stress_context.png", dpi=dpi, max_items=6)
    save_volume_orthoslices({k: elastic_stress[k] for k in ["Vp", "Vs", "density", "VpVs", "AI", "SI"] if k in elastic_stress}, fig_dir / "nature_elastic_orthoslices.png", dpi=dpi, max_items=6)
    save_volume_orthoslices({k: stress_features[k] for k in ["pore_pressure_MPa", "vertical_stress_Sv_MPa", "effective_pressure_physical_MPa", "overpressure_ratio", "effective_pressure_ratio"] if k in stress_features}, fig_dir / "nature_geomechanics_pressure_context.png", dpi=dpi, max_items=5)
    save_difference_panel(elastic_stress, elastic_physical, ["Vp", "Vs", "density", "AI", "VpVs"], fig_dir / "physical_peff_minus_stress_proxy.png", dpi=dpi)
    save_volume_orthoslices({**{k: seismic[k] for k in ["poststack_seismic", "AVO_intercept", "AVO_gradient"] if k in seismic}, **{k: horizon[k] for k in ["horizon_probability", "signed_distance_field"] if k in horizon}}, fig_dir / "nature_seismic_horizon_orthoslices.png", dpi=dpi, max_items=5)
    save_volume_orthoslices({**{k: seismic_physical[k] for k in ["poststack_seismic", "AVO_intercept", "AVO_gradient"] if k in seismic_physical}, **{k: elastic_physical[k] for k in ["effective_pressure_proxy_MPa", "AI"] if k in elastic_physical}}, fig_dir / "nature_physical_peff_softdata.png", dpi=dpi, max_items=5)
    if "facies" in grid.properties:
        save_crossplot(grid.properties["porosity"], elastic_stress["AI"], grid.properties["facies"], fig_dir / "porosity_ai_crossplot.png", "porosity", "AI", dpi=dpi)
        save_crossplot(grid.properties["oil_saturation"], seismic["AVO_gradient"], grid.properties["facies"][:, :, :-1], fig_dir / "oil_saturation_avo_gradient_crossplot.png", "oil saturation", "AVO gradient", dpi=dpi)


def setup_logging(log_path: Path) -> None:
    """Configure file and console logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _torch_to_numpy_dict(tensors: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in tensors.items():
        if hasattr(value, "detach"):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
