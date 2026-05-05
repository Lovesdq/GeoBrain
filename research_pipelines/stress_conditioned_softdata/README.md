# Stress-Conditioned 3D Geological Soft-Data Pipeline

This research pipeline converts a regular 3D geological point table into
stress-conditioned rock-physics, seismic, horizon and uncertainty soft data for
TGRS-style experiments.

## GeoBrain Audit

Directly reused:

- `geobrain.physics.rock`: `SoftSand`, `HertzMindlin` through `SoftSand`,
  `Gassmann`, `DensityModel`, `v_from_moduli`, mineral/fluid databases.
- `geobrain.physics.wave`: `compute_reflectivity`, `Shuey`, `RickerWavelet`,
  `create_conv_matrix`.
- `geobrain.io`: `write_segy` is wrapped as an optional export path when
  `segyio` is installed.
- `geobrain.geomodel.implicit`: available for point/orientation constrained
  implicit models, but not a direct full-grid facies-to-SDF interface.

New glue/custom implementation:

- CSV/Parquet schema mapping, facies encoding and `metadata.json`.
- XYZ point table to `[nx, ny, nz]` grid recovery with duplicate/missing checks.
- QC statistics, physical-range warnings and 300 dpi figures.
- Stress proxy features and stress-conditioned effective-pressure mapping.
- Connolly-style elastic impedance and stress/brittleness sweet-spot proxies.
- Facies/gradient-derived horizon probability and signed-distance fields.
- Monte Carlo perturbation summaries and ablation metrics.

Important limitation:

- With only SH1/SH2 and without pore pressure, vertical stress, fracture azimuth
  or stress direction, outputs must be described as `stress-informed` or
  `stress-conditioned` soft-data simulation, not strict 3D geomechanical forward
  modeling.

## Quick Demo

Run a deterministic synthetic small grid:

```bash
cd /home/likunxi/data/GeoBrain
python research_pipelines/stress_conditioned_softdata/run_pipeline.py \
  --demo \
  --device cpu \
  --output outputs/stress_conditioned_softdata_demo
```

Main outputs:

- `metadata.json`: resolved columns, units and facies map.
- `qc/qc_report.json` and `qc/figures/`: input data checks.
- `exports/*.npz` and `exports/*.pt`: grid, stress, elastic, seismic and horizon volumes.
- `figures/`: slice panels, crossplots and stress-aware minus baseline maps.
- `ablation/metrics.csv`: paper-oriented experiment metrics.

## Running On Your Grid

The repository currently contains `grid.csv` with columns like:
`x,y,z,PORO,GR,SOG,BI,sh1,sh2,OilPhase`.

```bash
cd /home/likunxi/data/GeoBrain
python research_pipelines/stress_conditioned_softdata/run_pipeline.py \
  --config research_pipelines/stress_conditioned_softdata/config.yaml \
  --input grid.csv \
  --output outputs/stress_conditioned_softdata_grid \
  --device cuda:4
```

If CUDA device 4 is unavailable, the code logs a warning and falls back to CPU.
For a Parquet table, pass `--input your_grid.parquet`; pandas/pyarrow must be
available in the environment.

Edit `config.yaml` to update:

- Column aliases under `columns`.
- Required/optional properties and facies code names under `schema`.
- Stress units and effective-pressure proxy scaling under `stress`.
- Facies mineral mixtures, fluid names and SoftSand/Gassmann parameters under
  `rock_physics`.
- AVO angles, wavelet frequencies, noise and filtering under `seismic`.

## Experiment Design Text

The experimental workflow first restores the uniform point table to a regular
3D grid and performs geometry, missing-node, duplicate-node, unit and outlier
QC. It then builds stress proxy features from SH1/SH2, including mean stress
proxy, differential stress, stress ratio and stress anisotropy index. A no-stress
baseline uses constant effective pressure in the GeoBrain SoftSand-Gassmann
chain. The stress-aware run replaces the constant effective pressure with a
spatial stress proxy, and a weak-physics mode can instead apply empirical
stress corrections to Vp, Vs and density.

Elastic outputs are converted to AI, SI, Vp/Vs and elastic impedance at the
configured angles. Shuey reflectivity is computed between adjacent depth
samples, convolved with Ricker wavelets to synthesize post-stack and pre-stack
angle volumes, and optionally perturbed with Gaussian noise/band-limited
filtering. Horizon soft constraints are extracted from reservoir facies or
attribute gradients as top/base maps, probability halos and signed-distance
fields. Ablation experiments compare hard-data statistics, no-stress versus
stress-aware rock physics, no-stress versus stress-aware seismic, uncertainty
sensitivity and with/without horizon constraints.

## Method Flowchart Text

Input point table -> schema mapping and facies encoding -> regular grid
reconstruction/QC -> stress proxy feature generation -> no-stress and
stress-aware SoftSand-Gassmann rock physics -> elastic attributes/EI -> Shuey
reflectivity and Ricker convolution -> noisy seismic soft data -> facies/gradient
horizon probability and SDF -> uncertainty summaries, ablation metrics, figures
and NPZ/PT/optional VTK/SEG-Y export.
