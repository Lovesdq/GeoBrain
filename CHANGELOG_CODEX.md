# Codex Change Log

## 2026-05-05

- Fixed orthoslice publication figures where row colorbars and tick-offset numbers could overlap the third slice panel; colorbars now use dedicated right-side axes with reserved margins and non-offset tick formatting.
- Regenerated the affected `grid_peff/figures/nature_*` plates under `research_pipelines/stress_conditioned_softdata_outputs`.

- Enhanced publication visualization for the stress-conditioned soft-data pipeline with unit-aware colorbars, display-name normalization, panel lettering, and an integrated Nature-style summary plate `nature_integrated_softdata_summary.png`.
- Materialized the real `grid.csv` run output inside the repository at `research_pipelines/stress_conditioned_softdata_outputs/grid_peff`, including exports, QC, ablation metrics, logs, geomechanics metadata, and all figures.

- Added the `Pp/Sv` physical effective-pressure extension experiment: optional `pore_pressure/Pp` and `vertical_stress/Sv` columns are now supported, with deterministic depth-gradient proxies when columns are absent.
- Added `compute_geomechanics_features`, `sv_minus_pore_pressure` rock-physics mode, `elastic_physical_pressure` and `seismic_physical_pressure` export groups, and Exp-7/Exp-8 ablation comparisons against the existing stress-proxy workflow.
- Added `geomechanics_metadata.json`, Peff/Pp/Sv correlations, and new publication figures including `nature_geomechanics_pressure_context.png`, `physical_peff_minus_stress_proxy.png`, and `nature_physical_peff_softdata.png`.
- Updated Chinese README and `中文总结.md` with the `Peff=clip(Sv-Pp)` formula, default gradients, column-priority behavior, proxy-estimation limits, and current real-data interpretation boundaries.
- Verified the extension with 10 unit tests, a synthetic demo to `/tmp/geobrain_softdata_demo_peff`, and a full real-data run to `/tmp/geobrain_softdata_grid_peff`; current `grid.csv` uses gradient proxies because `Pp/Sv` columns are absent.

- Updated the stress-conditioned soft-data pipeline for the real `grid.csv` schema: `PR` is now permeability and `SH` is now a general stress proxy, while `sh1/sh2` remain vertical maximum/minimum stress proxies.
- Reworked the pipeline README into Chinese, added `中文总结.md`, and documented the full experiment workflow, outputs, field definitions, validation results, and interpretation limits.
- Improved publication visualization with Nature-style rcParams, robust percentile color limits, NaN gray rendering, symmetric seismic/difference scales, orthoslice plates, and enhanced facies crossplots.
- Fixed ablation bookkeeping so Exp-0 reports only original hard data while soft-data correlations can still use stress-derived features.
- Re-ran validation with 7 unit tests, a synthetic demo, and a full real-data run to `/tmp/geobrain_softdata_grid_v2`; confirmed grid shape `124 x 283 x 30`, `4185` missing nodes, no duplicates, and successful `PR/SH` mapping.
- Added `grid.csv` to the repository commit scope for reproducible local experiments.

## 2026-05-05 Initial Pipeline

- Added `research_pipelines/stress_conditioned_softdata`, a TGRS-oriented 3D geological soft-data pipeline.
- Implemented schema mapping, XYZ-to-grid reconstruction, QC, stress proxy features, stress-conditioned SoftSand/Gassmann rock physics, Shuey/Ricker seismic soft data, horizon probability/SDF generation, uncertainty hooks, visualization, export, and ablation metrics.
- Added synthetic demo support and unit tests for grid recovery, stress features, rock physics, seismic soft data, and visualization edge cases.
- Verified with 6 unit tests, a synthetic end-to-end demo, and a full run on `grid.csv` to `/tmp/geobrain_softdata_grid`.
- Audit note: GeoBrain rock physics and wave/AVO modules are reused directly where available; facies-to-horizon soft constraints use a custom fallback because the implicit module is point/orientation-model based.
