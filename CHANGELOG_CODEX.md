# Codex Change Log

## 2026-05-05

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
