# Codex Change Log

## 2026-05-05

- Added `research_pipelines/stress_conditioned_softdata`, a TGRS-oriented 3D geological soft-data pipeline.
- Implemented schema mapping, XYZ-to-grid reconstruction, QC, stress proxy features, stress-conditioned SoftSand/Gassmann rock physics, Shuey/Ricker seismic soft data, horizon probability/SDF generation, uncertainty hooks, visualization, export, and ablation metrics.
- Added synthetic demo support and unit tests for grid recovery, stress features, rock physics, seismic soft data, and visualization edge cases.
- Verified with 6 unit tests, a synthetic end-to-end demo, and a full run on `grid.csv` to `/tmp/geobrain_softdata_grid`.
- Audit note: GeoBrain rock physics and wave/AVO modules are reused directly where available; facies-to-horizon soft constraints use a custom fallback because the implicit module is point/orientation-model based.
