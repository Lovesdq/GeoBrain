# 应力约束三维地质软数据科研流水线

本目录实现一套面向 TGRS / Nature 系列期刊论文实验的三维地质点云/规则网格软数据生成流水线。输入是长方体规则网格的点表 CSV/Parquet，输出包括岩石物理软数据、地震软数据、层位/结构软约束、应力耦合特征、消融实验指标、600 dpi 论文图件和可复现实验日志。

需要特别说明：当前数据只有 `SH`、`sh1`、`sh2` 等应力相关代理量，缺少孔隙压力、垂向总应力 Sv、裂缝方位和应力方向。因此本流水线在论文中应表述为 **stress-informed / stress-conditioned soft-data simulation**，不能夸大为严格三维地质力学正演。

## 1. GeoBrain 仓库审计结果

可以直接复用的能力：

- `geobrain.physics.rock`：`SoftSand`、`HertzMindlin`、`Gassmann`、`DensityModel`、`v_from_moduli`、矿物/流体数据库。
- `geobrain.physics.wave`：`compute_reflectivity`、`Shuey`、`RickerWavelet`、`create_conv_matrix`。
- `geobrain.io`：`write_segy` 可作为可选 SEG-Y 导出接口，前提是环境安装 `segyio`。
- `geobrain.geomodel.implicit`：已有隐式地质建模模块，适合由界面点和方向约束构造隐式模型。

需要新增胶水代码的部分：

- 点表字段映射、中文/英文别名配置、类别编码和 `metadata.json`。
- 从 `x,y,z` 自动恢复 `[nx, ny, nz]` 规则三维张量，并检查缺失点、重复点、单调性和网格间距。
- 将 `SH/sh1/sh2` 转换为应力代理特征，并将空间变化的 effective-pressure proxy 传入 GeoBrain `SoftSand`。
- 把 GeoBrain 的岩石物理输出继续组合成 `Vp/Vs/rho/AI/SI/EI`、AVO、合成地震和噪声版本。

自定义实现的部分：

- 从 facies 或属性突变提取顶/底界面、horizon probability volume 和 signed distance field。
- 论文级消融实验、相关系数、facies-wise 统计、separability 指标、stress-aware 与 baseline 差异图。
- 面向期刊图件的正交三切片图版、稳健百分位色标、facies crossplot 和高分辨率输出。

## 2. 当前真实数据字段约定

当前仓库根目录的 `grid.csv` 已按以下字段解释：

| 原始列名 | 物理含义 | pipeline 标准名 | 单位/约定 |
|---|---|---|---|
| `x` | X 坐标 | `x` | m |
| `y` | Y 坐标 | `y` | m |
| `z` | Z/深度坐标 | `z` | m，默认正向向下 |
| `PORO` | 孔隙度 | `porosity` | percent，内部转 fraction |
| `GR` | 自然伽马 | `gamma` | API |
| `SOG` | 含油饱和度 | `oil_saturation` | percent，内部转 fraction |
| `SH` | 地应力代理量 | `stress` | MPa |
| `PR` | 渗透率 | `permeability` | mD |
| `BI` | 脆性指数 | `brittleness_index` | percent，内部转 fraction |
| `sh1` | 垂向最大地应力/最大主应力代理 | `SH1` | MPa |
| `sh2` | 垂向最小地应力/最小主应力代理 | `SH2` | MPa |
| `OilPhase` | 储层分类标签 | `facies` | `0-3` 类别码 |

默认 facies 解释：

- `0`: `mudstone`，泥岩
- `1`: `oil_layer`，油层
- `2`: `poor_oil_layer`，差油层
- `3`: `dry_layer`，干层

字段别名、单位、facies 映射和岩石物理参数均在 `config.yaml` 中配置，不应在代码中硬编码。

## 3. 快速运行

推荐使用带 PyTorch 的环境：

```bash
cd /home/likunxi/data/GeoBrain
/home/likunxi/data/anaconda3/envs/pytorch_env/bin/python \
  research_pipelines/stress_conditioned_softdata/run_pipeline.py \
  --demo \
  --device cpu \
  --output outputs/stress_conditioned_softdata_demo
```

运行真实 `grid.csv`：

```bash
cd /home/likunxi/data/GeoBrain
/home/likunxi/data/anaconda3/envs/pytorch_env/bin/python \
  research_pipelines/stress_conditioned_softdata/run_pipeline.py \
  --config research_pipelines/stress_conditioned_softdata/config.yaml \
  --input grid.csv \
  --output outputs/stress_conditioned_softdata_grid \
  --device cuda:4
```

如果 4 号 GPU 不可用，代码会记录 warning 并自动降级到 CPU。当前测试环境中 `cuda:4` 不可用，真实数据在 CPU 下也能跑通。

## 4. 输出结构

一次完整运行会生成：

- `metadata.json`：字段映射、facies 编码、单位和输入列记录。
- `qc/qc_report.json`：网格形状、缺失点、重复点、异常值和范围警告。
- `qc/figures/`：各硬数据属性的直方图和切片 QC 图。
- `exports/grid_properties.*`：原始硬数据规则网格。
- `exports/stress_features.*`：应力派生代理量。
- `exports/elastic_no_stress.*`：无应力 baseline 岩石物理软数据。
- `exports/elastic_stress.*`：stress-conditioned 岩石物理软数据。
- `exports/seismic_stress.*`：stress-conditioned 反射系数、叠前/叠后合成地震和 AVO 属性。
- `exports/horizon_softdata.*`：储层 mask、顶/底界面、horizon probability 和 SDF。
- `figures/`：论文图件，包括普通切片图、差异图、crossplot 和 Nature 风格正交三切片图版。
- `ablation/metrics.csv`：Exp-0 到 Exp-6 的统计、相关系数、separability、差异和运行时间指标。

## 5. 科研方法路线

1. **点表到规则网格**：读取 CSV/Parquet，根据 `x,y,z` 自动推断 `nx,ny,nz,dx,dy,dz`，并检查规则性、缺失节点、重复节点和坐标单调性。
2. **硬数据 QC**：统计孔隙度、渗透率、伽马、含油饱和度、脆性指数、`SH/SH1/SH2` 的 min/max/mean/std/percentile、NaN/Inf 和物理范围警告。
3. **应力代理特征**：由 `sh1/sh2` 生成 `mean_stress_proxy`、`differential_stress`、`stress_ratio`、`stress_anisotropy_index`；由 `SH` 生成 `stress_MPa` 和归一化版本。
4. **岩石物理软数据**：复用 GeoBrain `SoftSand + Gassmann + DensityModel + v_from_moduli`。无应力 baseline 使用常数 effective pressure；强物理方案将空间变化的 `mean_stress_proxy` 映射为 effective-pressure proxy。
5. **弹性与阻抗属性**：输出 `Vp`、`Vs`、`density`、`Vp/Vs`、`AI`、`SI` 和多角度 `EI(angle)`。
6. **地震软数据**：复用 GeoBrain Shuey AVO、Ricker 子波和卷积矩阵，输出反射系数、叠后合成地震、12/24/36 度叠前地震、AVO intercept/gradient/curvature 和噪声版本。
7. **层位软约束**：从储层 facies 或属性突变提取顶/底界面，生成 horizon probability volume 和 signed distance field。
8. **消融实验**：比较硬数据统计、无应力岩石物理、应力感知岩石物理、无应力地震、应力感知地震、噪声/不确定性和 horizon soft constraints。

## 6. 实验设计文字版

实验首先将百万点级均匀长方体点表恢复为规则三维网格，并对几何完整性、属性范围和缺失值进行质量控制。随后构建两条岩石物理链路：第一条为无应力 baseline，使用常数 effective pressure；第二条为 stress-conditioned 链路，将空间变化的 `sh1/sh2` 平均应力代理映射为 effective pressure proxy，并将 `SH` 作为独立地应力代理属性参与相关性分析。两条链路均使用相同矿物、流体、孔隙度和 facies 参数，以保证对比只反映应力约束差异。

在弹性属性基础上，计算 AI、SI、Vp/Vs 和 EI，并用 Shuey 近似生成多角度反射系数，随后与 Ricker 子波卷积获得叠前/叠后合成地震体。储层类别标签用于提取顶/底界面、界面概率和 SDF，作为结构软约束。最终通过 facies-wise 统计、AI/VpVs/AVO gradient separability、硬数据-软数据相关系数、stress-aware 与 baseline 差异图以及运行时间/显存统计评价软数据质量。

## 7. 方法流程图文字版

输入点表 `grid.csv`  
-> 字段映射与 facies 编码  
-> 规则三维网格恢复与 QC  
-> `SH/sh1/sh2` 应力代理特征  
-> 无应力与应力感知 SoftSand-Gassmann 岩石物理  
-> `Vp/Vs/rho/AI/SI/EI` 弹性软数据  
-> Shuey AVO 反射系数与 Ricker 合成地震  
-> facies/属性突变层位概率与 SDF  
-> 不确定性、消融实验、论文图件和 NPZ/PT/可选 VTK/SEG-Y 导出。

## 8. 已验证情况

已验证项目：

- 单元测试覆盖网格恢复、应力特征、岩石物理、地震软数据和可视化边界条件。
- synthetic small grid 可以端到端运行。
- 当前真实 `grid.csv` 可恢复为规则网格候选；上一次运行识别到 `124 x 283 x 30` 网格、`4185` 个缺失节点、无重复节点。
- 原始数据和运行结果中存在缺失节点对应的 NaN，QC 报告会显式记录。

## 9. 论文表述边界

建议论文中使用以下表述：

- `stress-informed rock-physics soft-data simulation`
- `stress-conditioned effective-pressure proxy`
- `stress-aware seismic soft-data generation`

不建议使用：

- `full 3D geomechanical forward modeling`
- `true in-situ stress inversion`
- `mechanically rigorous coupled geomechanics`

除非后续补充孔隙压力、垂向总应力、裂缝方位、应力方向和力学边界条件。
