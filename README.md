# Channel Error Orientation and Calibrated Bit Loss in SIMO Receivers

Code and derived data for reproducing the figures and receiver comparisons in this paper. The theory code evaluates a two-branch 64-QAM receiver. The waveform analysis starts from frozen receiver outputs on 10,240 simulated NR-NTN slots across profiles A–D. The [protocol](protocol.json) lists the waveform settings, sample sizes and LLR calibration scales.

## Install

Run these commands from the repository root on Linux. The scripts run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Reproduce Figures 1 and 2

```bash
python theory/replay.py --mode quick --output outputs/theory.json
python waveform/replay.py --output outputs/waveform.json
python figures/plot.py --theory-result outputs/theory.json --waveform-result outputs/waveform.json --output-dir outputs
```

**Figure 1** is written to `outputs/fig1.pdf`. Quick mode combines the archived integration runs and directly recomputes selected points. To recompute the entire theory grid, use `--mode full` in the first command. See the [theory settings](theory/README.md) for the integration grid and accuracy criteria.

**Figure 2** is written to `outputs/fig2.pdf`. The waveform script recomputes gain-band means, refits both predictors on the same five whole-slot folds, and writes the estimates and intervals to `outputs/waveform.json`. Across the 12 estimator–profile pairs, the directional model reduces held-out loss-prediction MSPE by **0.48%–3.23%** relative to the 48-coefficient scalar model. For example, `["A"]["glsnet"]["matched"]["reduction_percent"]` gives the **2.46%** reduction for GLSNet on profile A. See the [waveform data dictionary](waveform/README.md) for units and group definitions.

## Reproduce the receiver comparisons

```bash
python controls/replay.py --output outputs/controls.json
python controls/mode_check.py
```

The first command recomputes the paired whole-slot comparisons. **14 of 18** primary contrasts pass the one-sided criterion. These comprise 8 of 8 comparisons with the initial receiver, 5 of 8 full- versus diagonal-curvature comparisons, and 1 of 2 profile-A interactions. The second command prints the phase-minus-orthogonal loss differences for all eight LMMSE/GLSNet profile pairs. See the [control definitions](controls/README.md) for the receiver arms, comparisons and subgroup intervals.

## Check individual resource elements and the representative SNR

```bash
python examples/check.py
python examples/check_scope.py
```

The first command computes bit LLRs, excess loss and orientation fractions from 48 example resource elements for each of the three estimators. The second recovers the representative combining SNR of **154.115** (linear) and median slot relative RMS errors of **0.0340** for LMMSE and **0.0478** for GLSNet. The [examples](examples/README.md) describe the inputs.

## 中文说明

在仓库根目录安装依赖，运行上面的理论、波形统计和绘图命令，即可在 `outputs/` 得到两张 PDF 及对应的数值结果。接收机对照检验写入 `outputs/controls.json`。波形部分使用已冻结的接收机输出；完整波形生成和网络训练不在本仓库的复现范围内。
