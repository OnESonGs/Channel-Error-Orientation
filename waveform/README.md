# Statistical replay from frozen waveform-derived observations

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python waveform/replay.py --output outputs/waveform.json
```

No training or private inference cache is needed. The code fits predictors
from scratch: a degree-3×3 tensor scalar basis (16 columns) and its products
with the radial and common-phase fractions form the 48-column directional
design. The 48-column scalar basis uses powers 0–6 on each coordinate except
(6,6). Controls also include the original 36-column scalar design where
reported. All models include an intercept, standardize nonconstant columns
on **training folds only**, and use the same mean-loss ridge coefficient
1e-4. The held-out fold consists of complete physical slots. Group means
use all 972 REs per slot; the diagnostic regressions use 64 presampled REs.
`expected.json` contains the frozen plot statistics (including undefined
intervals where applicable) and paired prediction results for comparison.
For the common-support phase/orthogonal diagnostic see `controls/mode_check.py`.

## Data dictionary

Each lowercase profile prefix a–d has one `*_slots.npz` and three estimate
files. Arrays do not require pickle. The slot order is shared within a profile.
The group dimension is always [weak, low, moderate, strong], with boundaries
−10, −5, 0 dB on 10 log10(P/2); the bins are exhaustive and left-inclusive
at each interior threshold.

- Shared slot file: `ids` is (generation seed, scenario ID); `folds` the locked
  0–4 fold; `selected` the 64 sorted, zero-based symbol-major data-RE indices
  per slot. `gamma` is P/t at those REs. `count`, `power`, `reference_ce_sum`,
  and `gamma_le_one_count` are counts or sums for each slot/gain bin. The CE
  reference uses unit LLR scale and true diagonal CSI. Relative-error sums
  for LMMSE/GLSNet permit checking the reported weak/strong squared-relative-
  error contrasts. These are sums of E/P, not ratios of two aggregate sums.
- Estimator file: `energy` and `excess` are the sum of squared complex channel
  error and the sum of signed excess CE (nats/bit summed over REs) in each
  slot/gain bin. `target`, `error_over_noise`, `radial`, and `phase` are
  sampled (slots,64) arrays; the latter two are fractions of error energy and
  the complex-orthogonal share is their complement. Targets are nats/bit.
  Divide by counts to obtain a group mean, not by the total slot count.
- LMMSE/GLSNet samples are taken from the original frozen arrays. DnCNN
  derived fields were exported using the same frozen predictions, labels and
  validation scales. Tiny roundoff differences between full-constellation and
  separable bit-log-MAP evaluations are tested numerically, not bitwise.
- Receiver rate is 6(1−mean estimated CE/ln 2) bits/data RE. Data-RE NMSE
  is 10 log10(ΣE/ΣP), not an average of per-slot dB values. Original grouped
  intervals are slot-cluster delta intervals; matched prediction intervals
  recompute the paired MSPE ratio in 2,000 whole-slot bootstrap draws without
  refitting. Zero-support intervals remain null. See [protocol](../protocol.json)
  for frozen validation scales and scenario settings.
