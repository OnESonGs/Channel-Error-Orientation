# Resource-element checks

These are simulated observations, not measurements or full OFDM waveforms.
Run `python examples/check.py` from the project root. The checker enumerates
all 64 unit-power NR Gray-labeled symbols rather than using the original
axis-factorized receiver implementation.

There are 12 REs per profile: first, middle and last physical test slots
and indices 0, 21, 42, 63 within each slot's existing 64-RE diagnostic draw
(zero based). This deterministic selection does not use the outcome.
It is for checking conventions and arithmetic, not estimating performance.

In the NPZ file the leading dimension is RE. `slot_id` is the pair
(generation seed, scenario ID); `data_re_index` is the symbol-major index
among the 972 payload REs; `fold` and `sample_index` identify the associated
diagnostic observation. `H` is the true diagonal frequency response, `Y` the
actual two-branch FFT observation, and `variance` is complex noise power per
branch. `bits` contains six actual scrambled coded-bit labels, in alternating
I/Q order. `estimates`, `scales` and the expected arrays have estimator order
joint LMMSE, GLSNet, DnCNN. `expected_logits` are **unscaled**, positive for bit
zero; `expected_target_nats` is scaled hard-label mean CE minus the unit-scale
true-diagonal reference CE, in **nats/bit** (multiply by 6/ln 2 for bits/data
RE). Modal fractions are radial, common phase, complex orthogonal relative to
the true channel. The reference is not an oracle posterior for the ICI waveform.

`python examples/check_scope.py` independently selects the profile-A slot
nearest the median slot SNR in dB and its RE nearest the median within-slot
linear SNR, using all 4,096 slot means and the 972 gains for the originally
selected slot. The two columns of slot error energies are Joint LMMSE and
GLSNet. This also reproduces the medians of per-slot relative RMS error; it
does not assert that the measured errors have the Gaussian error law.
