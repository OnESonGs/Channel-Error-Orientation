# Matched-model numerical verification

Run from the release root (NumPy and SciPy, CPU):

```bash
python theory/replay.py --mode quick --output outputs/theory.json
# The following command regenerates the complete integration grids. Substantially slower:
python theory/replay.py --mode full --output outputs/theory-full.json
```

`orientation.json` and `scope.json` contain the exact model, scrambled Sobol
seeds, nested powers and frozen per-scramble scalar records. Neither file
contains receiver training data. In the model, h is fixed, noise is circular
Gaussian with variance r²/γ per complex branch, and errors are independent,
proper, centered Gaussian with covariance ε²C. The two eigenvalues of C
are positive and distinct. The constellation is the NR Gray-labeled 64-QAM
alphabet, not a Gaussian input.

- **summary:** reaggregate the frozen scramble records (and compute the
  Gauss–Hermite curvature reference); does not generate new Sobol points.
- **quick** (default): also regenerate the first scramble, smallest archived
  nested power, first SNR and amplitude at x=0, 0.5, 1 in each study. Check
  calibrated risk, shared scale and orientation derivative against archived
  values. The other settings in the reported outputs still come from archived
  per-scramble evaluations.
- **full:** directly regenerate every grid point and scramble and compare
  scales, risks and derivatives with the archived values. Allocates arrays for one
  scramble/setting at a time, up to 2^18 points, ± error, and six bits.
  Runtime depends strongly on CPU (budget hours rather than seconds).

Each scramble approximates one common nonnegative LLR scale outside the
observation, bit and error averages, not a scale per sample. Monte Carlo
uncertainty is across independent scrambles, paired across orientations.
Figure 1(b) is the γ=10 slice normalized by ε²(κ₁−κ₂)d/4; the 72 numerical
adjacent comparisons additionally include γ=3 and 30. The 36 adjacent checks
at γ≈154.115 and the 5%/10% risk-gap accuracy prefix use a different grid
(9 amplitudes × 5 orientations). Numerical ordering, derivative diagnostics
and approximation accuracy are distinct; a grid does not certify the uniform
interval proved in the article. No derivative-convergence claim is made for
settings that merely have positive point estimates. See also
`python examples/check_scope.py` for the waveform scale comparison.
