"""Common-support waveform mode comparison. / 从共同支持分箱核验模态关联的符号。"""
import json
from pathlib import Path

import numpy as np


def mode_results():
    """Energy-dominant phase versus complex-orthogonal group, all data REs."""
    rows = []
    with np.load(Path(__file__).resolve().parent / "data/phase_cells.npz", allow_pickle=False) as archive:
        for profile in "ABCD":
            for base in ("joint_lmmse", "glsnet"):
                key = f"{profile}_{base}_"
                count = archive[key + "size"]
                support = archive[key + "support"]
                sums = archive[key + "sums"]
                # Both groups must have enough RE and independent-slot support.
                eligible = np.all((count >= 200) & (support >= 20), axis=1)
                weights = count.min(1) * eligible
                weights = weights / weights.sum()
                means = np.divide(sums, count[..., None], out=np.zeros_like(sums),
                                  where=count[..., None] > 0)
                standardized = np.einsum("c,cmk->mk", weights, means)
                difference = float(standardized[0, 0] - standardized[1, 0])
                np.testing.assert_allclose(difference, archive[key + "expected"], atol=2e-14, rtol=2e-12)
                rows.append(dict(profile=profile, base=base, eligible_cells=int(eligible.sum()),
                                 phase_minus_orthogonal_excess_ce_nats_per_bit=difference,
                                 phase_orthogonal_weighted_mean_gamma=standardized[:, 1].tolist(),
                                 phase_orthogonal_weighted_mean_relative_energy=standardized[:, 2].tolist()))
    return rows


if __name__ == "__main__":
    print(json.dumps(mode_results(), indent=2, allow_nan=False))
