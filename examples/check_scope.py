"""Profile-A scale anchoring using frozen slot sums. / 以冻结时隙统计核验理论与实测尺度的对照。"""
from pathlib import Path
import numpy as np


def nearest(value, target, ids):
    distance = abs(value - target)
    # Resolve numerical ties by physical identifier. / 近似并列时按物理编号选取。
    tol = 16 * np.finfo(float).eps * max(1., abs(target))
    candidates = np.flatnonzero(distance <= distance.min() + tol)
    return min(candidates, key=lambda k: tuple(np.atleast_1d(ids[k])))


with np.load(Path(__file__).with_name("scope_inputs.npz"), allow_pickle=False) as a:
    db = 10 * np.log10(a["mean_gamma"])
    slot = nearest(db, np.median(db), a["slot_ids"])
    assert slot == int(a["selected_slot_index"])
    gains = a["selected_slot_re_gamma"]
    re = nearest(gains, np.median(gains), np.arange(len(gains)))
    gamma = gains[re]
    rms = np.median(np.sqrt(a["error"] / a["signal"][:, None]), axis=0)
    np.testing.assert_allclose(gamma, 154.11523261649407, rtol=1e-12)
    np.testing.assert_allclose(rms, [.03402960876, .04781487353], atol=1e-10, rtol=0)
    print(f"Representative combining SNR: {gamma:.8f} (linear); slot/RE indices {slot}/{re}.")
    print(f"Median slot relative RMS errors: LMMSE {rms[0]:.8f}, GLSNet {rms[1]:.8f}.")
