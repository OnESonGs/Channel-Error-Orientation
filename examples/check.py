"""Verify a few released synthetic REs with a direct 64-symbol soft demapper.
用独立的全星座计算核验少量合成 RE；不训练估计器。
"""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def logits(channel, received, variance):
    """Return six log P(bit=0)/P(bit=1) metrics. / 返回偏向零比特的六个 LLR。"""
    words = ((np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1)
    signs = 1 - 2 * words
    i = signs[:, 0] * (4 - signs[:, 2] * (2 - signs[:, 4]))
    q = signs[:, 1] * (4 - signs[:, 3] * (2 - signs[:, 5]))
    points = (i + 1j * q) / np.sqrt(42)
    residual = received[:, None, :] - channel[:, None, :] * points[None, :, None]
    score = -np.sum(abs(residual)**2, axis=-1) / variance[:, None]
    return np.stack([
        logsumexp(score[:, words[:, j] == 0], axis=1)
        - logsumexp(score[:, words[:, j] == 1], axis=1)
        for j in range(6)
    ], axis=-1)


def check():
    path = Path(__file__).with_name("resource_elements.npz")
    with np.load(path, allow_pickle=False) as data:
        H = data["H"].astype(np.complex128)
        Y = data["Y"].astype(np.complex128)
        bit_sign = 2.0 * data["bits"] - 1
        reference = np.logaddexp(0, bit_sign * logits(H, Y, data["variance"])).mean(-1)
        u = H / np.linalg.norm(H, axis=1, keepdims=True)
        for m in range(3):
            g = data["estimates"][:, m].astype(np.complex128)
            L = logits(g, Y, data["variance"])
            loss = np.logaddexp(0, data["scales"][:, m, None] * bit_sign * L).mean(-1)
            e = g - H
            parallel = (u.conj() * e).sum(-1)
            orthogonal = e - u * parallel[:, None]
            energies = np.stack([parallel.real**2, parallel.imag**2,
                                 np.sum(abs(orthogonal)**2, axis=-1)], axis=1)
            total = energies.sum(axis=1)
            for actual, key in ((L, "expected_logits"), (loss - reference, "expected_target_nats"),
                                (total, "expected_error_energy"),
                                (energies / total[:, None], "expected_fractions")):
                np.testing.assert_allclose(actual, data[key][:, m], atol=2e-12, rtol=2e-12)
        # Link each illustrative record back to the released regression inputs. / 逐项对齐公开的统计输入。
        for profile in "ABCD":
            mask = data["profile"] == profile
            prefix = path.parents[1] / "waveform" / profile.lower()
            with np.load(str(prefix) + "_slots.npz", allow_pickle=False) as common:
                lookup = {tuple(pair): i for i, pair in enumerate(common["ids"])}
                rows = [lookup[tuple(pair)] for pair in data["slot_id"][mask]]
                indices = (rows, data["sample_index"][mask])
                np.testing.assert_array_equal(common["selected"][indices], data["data_re_index"][mask])
                np.testing.assert_array_equal(common["folds"][rows], data["fold"][mask])
            for m, method in enumerate(("joint_lmmse", "glsnet", "dncnn")):
                with np.load(str(prefix) + "_" + method + ".npz", allow_pickle=False) as public:
                    for key, values in (("target", data["expected_target_nats"][mask, m]),
                                        ("radial", data["expected_fractions"][mask, m, 0]),
                                        ("phase", data["expected_fractions"][mask, m, 1])):
                        np.testing.assert_allclose(public[key][indices], values, atol=2e-12, rtol=2e-12)
        print(f"Verified {len(H)} REs × 3 frozen estimates (all four profiles).")


if __name__ == "__main__":
    check()
