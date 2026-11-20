"""Recompute all reported frozen-head contrasts. / 重算冻结接收机的配对比较。"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from head import apply_head

HERE = Path(__file__).resolve().parent
ARMS = ("base", "old_D", "old_F", "new_D", "new_F", "new_MSE", "rotation")
GROUPS = ("all", "deep", "low", "moderate", "strong", "spectral_notch", "no_spectral_notch")
CONTRASTS = {
    "newF_newD": (("new_F", 1), ("new_D", -1)),
    "newF_base": (("new_F", 1), ("base", -1)),
    "interaction": (("new_F", 1), ("new_D", -1), ("old_F", -1), ("old_D", 1)),
    "newF_MSE": (("new_F", 1), ("new_MSE", -1)),
    "newF_rotation": (("new_F", 1), ("rotation", -1)),
}


def confidence(draws, tail):
    return [float(x) for x in np.quantile(draws, [tail, 1-tail])]


def contrasts(profile, snr, rates):
    bases = ["joint_lmmse", "glsnet", "dncnn"] if profile == "A" else ["joint_lmmse", "glsnet"]
    columns, definitions = [], []
    for b, name in enumerate(bases):
        for key, terms in CONTRASTS.items():
            columns.append(sum(w * rates[:, b, ARMS.index(arm)] for arm, w in terms))
            primary = (name != "dncnn" and
                       (key in ("newF_newD", "newF_base") or (key == "interaction" and profile == "A")))
            definitions.append((name, key, primary))

    # Whole-slot draws; never resample antennas, REs or individual bits as independent.
    values = np.stack(columns, axis=1)
    strata = np.floor((snr-15.0) / 5.0)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    rng = np.random.default_rng(991301)
    samples = np.empty((20000, values.shape[1]))
    for j in range(len(samples)):
        index = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in groups])
        samples[j] = values[index].mean(axis=0)
    out = []
    for j, (name, key, primary) in enumerate(definitions):
        tail = 0.05 / 18 if primary else 0.025
        lower, upper = confidence(samples[:, j], tail)
        mean = float(values[:, j].mean())
        out.append(dict(base=name, comparison=key, role="primary" if primary else "secondary",
                        tail=tail, mean_gain=mean, lower_bound=lower, upper_bound=upper,
                        confirmed=(mean > 0 and lower > 0) if primary else None))
    return out


def subgroups(counts, ce_sum):
    out = {}
    # CE entries sum the six-bit average (nats/bit) over the data REs in each group.
    conversion = 6 / np.log(2)
    for b, base in enumerate(("joint_lmmse", "glsnet")):
        for j, group in enumerate(GROUPS):
            n = counts[:, j]
            n_re = int(n.sum())
            support = int(np.count_nonzero(n))
            result = dict(RE_count=n_re, slots=support, MSE_rate=None, gain=None, interval=None)
            if n_re:
                loss_f, loss_mse = ce_sum[:, b, j, :].T
                result["MSE_rate"] = float(6 - conversion * loss_mse.sum() / n_re)
                gain_sum = conversion * (loss_mse - loss_f)
                mean = float(gain_sum.sum() / n_re)
                result["gain"] = mean
                if support >= 20:
                    influence = (gain_sum - mean * n) / n.mean()
                    half = float(t.ppf(.975, len(n)-1)) * influence.std(ddof=1) / np.sqrt(len(n))
                    result["interval"] = [float(mean-half), float(mean+half)]
            out[f"{base}/{group}"] = result
    return out


def check(actual, expected):
    """Recursively check every reference number and every missing/unconfirmed result."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Reference label mismatch")
        for key in expected:
            check(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError("Reference length mismatch")
        for a, e in zip(actual, expected):
            check(a, e)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-11)
    elif actual != expected:
        raise ValueError(f"Reference mismatch: {actual!r} vs {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write full contrast and subgroup results to this path.")
    args = parser.parse_args()
    out, seen = {}, set()
    for profile in "ABCD":
        with np.load(HERE / "data" / f"{profile}.npz", allow_pickle=False) as a:
            ids = set(map(tuple, a["slot_id"]))
            if len(ids) != len(a["rates"]) or any(i in seen for i in ids):
                raise ValueError("Physical slot identifiers must be unique across profiles")
            seen.update(ids)
            counts = a["group_size"].astype(float)
            np.testing.assert_array_equal(counts[:, 0], np.full(len(counts), 972))
            np.testing.assert_array_equal(counts[:, 0], counts[:, 1:5].sum(1))
            np.testing.assert_array_equal(counts[:, 0], counts[:, 5:].sum(1))
            result = contrasts(profile, a["snr_db"], a["rates"])
            out[profile] = dict(contrasts=result, subgroups=subgroups(counts, a["ce_sum"]))
    check(out, json.loads((HERE / "reference.json").read_text(encoding="utf-8")))
    with np.load(HERE / "data" / "sample_head.npz", allow_pickle=False) as a:
        for j, arm in enumerate(ARMS):
            h = apply_head(a["h0"], a["h_p"], a["covariance"], a["variance"], a["alpha"][j], arm)
            np.testing.assert_allclose(h, a["expected"][:, j], atol=8e-14, rtol=2e-13)
    primary = [c for row in out.values() for c in row["contrasts"] if c["role"] == "primary"]
    print(f"Verified {len(seen)} distinct slots; {sum(c['confirmed'] for c in primary)}/{len(primary)} "
          "primary contrasts passed. Secondary comparisons and seven frozen-head examples also match.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
