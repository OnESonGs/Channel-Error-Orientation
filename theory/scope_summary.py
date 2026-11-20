import math
import numpy as np
def relative_rms(epsilon_over_r, kappa):
    """总复误差RMS，非单支路标准差。 / Total complex-error RMS, not per-branch deviation."""
    values = np.asarray(kappa, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or not values[0] > values[1] > 0:
        raise ValueError("need kappa1 > kappa2 > 0")
    if not np.isfinite(epsilon_over_r) or epsilon_over_r < 0:
        raise ValueError("need finite nonnegative epsilon/r")
    return float(epsilon_over_r * math.sqrt(values.sum()))

def moments(values):
    """独立scramble的数值SE，不是时隙统计CI。 / Numerical SE across independent scrambles."""
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite independent replicates")
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))

def summarize(replicates, plan, reference, powers):
    """配对风险差；精度门槛不依赖符号。 / Paired risk gaps with sign-blind precision gates."""
    model, rule = plan["model"], plan["numerical_precision_rule"]
    delta = model["kappa"][0] - model["kappa"][1]
    summaries, adjacent = [], []
    for power in powers:
        for eps in model["epsilon_over_r"]:
            quadratic_endpoint = eps ** 2 * reference["d"] * delta / 4
            selected = {}
            for x in model["x"]:
                rows = sorted((r for r in replicates if r["power"] == power and r["epsilon_over_r"] == eps and r["x"] == x), key=lambda r: r["seed"])
                if [r["seed"] for r in rows] != sorted(plan["integration"]["scramble_seeds"]):
                    raise ValueError("missing or unpaired scrambles")
                selected[x] = np.array([r["fit"]["risk_nats_per_bit"] for r in rows])
            for x in model["x"][1:]:
                mean, se = moments(selected[x] - selected[0.])
                expected = quadratic_endpoint * x
                summaries.append(dict(power=power, epsilon_over_r=eps, relative_error_rms=relative_rms(eps, model["kappa"]), x=x,
                    mean_gap=mean, numerical_se=se, quadratic_gap=expected,
                    ratio_to_quadratic=mean / expected, ratio_numerical_se=se / expected))
            for low, high in zip(model["x"][:-1], model["x"][1:]):
                mean, se = moments(selected[high] - selected[low])
                adjacent.append(dict(power=power, epsilon_over_r=eps, x_low=low, x_high=high,
                    mean_gap=mean, numerical_se=se, quadratic_gap=quadratic_endpoint * (high - low)))
    high, low = powers[-2:][::-1]
    diagnostics = []
    for collection, keys, kind in ((summaries, ("epsilon_over_r", "x"), "gap_from_x0"),
                                   (adjacent, ("epsilon_over_r", "x_low", "x_high"), "adjacent_gap")):
        for row in (r for r in collection if r["power"] == high):
            previous = next(r for r in collection if r["power"] == low and all(r[k] == row[k] for k in keys))
            se = row["numerical_se"] / row["quadratic_gap"]
            drift = abs(row["mean_gap"] - previous["mean_gap"]) / row["quadratic_gap"]
            diagnostics.append(dict(kind=kind, **{k: row[k] for k in keys}, relative_se=se, relative_drift=drift,
                passes=se <= rule["se_relative_to_quadratic_gap"] and drift <= rule["nested_drift_relative_to_quadratic_gap"]))
    amplitude = []
    for eps in model["epsilon_over_r"]:
        rows = [r for r in summaries if r["power"] == high and r["epsilon_over_r"] == eps]
        gaps = [r for r in adjacent if r["power"] == high and r["epsilon_over_r"] == eps]
        precise = all(d["passes"] for d in diagnostics if d["epsilon_over_r"] == eps)
        deviation = max(abs(r["ratio_to_quadratic"] - 1) for r in rows)
        allowance = max(abs(r["ratio_to_quadratic"] - 1) + 2 * r["ratio_numerical_se"] for r in rows)
        amplitude.append(dict(epsilon_over_r=eps, relative_error_rms=relative_rms(eps, model["kappa"]), precision_pass=precise,
            max_relative_quadratic_deviation=deviation, max_deviation_plus_two_numerical_se=allowance,
            approximation={str(t): bool(precise and allowance <= t) for t in plan["approximation_tolerances"]},
            all_sampled_adjacent_means_positive=all(r["mean_gap"] > 0 for r in gaps),
            all_sampled_adjacent_means_above_two_numerical_se=bool(precise and all(r["mean_gap"] > 2 * r["numerical_se"] for r in gaps))))
    return dict(summaries=summaries, adjacent_gaps=adjacent, precision=diagnostics, amplitudes=amplitude,
        all_precision_targets_met=all(d["passes"] for d in diagnostics))

def sampled_prefix(amplitudes, tolerance):
    """只报告从最小采样幅度开始的连续通过点。 / Report only the initial passing sampled prefix."""
    last = None
    for row in sorted(amplitudes, key=lambda r: r["epsilon_over_r"]):
        if not row["approximation"][str(tolerance)]:
            return dict(last_passing_sample=last, first_nonpassing_sample=row["epsilon_over_r"],
                reason="quadratic_gap_deviation_exceeds_tolerance" if row.get("precision_pass", True) else "numerical_precision_target_not_met")
        last = row["epsilon_over_r"]
    return dict(last_passing_sample=last, first_nonpassing_sample=None, reason=None)
