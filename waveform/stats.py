"""Frozen slot-level analysis. / 以物理时隙重算统计。"""
import math
import numpy as np
import scipy.linalg
from scipy.stats import t as student_t
def cluster_ratio(numerator, denominator, minimum_slots=20):
    """整槽ratio的delta法SE；零分母槽保留在抽样总体。 / Slot-cluster delta SE for ratios."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b < 0):
        raise ValueError("need finite matched slot vectors and nonnegative denominators")
    support = int(np.sum(b > 0))
    if b.sum() == 0:
        return dict(mean=None, cluster_se=None, ci95=None, contributing_slots=0)
    estimate = float(a.sum() / b.sum())
    influence = (a - estimate * b) / b.mean()
    se = float(influence.std(ddof=1) / math.sqrt(len(a)))
    width = float(student_t.ppf(.975, len(a) - 1)) * se
    return dict(mean=estimate, cluster_se=se if support >= minimum_slots else None,
                ci95=[estimate - width, estimate + width] if support >= minimum_slots else None,
                contributing_slots=support)

def tensor_features(gamma, error_over_noise, degree):
    """固定光滑标量特征，无接收机训练。 / Fixed smooth scalar features for diagnostic regression only."""
    u = np.log1p(gamma) / math.log1p(10000.)
    v = np.log1p(error_over_noise) / math.log1p(100.)
    return np.stack([u ** i * v ** j for i in range(degree + 1) for j in range(degree + 1)], -1)

def ridge_predict(train_x, train_y, test_x, penalty):
    """只用训练折标准化，截距不惩罚。 / Standardize using training folds; leave the intercept unpenalized."""
    mean, scale = train_x.mean(0), train_x.std(0)
    mean[0], scale[0] = 0., 1.
    scale = np.where(scale < 1e-12, 1., scale)
    x, z = (train_x - mean) / scale, (test_x - mean) / scale
    gram = x.T @ x / len(x)
    regularizer = np.full(x.shape[1], penalty)
    regularizer[0] = 0.
    coef = scipy.linalg.solve(gram + np.diag(regularizer), x.T @ train_y / len(x), assume_a="pos")
    return z @ coef

def paired_prediction_bootstrap(difference, mask, repetitions, seed):
    """固定折预测的整槽配对bootstrap，不重新拟合回归。 / Paired slot bootstrap conditional on fitted fold predictions."""
    sums=np.sum(difference*mask,1);count=mask.sum(1)
    summary=cluster_ratio(sums,count)
    if np.sum(count>0)<20:
        return summary
    rng=np.random.default_rng(seed)
    draws=[]
    for start in range(0,repetitions,64):
        idx=rng.integers(len(count),size=(min(64,repetitions-start),len(count)))
        den=count[idx].sum(1)
        draws.extend(np.divide(sums[idx].sum(1),den,out=np.full(len(den),np.nan),where=den>0).tolist())
    summary["ci95"]=np.quantile(draws,[.025,.975]).tolist() if np.isfinite(draws).all() else None
    summary["interval_method"]="paired whole-slot percentile bootstrap, conditional on fitted folds"
    return summary

def compare(target, gamma, error_over_noise, direction_error, folds, settings):
    """仅拟合新增标量模型；按槽配对。 / Fit only the new scalar model; pair by slot."""
    slots, draws = target.shape
    grid = tensor_features(gamma.ravel(), error_over_noise.ravel(), degree=6)
    features = grid[:, :-1]  # (6,6) 是最后一列。 / (6,6) is the last column.
    if features.shape != (slots * draws, 48) or not np.all(features[:, 0] == 1):
        raise ValueError("Expected exactly 48 columns including an intercept")
    y = target.ravel()
    prediction = np.empty_like(y)
    row_folds = np.repeat(folds, draws)
    for fold in range(settings["folds"]):
        held_out = row_folds == fold
        prediction[held_out] = ridge_predict(
            features[~held_out], y[~held_out], features[held_out], settings["ridge"]
        )
    scalar = np.square(prediction.reshape(target.shape) - target).mean(axis=1)
    direction = direction_error.mean(axis=1)
    delta = scalar - direction
    if not np.isfinite([scalar, direction]).all() or np.any(scalar <= 0):
        raise ValueError("Nonfinite error or nonpositive scalar denominator")
    # 两个统计量均在每次整槽重采样中重算。 / Recompute both contrasts for each slot resample.
    rng = np.random.default_rng(settings["bootstrap_seed"])
    samples = np.empty((settings["bootstrap_repetitions"], 2))
    for start in range(0, len(samples), 64):
        end = min(start + 64, len(samples))
        index = rng.integers(slots, size=(end - start, slots))
        sampled_delta = delta[index].mean(axis=1)
        samples[start:end, 0] = sampled_delta
        samples[start:end, 1] = 100 * sampled_delta / scalar[index].mean(axis=1)
    intervals = np.quantile(samples, [.025, .975], axis=0).T
    return {
        "slot_count": slots, "re_per_slot": draws,
        "scalar_mspe": float(scalar.mean()), "direction_mspe": float(direction.mean()),
        "scalar_minus_direction_mspe": float(delta.mean()), "difference_ci95": intervals[0].tolist(),
        "reduction_percent": float(100 * delta.mean() / scalar.mean()),
        "reduction_percent_ci95": intervals[1].tolist(),
        "fold_mspe_differences": [float(delta[folds == k].mean()) for k in range(settings["folds"])],
    }
