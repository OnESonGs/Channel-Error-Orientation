"""64-QAM bit-risk kernel and calibrated covariance-orientation comparison. / 定谱误差朝向的比特损失积分。"""
from functools import lru_cache
from numbers import Integral, Real
import math
import numpy as np
from scipy.stats import qmc
from scipy.special import expit, log_expit, logsumexp, ndtri, ive

def nr_qam_modulate(bits):
    """TS38.211 §5.1.3–6交替I/Q映射，最后一轴为2/4/6/8个比特。 / NR alternating-I/Q mapping with 2/4/6/8 bits on the last axis.
    星座平均功率归一。 / Unit ensemble constellation power.
    """
    bits = np.asarray(bits)
    if bits.ndim < 1 or bits.shape[-1] not in (2, 4, 6, 8) or not np.isin(bits, [0, 1]).all():
        raise ValueError("NR bits must have a binary final axis of length 2, 4, 6 or 8")
    qm = bits.shape[-1]
    axes = []
    for axis in (0, 1):
        signs = 1 - 2 * bits[..., axis::2].astype(np.float64)
        value = np.ones(bits.shape[:-1], dtype=np.float64)
        for position in range(qm // 2 - 1, 0, -1):
            value = 2 ** (qm // 2 - position) - signs[..., position] * value
        axes.append(signs[..., 0] * value)
    return (axes[0] + 1j * axes[1]) / np.sqrt(2 * ((1 << qm) - 1) / 3)

def _scalar(value, name, allow_zero=False):
    """检查有限实标量，不接受布尔值。 / Validate a finite real scalar, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value

@lru_cache(maxsize=8)
def _quadrature(order):
    """缓存标准 Hermite 节点，权重之和为 sqrt(pi)。 / Cache standard Hermite nodes with weights summing to sqrt(pi)."""
    return np.polynomial.hermite.hermgauss(order)

def _axis_constellation(qm):
    """按 TS 38.211 §5.1.3–5.1.6 构造单轴幅度及 NR 比特标签。 / Construct one-axis amplitudes and NR labels from TS 38.211."""
    count = qm // 2
    indices = np.arange(1 << count)
    bits = (indices[:, None] >> np.arange(count - 1, -1, -1)) & 1
    signs = 1 - 2 * bits
    inner = np.ones(len(indices), dtype=np.float64)
    for position in range(count - 1, 0, -1):
        inner = 2 ** (count - position) - signs[:, position] * inner
    amplitudes = signs[:, 0] * inner / math.sqrt(2 * ((1 << qm) - 1) / 3)
    return amplitudes, bits

def _conditional_moments(scores, amplitudes):
    """稳定计算子星座的 log 配分函数、一阶和二阶矩。 / Stably compute subset log partition and first/second moments."""
    maximum = np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores - maximum)
    partition = probabilities.sum(axis=-1, keepdims=True)
    probabilities /= partition
    return (maximum[..., 0] + np.log(partition[..., 0]),
            np.sum(probabilities * amplitudes, axis=-1),
            np.sum(probabilities * amplitudes ** 2, axis=-1))

def mode_weights(channel_norm, noise_variance, qm=6, order=96):
    """返回径向、相位、复正交模态权重及 oracle 逐比特 NLL。 / Return radial, phase, complex-orthogonal weights and oracle bit NLL.
    使用均匀 QAM、精确比特 log-MAP 和总体观测分布；order 控制积分精度。 / Assume uniform QAM, exact bit log-MAP and population observations; order controls quadrature accuracy."""
    radius = _scalar(channel_norm, "channel_norm", allow_zero=True)
    variance = _scalar(noise_variance, "noise_variance")
    if isinstance(qm, (bool, np.bool_)) or not isinstance(qm, Integral) or qm not in (2, 4, 6, 8):
        raise ValueError("qm must be one of 2, 4, 6, 8")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order < 2:
        raise ValueError("order must be an integer of at least 2")
    amplitudes, bits = _axis_constellation(int(qm))
    if radius == 0:
        differences = [amplitudes[bits[:, bit] == 0].mean() - amplitudes[bits[:, bit] == 1].mean()
                       for bit in range(qm // 2)]
        weight = float(np.mean(np.square(differences)) / (2 * variance))
        return dict(radial=weight, phase=weight, orthogonal=weight, oracle_nll=math.log(2))
    nodes, weights = _quadrature(int(order))
    # W=r*A+N，N~N(0,t/2)，Hermite 变量因此乘 sqrt(t)。 / W=r*A+N; Hermite nodes scale by sqrt(t).
    observed = radius * amplitudes[:, None] + math.sqrt(variance) * nodes[None]
    scores = -(observed[..., None] - radius * amplitudes) ** 2 / variance
    radial = coefficient = oracle_nll = 0.0
    for bit in range(qm // 2):
        zero, one = bits[:, bit] == 0, bits[:, bit] == 1
        log_zero, first_zero, second_zero = _conditional_moments(scores[..., zero], amplitudes[zero])
        log_one, first_one, second_one = _conditional_moments(scores[..., one], amplitudes[one])
        logits = log_zero - log_one
        alpha = 2 * (first_zero - first_one) / variance
        beta = -(second_zero - second_one) / variance
        small = np.exp(-np.abs(logits))
        bernoulli = small / (1 + small) ** 2

        def average(values):
            return float(np.mean(values @ weights) / math.sqrt(math.pi))

        coefficient += average(bernoulli * alpha ** 2)
        radial += average(bernoulli * (observed * alpha + 2 * radius * beta) ** 2)
        signs = 1 - 2 * bits[:, bit, None]
        oracle_nll += average(np.logaddexp(0, -signs * logits))
    coefficient /= qm // 2
    result = dict(radial=radial / (qm // 2),
                  phase=(radius ** 2 + variance) * coefficient / 2,
                  orthogonal=variance * coefficient / 2,
                  oracle_nll=oracle_nll / (qm // 2))
    if not all(math.isfinite(value) and value >= 0 for value in result.values()):
        raise FloatingPointError("population integration overflowed; use representable channel/noise scales")
    return result

@lru_cache(maxsize=4)
def constellation(qm=6):
    """仅复用NR映射，风险核独立。 / Reuse only the NR mapper, with an independent risk kernel."""
    if type(qm) is not int or qm not in (2, 4, 6, 8):
        raise ValueError("qm must be 2, 4, 6 or 8")
    words = ((np.arange(1 << qm)[:, None] >> np.arange(qm - 1, -1, -1)) & 1).astype(np.uint8)
    bits = ((np.arange(1 << (qm // 2))[:, None] >> np.arange(qm // 2 - 1, -1, -1)) & 1).astype(np.uint8)
    axis_words = np.zeros((len(bits), qm), dtype=np.uint8)
    axis_words[:, ::2] = bits
    return nr_qam_modulate(words), words, nr_qam_modulate(axis_words).real, bits

def bit_logits(channel, received, variance, qm=6):
    """方形QAM精确log(P0/P1)，不使用局部展开。 / Exact square-QAM log(P0/P1), without a local expansion."""
    h, y = np.broadcast_arrays(np.asarray(channel, dtype=np.complex128), np.asarray(received, dtype=np.complex128))
    if h.ndim < 1 or h.shape[-1] != 2 or not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError("need finite complex two-RX channels and observations")
    t = np.broadcast_to(np.asarray(variance, dtype=float), h.shape[:-1])
    if not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("complex noise variance must be positive")
    _, _, levels, labels = constellation(qm)
    z, power = np.sum(h.conj() * y, -1), np.sum(abs(h) ** 2, -1)
    result = np.empty((*h.shape[:-1], qm))
    for axis, projection in enumerate((z.real, z.imag)):
        score = (2 * projection[..., None] * levels - power[..., None] * levels ** 2) / t[..., None]
        for b in range(qm // 2):
            result[..., 2 * b + axis] = logsumexp(score[..., labels[:, b] == 0], -1) - logsumexp(score[..., labels[:, b] == 1], -1)
    return result

def fit_scale(teacher_logits, candidate_logits, tolerance=2e-12):
    """跨所有观测/正负误差/比特只拟合一个尺度。 / Fit one scale over all observations, error signs and bits."""
    teacher, logits = np.asarray(teacher_logits, float), np.asarray(candidate_logits, float)
    if teacher.ndim != 2 or logits.ndim != 3 or logits.shape != (len(teacher), 2, teacher.shape[1]):
        raise ValueError("expected teacher (N,qm), candidate (N,2,qm)")
    if not np.isfinite(teacher).all() or not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    before = teacher[:, None, :]
    p0, p1 = expit(before), expit(-before)
    logp0, logp1 = log_expit(before), log_expit(-before)
    def derivatives(scale):
        z = scale * logits
        q0, q1 = expit(z), expit(-z)
        return float(np.mean((q0 - p0) * logits)), float(np.mean(q0 * q1 * logits ** 2))
    def risk(scale):
        z = scale * logits
        # 两概率分别求值，避免1-p在高SNR舍入为零。 / Evaluate both probabilities without subtracting from one at high SNR.
        return float(np.mean(p0 * (logp0 - log_expit(z)) + p1 * (logp1 - log_expit(-z))))
    low, high, scale = 0., 2., 1.
    g0, _ = derivatives(0.)
    if g0 >= 0:
        scale, iterations = 0., 0
    else:
        while derivatives(high)[0] < 0 and high < 1024:
            high *= 2
        if derivatives(high)[0] < 0:
            raise ArithmeticError("no global-scale bracket")
        for iterations in range(1, 41):
            gradient, curvature = derivatives(scale)
            if abs(gradient) <= tolerance:
                break
            if gradient > 0:
                high = scale
            else:
                low = scale
            step = scale - gradient / curvature if curvature > 0 else float("nan")
            scale = step if low < step < high else (low + high) / 2
        else:
            raise ArithmeticError("global-scale fit did not converge")
    gradient, curvature = derivatives(scale)
    value = risk(scale)
    if value < -2e-13:
        raise ArithmeticError("negative direct posterior KL beyond roundoff")
    return dict(scale=scale, risk_nats_per_bit=value, uncalibrated_risk_nats_per_bit=risk(1.),
        gradient=gradient, curvature=curvature, iterations=iterations, boundary=scale == 0,
        converged=scale == 0 or abs(gradient) <= tolerance, gradient_tolerance=tolerance,
        scale_scope="one across all N x 2 signs x qm bits in this independent randomized quadrature")

def _parameters(x, kappa):
    spectrum = np.asarray(kappa, dtype=float)
    if spectrum.shape != (2,) or not np.isfinite(spectrum).all() or not spectrum[0] > spectrum[1] > 0:
        raise ValueError("need finite kappa1 > kappa2 > 0")
    if np.ndim(x) != 0 or not np.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("orientation x must be a finite scalar in [0,1]")
    k1, k2 = spectrum.tolist()
    a = k2 + float(x) * (k1 - k2)
    b = (k1 - k2) * math.sqrt(float(x) * (1 - float(x)))
    return k1, k2, a, b

def orientation_error(standard_complex, x, kappa=(1.09, .09)):
    """酉转动特征因子；同z跨x逐样本同范数。 / Rotate eigenfactors, preserving paired norms.

    输入最后一维为两个独立CN(0,1)。 / Last dimension holds two independent CN(0,1).
    """
    k1, k2, _, _ = _parameters(x, kappa)
    z = np.asarray(standard_complex, dtype=complex)
    if z.ndim < 1 or z.shape[-1] != 2 or not np.isfinite(z).all():
        raise ValueError("need finite (...,2) complex standard errors")
    strong, weak = math.sqrt(k1) * z[..., 0], math.sqrt(k2) * z[..., 1]
    return np.stack((math.sqrt(x) * strong - math.sqrt(1 - x) * weak,
                     math.sqrt(1 - x) * strong + math.sqrt(x) * weak), axis=-1)

def _energies(error):
    nu = np.asarray(error, dtype=complex)
    if nu.ndim < 1 or nu.shape[-1] != 2 or not np.isfinite(nu).all():
        raise ValueError("need finite (...,2) complex errors")
    with np.errstate(over="ignore"):
        energy = abs(nu) ** 2
    if not np.isfinite(energy).all():
        raise ValueError("error energies overflow")
    return energy[..., 0], energy[..., 1]

def phase_averaged_score(error, x, kappa=(1.09, .09)):
    """相位平均密度的∂a log f，端点连续。 / Phase-averaged density score with respect to a.

    独立真Y平均消去ν1相位后才能用E[KL*S]；不是固定Y下原CN密度的score。
    E[KL*S] applies after independent true-observation averaging, not as the original fixed-Y CN score.
    """
    k1, k2, a, b = _parameters(x, kappa)
    A, B = _energies(error)
    determinant = k1 * k2
    product = (A / determinant) * (B / determinant)
    if not np.isfinite(product).all():
        raise ValueError("score argument overflows")
    z = b * b * product
    factor = np.empty_like(z)
    small = z < 1e-8
    factor[small] = 1 - z[small] / 2 + z[small] ** 2 / 3
    root = np.sqrt(z[~small])
    factor[~small] = ive(1, 2 * root) / (root * ive(0, 2 * root))
    return (A - B) / determinant + (k1 + k2 - 2 * a) * product * factor

def sobol_standard_inputs(power, seed, qm=6):
    """十独立维度供X/W/z；每个嵌套前缀符号均衡。 / Ten Sobol dimensions for X, W and z."""
    if type(power) is not int or power < qm or type(seed) is not int or seed < 0:
        raise ValueError("need integer power>=qm and nonnegative integer seed")
    _, _, levels, _ = constellation(qm)
    sample = qmc.Sobol(10, scramble=True, seed=seed).random_base2(power)
    index = np.floor(sample[:, :2] * len(levels)).astype(int)
    count = np.bincount(index[:, 0] * len(levels) + index[:, 1], minlength=1 << qm)
    if np.any(count != len(sample) // (1 << qm)):
        raise ArithmeticError("Sobol symbol strata are unbalanced")
    normal = ndtri(np.clip(sample[:, 2:], 2. ** -53, 1 - 2. ** -53))
    noise = (normal[:, :4:2] + 1j * normal[:, 1:4:2]) / math.sqrt(2)
    standard = (normal[:, 4::2] + 1j * normal[:, 5::2]) / math.sqrt(2)
    return dict(symbols=levels[index[:, 0]] + 1j * levels[index[:, 1]], noise=noise, standard=standard)

def _sample_kl(teacher, candidate, scale):
    # 两概率独立计算，保留微小KL的舍入，不裁剪。 / Separate tails; retain KL roundoff without clipping.
    old, new = teacher[:, None, :], scale * candidate
    value = expit(old) * (log_expit(old) - log_expit(new))
    value += expit(-old) * (log_expit(-old) - log_expit(-new))
    return value.mean(axis=(1, 2))

def _moments(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "raw_sample_standard_error": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None}

def evaluate_nested(inputs, x, epsilon_over_r, gamma, powers, r=1., qm=6, kappa=(1.09, .09), observations=None):
    """直接积分及envelope密度导数；每前缀只拟合一个s。 / Direct risks and score derivatives.

    raw sample SE只是RQMC相关点诊断；独立scramble的SE才用于精度门槛。
    Raw sample SE is diagnostic for dependent RQMC points; precision uses independent scrambles.
    """
    _parameters(x, kappa)
    if not all(np.isfinite(v) for v in (r, gamma, epsilon_over_r)) or r <= 0 or gamma <= 0 or epsilon_over_r < 0:
        raise ValueError("need r>0, gamma>0 and epsilon_over_r>=0")
    symbols, noise, standard = [np.asarray(inputs[k]) for k in ("symbols", "noise", "standard")]
    n = len(symbols)
    if symbols.shape != (n,) or noise.shape != (n, 2) or standard.shape != (n, 2) or not all(np.isfinite(v).all() for v in (symbols, noise, standard)):
        raise ValueError("inconsistent or nonfinite quadrature inputs")
    if not powers or any(type(p) is not int or p < qm or 1 << p > n for p in powers):
        raise ValueError("invalid nested powers")
    n = 1 << max(powers)
    symbols, noise, standard = symbols[:n], noise[:n], standard[:n]
    h, variance = np.array([r, 0], dtype=complex), r * r / gamma
    if observations is None:
        y = symbols[:, None] * h + math.sqrt(variance) * noise
        teacher = bit_logits(h, y, variance, qm)
    else:
        y, teacher = observations[0][:n], observations[1][:n]
    error = orientation_error(standard, x, kappa)
    score = phase_averaged_score(error, x, kappa)
    epsilon = r * epsilon_over_r
    candidate = np.stack([bit_logits(h + sign * epsilon * error, y, variance, qm) for sign in (1, -1)], axis=1)
    rows = []
    for power in powers:
        count = 1 << power
        fit = fit_scale(teacher[:count], candidate[:count])
        sample_risk = _sample_kl(teacher[:count], candidate[:count], fit["scale"])
        if not np.isclose(sample_risk.mean(), fit["risk_nats_per_bit"], rtol=2e-13, atol=2e-16):
            raise ArithmeticError("sample and global KL reductions disagree")
        weighted = sample_risk * score[:count]
        derivative, score_diagnostic = _moments(weighted), _moments(score[:count])
        rows.append(dict(x=float(x), a=float(kappa[1] + x * (kappa[0] - kappa[1])),
            gamma=float(gamma), epsilon_over_r=float(epsilon_over_r), epsilon=epsilon,
            power=power, sobol_points=count, dependent_antithetic_evaluations=2 * count,
            fit=fit, derivative_wrt_a=derivative["mean"], derivative_sign=int(np.sign(derivative["mean"])),
            raw_sample_derivative_se=derivative["raw_sample_standard_error"],
            mean_score=score_diagnostic["mean"], raw_sample_score_se=score_diagnostic["raw_sample_standard_error"],
            raw_sample_risk_se=_moments(sample_risk)["raw_sample_standard_error"],
            score_scope="phase-averaged density after independent Y averaging; not a fixed-Y Gaussian score"))
    return rows

def summarize(replicates, reference, model, powers):
    """四scramble配对差及导数，完整保留符号。 / Paired scramble gains and signed derivatives."""
    rows, adjacent = [], []
    delta = model["kappa"][0] - model["kappa"][1]
    for power in powers:
        for gamma in model["gamma"]:
            for fraction in model["epsilon_over_r"]:
                epsilon = model["r"] * fraction
                slope = epsilon ** 2 * reference[gamma]["d"] / 4
                selected = {x: sorted((row for row in replicates if row["power"] == power and row["gamma"] == gamma and row["epsilon_over_r"] == fraction and row["x"] == x), key=lambda row: row["seed"]) for x in model["x"]}
                seeds = [row["seed"] for row in selected[model["x"][0]]]
                if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any([v["seed"] for v in values] != seeds for values in selected.values()):
                    raise ValueError("unpaired or missing scramble rows")
                first = np.array([v["fit"]["risk_nats_per_bit"] for v in selected[0.]])
                for x, values in selected.items():
                    risk = np.array([v["fit"]["risk_nats_per_bit"] for v in values])
                    derivative = np.array([v["derivative_wrt_a"] for v in values])
                    gap, D = _moments(risk - first), _moments(derivative)
                    rows.append(dict(power=power, gamma=gamma, epsilon_over_r=fraction, x=x,
                        a=model["kappa"][1] + x * delta, epsilon=epsilon,
                        scrambled_replicates=len(values), mean_risk=float(risk.mean()),
                        mean_scale=float(np.mean([v["fit"]["scale"] for v in values])),
                        derivative_wrt_a=D["mean"], derivative_numerical_se=D["raw_sample_standard_error"],
                        derivative_sign=int(np.sign(D["mean"])), quadratic_slope_wrt_a=slope,
                        gap_from_x0=gap["mean"], gap_numerical_se=gap["raw_sample_standard_error"],
                        normalized_gap=gap["mean"] / (slope * delta), normalized_gap_se=gap["raw_sample_standard_error"] / abs(slope * delta),
                        mean_score=float(np.mean([v["mean_score"] for v in values]))))
                for low, high in zip(model["x"][:-1], model["x"][1:]):
                    gain = _moments([b["fit"]["risk_nats_per_bit"] - a["fit"]["risk_nats_per_bit"] for a, b in zip(selected[low], selected[high])])
                    adjacent.append(dict(power=power, gamma=gamma, epsilon_over_r=fraction,
                        x_low=low, x_high=high, paired_mean_gain=gain["mean"],
                        numerical_se=gain["raw_sample_standard_error"], gain_sign=int(np.sign(gain["mean"])),
                        quadratic_gain=slope * delta * (high - low)))
    return rows, adjacent

