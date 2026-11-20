"""Apply a frozen three-coefficient correction. / 应用冻结的三参数修正，不训练接收机。"""
import numpy as np


def apply_head(h0, h_p, covariance, noise, alpha, arm):
    """Inputs are pilot-derived; neither ground-truth channel nor payload labels enter."""
    h0 = np.asarray(h0, dtype=np.complex128)
    alpha = np.asarray(alpha, dtype=np.float64)
    if arm == "base":
        return h0.copy()
    if arm not in ("old_D", "old_F", "new_D", "new_F", "new_MSE", "rotation"):
        raise ValueError(f"Unknown arm: {arm}")
    length = np.linalg.norm(h0, axis=-1, keepdims=True)
    unit = np.divide(h0, length, out=np.zeros_like(h0), where=length > 0)
    residual = np.asarray(h_p, dtype=np.complex128) - h0
    if arm.startswith("new_") or arm == "rotation":
        # W=(I+C_e/t)^-1. Both the old and rotated vectors have the same norm.
        rotated = np.linalg.solve(np.eye(2) + covariance / np.asarray(noise)[..., None, None],
                                  h0[..., None])[..., 0]
        rotated_length = np.linalg.norm(rotated, axis=-1, keepdims=True)
        displacement = np.divide(rotated * length, rotated_length, out=h0.copy(),
                                 where=rotated_length > 0) - h0
        if arm == "rotation":
            return h0 + alpha[0] * displacement
        residual = residual + displacement
    inner = np.sum(unit.conj() * residual, axis=-1, keepdims=True)
    radial = np.where(length > 0, unit * inner.real, residual)
    phase = 1j * unit * inner.imag
    orthogonal = residual - radial - phase
    return h0 + alpha[0] * radial + alpha[1] * phase + alpha[2] * orthogonal
