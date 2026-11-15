"""Smoothed score of second-autocorrelation-inequality and its gradient, for a batch of chains.

Every function takes the array namespace `xp`: `numpy` (CPU reference, used by the unit tests) or `torch` (GPU,
float64). Both backends run the same lines, so the GPU objective is the tested one.

Exact score (the frozen verifier's formula in closed form, c = f * f the full autoconvolution of length 2n-1):
    C = (2 sum c_k^2 + sum c_k c_{k+1}) / (3 * sum c_k * max c_k)
Only two things differ from the verifier (CLAUDE.md, WS3 rule 3):
- non-negativity: f = w^2;
- the maximum: max u is replaced by LSE_beta(u) = max u + log(sum exp(beta (u - max u))) / beta >= max u.
To give `beta` one meaning along a chain, the objective works on g = kappa f / sum(f), u = g * g. `kappa` is fixed
per chain so that max u = 1 at the chain's start, hence beta is the sharpness relative to the maximum. The
objective is then exactly invariant under f -> a f (the plain LogSumExp is not: scaling f up would raise the
effective beta, a direction the optimizer could exploit).

    C_beta = num(u) / (kappa^2 LSE_beta(u)) <= C,    phi = -log C_beta   (minimized)

The FFT is zero-padded to m >= 2n-1 points, so the convolution is linear, not circular.
"""
from __future__ import annotations


def fft_length(n: int, kind: str = "smooth") -> int:
    """FFT size for the autoconvolution of n values: >= 2n-1, and a product of 2, 3, 5, 7 ("smooth", fast in
    cuFFT and pocketfft) or a power of two ("pow2")."""
    need = 2 * n - 1
    if kind == "pow2":
        return 1 << (need - 1).bit_length()
    if kind != "smooth":
        raise ValueError(f"unknown fft length kind {kind!r}")
    best = 1 << (need - 1).bit_length()
    p7 = 1
    while p7 < best:
        p5 = p7
        while p5 < best:
            p3 = p5
            while p3 < best:
                p2 = p3
                while p2 < need:
                    p2 *= 2
                best = min(best, p2)
                p3 *= 3
            p5 *= 5
        p7 *= 7
    return best


def autoconvolution(xp, g, m: int):
    """Full autoconvolution of each row of g (B, n): (B, 2n-1), and the spectrum of g."""
    n = g.shape[-1]
    spec = xp.fft.rfft(g, m)
    return xp.fft.irfft(spec * spec, m)[:, : 2 * n - 1], spec


def kappa_for(xp, f, m: int):
    """(B, 1): kappa such that g = kappa f / sum(f) has max(g * g) = 1."""
    s = xp.sum(f, -1)[:, None]
    u, _ = autoconvolution(xp, f / s, m)
    return 1.0 / xp.sqrt(xp.amax(u, -1))[:, None]


def exact_score(xp, f, m: int):
    """(B,): the exact score of each non-negative row of f (hard maximum)."""
    u, _ = autoconvolution(xp, f, m)
    num = (2.0 * xp.sum(u * u, -1) + xp.sum(u[:, :-1] * u[:, 1:], -1)) / 3.0
    return num / (xp.sum(u, -1) * xp.amax(u, -1))


def evaluate(xp, w, beta, kappa, m: int):
    """Objective and gradient at w (B, n), with f = w^2; beta and kappa are (B, 1).

    Returns (phi, grad_w, aux): phi (B,) = -log C_beta, grad_w (B, n) = d phi / d w, and
    aux = {"smooth": C_beta (B,), "exact": C with the hard maximum (B,), "grad_f": d log C_beta / d f (B, n),
    which is also defined on the cells where w = 0}. Cost: 4 FFTs of m points per row.
    """
    f = w * w
    n = f.shape[-1]
    s = xp.sum(f, -1)[:, None]
    scale = kappa / s
    g = f * scale
    spec = xp.fft.rfft(g, m)
    u = xp.fft.irfft(spec * spec, m)[:, : 2 * n - 1]
    umax = xp.amax(u, -1)[:, None]
    e = xp.exp(beta * (u - umax))
    z = xp.sum(e, -1)[:, None]
    lse = umax + xp.log(z) / beta
    num = (2.0 * xp.sum(u * u, -1) + xp.sum(u[:, :-1] * u[:, 1:], -1))[:, None] / 3.0
    k2 = kappa * kappa
    smooth = num / (k2 * lse)
    exact = num / (k2 * umax)
    # d log C_beta / d u
    du = 4.0 * u
    du[:, 1:] += u[:, :-1]
    du[:, :-1] += u[:, 1:]
    du = du / (3.0 * num) - e / (z * lse)
    # d / d g_j = 2 sum_k du_k g_{k-j}: a correlation, done in Fourier space
    corr = xp.fft.irfft(xp.fft.rfft(du, m) * xp.conj(spec), m)[:, :n]
    gv = 2.0 * corr
    # chain rule through g = kappa f / sum(f): removes the component along f (scale invariance)
    grad_f = scale * (gv - xp.sum(gv * g, -1)[:, None] / kappa)
    phi = -xp.log(smooth[:, 0])
    return phi, -2.0 * w * grad_f, {"smooth": smooth[:, 0], "exact": exact[:, 0], "grad_f": grad_f}
