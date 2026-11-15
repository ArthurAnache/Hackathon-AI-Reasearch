"""Objective of second-autocorrelation-inequality: exact score, smoothed value and gradient, evaluation counter.

With c = f * f (full autoconvolution of the non-negative f, length 2n - 1):

    C(f) = (2 sum c_k^2 + sum c_k c_{k+1}) / (3 * sum c_k * max c_k)

This closed form equals the frozen verifier (problems/second-autocorrelation-inequality/verifier_pinned.py) to
about 1e-15. It is used to steer the search only: every score that is reported comes from the frozen verifier.

Smoothed objective (same as knockout/seeds/second_autocorrelation.py): f = w^2, and max c is replaced by

    m_beta = (1 / beta) * log sum_k lambda_k exp(beta * c_k)

`beta` is meant for a c scaled to max 1 (see `normalize`). Without multipliers (lambda_k = 1) this is the LogSumExp
upper bound of the seed. With multipliers (`log_lambda`, normalized so that sum lambda_k = 1) it is the smoothed
term of the exponential method of multipliers for min-max problems: the multipliers are updated between stages with
`update_multipliers`, and a moderate beta can then reach the exact maximum without the LogSumExp bias.

Cost unit: one `value_and_grad` = one evaluation (4 FFTs of about 2n points). `exact`, `autoconvolution` and
`update_multipliers` cost half an evaluation (2 FFTs).
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sf


class BudgetSpent(Exception):
    """Raised by the oracle when the next call would exceed the evaluation budget."""


def exact_score_from_conv(c: np.ndarray) -> float:
    num = (2.0 * np.dot(c, c) + np.dot(c[:-1], c[1:])) / 3.0
    return float(num / (c.sum() * c.max()))


class Oracle:
    """Evaluations of the objective for vectors of length n, counted in units of one value-and-gradient call.

    The oracle also remembers the best exact point it has evaluated (`best_f`, `best_score`): the exact score of
    every point comes for free with the smoothed value.
    """

    def __init__(self, n: int, budget: float | None = None):
        self.n = int(n)
        self.m = sf.next_fast_len(2 * self.n - 1, real=True)
        self.budget = budget
        self.calls = 0.0
        self.best_score = -np.inf
        self.best_f: np.ndarray | None = None

    # ------------------------------------------------------------------ bookkeeping
    def remaining(self) -> float:
        return np.inf if self.budget is None else self.budget - self.calls

    def _charge(self, cost: float) -> None:
        if self.budget is not None and self.calls + cost > self.budget + 1e-9:
            raise BudgetSpent
        self.calls += cost

    def _track(self, f: np.ndarray, score: float) -> None:
        if np.isfinite(score) and score > self.best_score:
            self.best_score = float(score)
            self.best_f = f.copy()

    # ------------------------------------------------------------------ evaluations
    def _conv(self, f: np.ndarray):
        spec = sf.rfft(f, self.m)
        return sf.irfft(spec * spec, self.m)[: 2 * self.n - 1], spec

    def autoconvolution(self, f: np.ndarray) -> np.ndarray:
        self._charge(0.5)
        return self._conv(f)[0]

    def exact(self, f: np.ndarray) -> float:
        """Exact score (closed form) of a non-negative f."""
        self._charge(0.5)
        score = exact_score_from_conv(self._conv(f)[0])
        self._track(f, score)
        return score

    def normalize(self, f: np.ndarray) -> np.ndarray:
        """f scaled so that max(f * f) = 1, the scale `beta` refers to. Costs half an evaluation."""
        self._charge(0.5)
        return f / np.sqrt(self._conv(f)[0].max())

    def grad_f(self, f: np.ndarray, beta: float, log_lambda: np.ndarray | None = None) -> np.ndarray:
        """Gradient of log C_beta with respect to f itself (ascent direction), also on the zero cells."""
        self._charge(1.0)
        return self._value_grad_f(f, beta, log_lambda)[1]

    def value_and_grad(self, w: np.ndarray, beta: float, log_lambda: np.ndarray | None = None):
        """(-log C_beta, its gradient with respect to w), for f = w^2. This is what the polisher minimizes."""
        self._charge(1.0)
        value, gf = self._value_grad_f(w * w, beta, log_lambda)
        return -value, -2.0 * w * gf

    def _value_grad_f(self, f, beta, log_lambda):
        c, spec = self._conv(f)
        cmax = c.max()
        a = beta * (c - cmax)
        if log_lambda is not None:
            a += log_lambda
        amax = a.max()
        e = np.exp(a - amax)
        z = e.sum()
        m_beta = cmax + (amax + np.log(z)) / beta
        num = (2.0 * np.dot(c, c) + np.dot(c[:-1], c[1:])) / 3.0
        s = f.sum()
        self._track(f, num / (s * s * cmax))           # sum c = (sum f)^2
        value = np.log(num) - 2.0 * np.log(s) - np.log(m_beta)
        g = 4.0 * c
        g[1:] += c[:-1]
        g[:-1] += c[1:]
        big = g / (3.0 * num) - (e / z) / m_beta        # d value / d c
        corr = sf.irfft(sf.rfft(big, self.m) * np.conj(spec), self.m)[: self.n]
        return value, 2.0 * corr - 2.0 / s

    def update_multipliers(self, f: np.ndarray, beta: float, log_lambda: np.ndarray | None = None) -> np.ndarray:
        """Multiplier step of the exponential method: lambda_k <- lambda_k exp(beta c_k), renormalized (in logs)."""
        self._charge(0.5)
        c = self._conv(f)[0]
        a = beta * (c - c.max())
        if log_lambda is not None:
            a += log_lambda
        amax = a.max()
        return a - (amax + np.log(np.exp(a - amax).sum()))
