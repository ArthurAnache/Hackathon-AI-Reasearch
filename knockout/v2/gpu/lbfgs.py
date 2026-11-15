"""L-BFGS for a batch of independent rows, on `numpy` or `torch` arrays (same code, see objective.py).

One call of the objective evaluates one point per row, so every evaluation is useful for every row: a row whose
step was accepted moves on to its next direction while another row is still backtracking. The loop is

    xt = opt.trial()                       # (B, n): one point per row
    phi, grad, aux = evaluate(xt)          # one batched evaluation
    took = opt.update(xt, phi, grad)       # (B,) bool: rows whose current point is now xt

Line search: Armijo backtracking from t = 1 (quadratic interpolation, clamped to [0.1 t, 0.5 t]).
History: the last `history` evaluations, shared slots for all rows; a slot is empty for the rows that did not
accept a step at that evaluation, or whose pair fails the curvature test s.y > 0.
A row that exhausts its backtracks forgets its history and restarts from steepest descent.
"""
from __future__ import annotations

import math


def _copy(a):
    return a.clone() if hasattr(a, "clone") else a.copy()


class BatchedLBFGS:
    def __init__(self, xp, x0, history: int = 8, c1: float = 1e-4, max_backtracks: int = 16):
        self.xp = xp
        self.m = int(history)
        self.c1 = float(c1)
        self.max_backtracks = int(max_backtracks)
        b = x0.shape[0]
        kw = {"dtype": x0.dtype, "device": x0.device}
        self.x = _copy(x0)
        self.f = xp.full((b,), math.inf, **kw)
        self.g = xp.zeros_like(x0)
        self.d = xp.zeros_like(x0)
        self.t = xp.ones((b,), **kw)
        self.gd = xp.zeros((b,), **kw)
        self.S = [None] * self.m
        self.Y = [None] * self.m
        self.rho = xp.zeros((self.m, b), **kw)
        self.gamma = xp.ones((b,), **kw)
        self.head = 0
        self.fresh = xp.ones((b,), dtype=xp.bool, device=x0.device)
        self.backtracks = xp.zeros((b,), dtype=xp.int64, device=x0.device)
        # bookkeeping, as plain Python lists
        self.accepted = [0] * b
        self.rejected = [0] * b
        self.restarts = [0] * b

    def mark_fresh(self, rows) -> None:
        """The objective of these rows changed (new beta): re-evaluate them at their current point."""
        for i in rows:
            self.fresh[i] = True

    def trial(self):
        xp = self.xp
        step = xp.where(self.fresh, xp.zeros_like(self.t), self.t)
        return self.x + step[:, None] * self.d

    def _direction(self, g):
        xp = self.xp
        order = [(self.head - 1 - j) % self.m for j in range(self.m)]  # newest first
        q = g
        alphas = []
        for k in order:
            if self.S[k] is None:
                alphas.append(None)
                continue
            a = self.rho[k] * xp.sum(self.S[k] * q, -1)
            q = q - a[:, None] * self.Y[k]
            alphas.append(a)
        r = self.gamma[:, None] * q
        for k, a in zip(reversed(order), reversed(alphas)):
            if a is None:
                continue
            bcoef = self.rho[k] * xp.sum(self.Y[k] * r, -1)
            r = r + (a - bcoef)[:, None] * self.S[k]
        return -r

    def _steepest(self, rows, keep_length: bool = False):
        """Rows `rows` (bool) forget their history and point down the gradient. With `keep_length` the first
        trial step is half as long as the last rejected one, so the search for a step length that works goes on
        downwards across restarts instead of starting again from 1 / |g| (at beta = 1e9 the smoothed maximum is
        only smooth over relative steps of about 1e-9)."""
        xp = self.xp
        gg = xp.sum(self.g * self.g, -1)
        norm = xp.sqrt(gg)
        safe = xp.where(norm > 0.0, norm, xp.ones_like(norm))
        if keep_length:
            t_new = 0.5 * self.t * xp.sqrt(xp.sum(self.d * self.d, -1)) / safe
        else:
            t_new = 1.0 / xp.where(norm > 1.0, norm, xp.ones_like(norm))
        self.rho = xp.where(rows[None, :], 0.0, self.rho)
        self.gamma = xp.where(rows, 1.0, self.gamma)
        self.d = xp.where(rows[:, None], -self.g, self.d)
        self.gd = xp.where(rows, -gg, self.gd)
        self.t = xp.where(rows, t_new, self.t)
        self.backtracks = xp.where(rows, 0, self.backtracks)

    def update(self, xt, ft, gt):
        xp = self.xp
        fresh = self.fresh
        finite = xp.isfinite(ft)
        armijo = finite & (ft <= self.f + self.c1 * self.t * self.gd)
        accept = armijo & ~fresh
        take = (fresh & finite) | accept
        reject = ~take

        if bool(accept.any()):
            am = accept[:, None]
            s = xp.where(am, xt - self.x, 0.0)
            y = xp.where(am, gt - self.g, 0.0)
            sy = xp.sum(s * y, -1)
            yy = xp.sum(y * y, -1)
            ss = xp.sum(s * s, -1)
            good = accept & (yy > 0.0) & (sy > 1e-10 * xp.sqrt(ss * yy))
            self.S[self.head] = s
            self.Y[self.head] = y
            self.rho[self.head] = xp.where(good, 1.0 / xp.where(good, sy, 1.0), 0.0)
            self.gamma = xp.where(good, sy / xp.where(good, yy, 1.0), self.gamma)
            self.head = (self.head + 1) % self.m

        if bool(take.any()):
            tm = take[:, None]
            self.x = xp.where(tm, xt, self.x)
            self.f = xp.where(take, ft, self.f)
            self.g = xp.where(tm, gt, self.g)
            d_new = self._direction(self.g)
            gd_new = xp.sum(self.g * d_new, -1)
            has_history = (self.rho > 0.0).any(0)
            descent = take & has_history & xp.isfinite(gd_new) & (gd_new < 0.0)
            self.d = xp.where(descent[:, None], d_new, self.d)
            self.gd = xp.where(descent, gd_new, self.gd)
            self.t = xp.where(descent, 1.0, self.t)
            self.backtracks = xp.where(take, 0, self.backtracks)
            plain = take & ~descent
            if bool(plain.any()):
                self._steepest(plain)

        if bool(reject.any()):
            denom = 2.0 * (ft - self.f - self.gd * self.t)
            ok = xp.isfinite(denom) & (denom > 0.0)
            tq = -self.gd * self.t * self.t / xp.where(ok, denom, 1.0)
            tq = xp.minimum(xp.maximum(tq, 0.1 * self.t), 0.5 * self.t)
            self.t = xp.where(reject, xp.where(ok, tq, 0.5 * self.t), self.t)
            self.backtracks = xp.where(reject, self.backtracks + 1, self.backtracks)
            over = reject & (self.backtracks > self.max_backtracks)
            if bool(over.any()):
                self._steepest(over, keep_length=True)
                for i, flag in enumerate(over.tolist()):
                    self.restarts[i] += int(flag)

        for i, (acc, rej) in enumerate(zip(accept.tolist(), reject.tolist())):
            self.accepted[i] += int(acc)
            self.rejected[i] += int(rej)
        self.fresh = xp.zeros_like(fresh)
        return take
