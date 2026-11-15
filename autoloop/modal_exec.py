"""ModalExecutor: every seed and refine job runs on Modal CPUs through the DEPLOYED app autoloop/modal_app.py.

The executor spawns calls (no attached client), writes their ids to RUN/modal_calls/<tag>.json, then polls and
pulls. A job that has not returned by its deadline is read from its last checkpoint on the Volume. The exact checks
(certify) stay local: they are light (seconds to a minute) and must not depend on remote code.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from .safety import safe_err

APP_NAME, VOLUME = "revolutionaries-autoloop", "revolutionaries-autoloop"
USD_PER_JOB_S = 1.0 * 0.0000131 + 3072 / 1024 * 0.00000222
MAX_AT_ONCE = 100


class ModalExecutor:
    real = True
    usd_per_cpu_s = 0.0

    def __init__(self, workdir, prob, max_cells, usd_cap, run_name, poll_s=20.0):
        import modal
        self.modal = modal
        self.dir = Path(workdir); self.P = prob; self.max_cells = max_cells
        self.usd_cap, self.run_name, self.poll_s = usd_cap, run_name, poll_s
        self.spent_usd = 0.0
        (self.dir / "certs").mkdir(parents=True, exist_ok=True); (self.dir / "modal_calls").mkdir(exist_ok=True)
        self.fn = modal.Function.from_name(APP_NAME, "run_job")
        self.vol = modal.Volume.from_name(VOLUME)
        self._local = prob.LocalExecutor(workdir, max_cells=max_cells)   # certify only

    # ------------------------------------------------------------------ core: spawn, record, poll, pull
    def _run(self, jobs, budget, tag):
        n = len(jobs)
        est = n * (budget + 120) * USD_PER_JOB_S
        if self.spent_usd + est > self.usd_cap:
            return [dict(ok=False, error=f"Modal estimate USD {est:.2f} would pass the cap {self.usd_cap}")] * n, 0.0
        calls = []
        for job, start in jobs:
            job = dict(job, run=self.run_name, budget=float(budget))
            calls.append((job, self.fn.spawn(job, start)))
        self.spent_usd += est
        (self.dir / "modal_calls" / f"{tag}.json").write_text(json.dumps(
            [dict(key=j["key"], call_id=c.object_id, job=j) for j, c in calls], indent=1))
        waves = -(-n // MAX_AT_ONCE)
        deadline = time.time() + waves * (budget + 600) + 900
        out = {}
        while len(out) < n and time.time() < deadline:
            for job, c in calls:
                if job["key"] in out:
                    continue
                try:
                    out[job["key"]] = c.get(timeout=0)
                except (TimeoutError, self.modal.exception.TimeoutError):
                    pass
                except Exception as e:  # noqa: BLE001
                    out[job["key"]] = dict(ok=False, err_tail=f"call failed: {e!r}")
            if len(out) < n:
                time.sleep(self.poll_s)
        rows = []
        for job, c in calls:
            r = out.get(job["key"])
            local = self.dir / "certs" / f"{job['key']}.npy"
            if r and r.get("ok"):
                local.write_bytes(r["npy"])
                rows.append(dict(ok=True, path=str(local), summary=r["summary"], seconds=r.get("seconds"), job=job))
                continue
            try:                                   # no result in time: last checkpoint from the Volume
                data = b"".join(self.vol.read_file(f"{self.run_name}/{job['key']}.npy"))
                np.load(io.BytesIO(data))
                local.write_bytes(data)
                rows.append(dict(ok=True, path=str(local), summary=None, from_checkpoint=True, job=job))
            except Exception:  # noqa: BLE001
                rows.append(dict(ok=False, job=job, error=safe_err((r or {}).get("err_tail") or "no result before the deadline")))
        return rows, n * budget

    # ------------------------------------------------------------------ executor interface
    def seed(self, n, budget_s, cells, seed0):
        kind = self.P.MODAL_KINDS["seed"]
        jobs = [(dict(kind=kind, key=f"seed_s{seed0 + i}_n{cells}", cells=cells, seed=seed0 + i), None) for i in range(n)]
        rows, cpu = self._run(jobs, budget_s, f"seed_{seed0}")
        out = []
        for i, r in enumerate(rows):
            if not r["ok"]:
                out.append(dict(ok=False, error=r.get("error"))); continue
            v = np.load(r["path"])
            out.append(dict(ok=True, path=r["path"], reported=self.P.score(v),
                            params=dict(seed=seed0 + i, cells=cells, budget_s=budget_s, executor="modal",
                                        optimizer="WS2 tuned optimizer", from_checkpoint=r.get("from_checkpoint", False))))
        return out, cpu

    def refine(self, parent_path, split, p0s, budget_s, tag):
        return self.refine_many([parent_path], split, p0s, budget_s, tag)

    def refine_many(self, parent_paths, split, p0s, budget_s, tag):
        kind = self.P.MODAL_KINDS["refine"]
        jobs, meta = [], []
        for i, pp in enumerate(parent_paths):
            buf = io.BytesIO(); np.save(buf, self.P.load_vec(pp).astype(np.float64)); start = buf.getvalue()
            for p0 in p0s:
                jobs.append((dict(kind=kind, key=f"{tag}_{i:02d}_x{split}_p{p0:g}", split=split, p0=p0), start))
                meta.append((i, p0))
        rows, cpu = self._run(jobs, budget_s, tag)
        out = []
        for (i, p0), r in zip(meta, rows):
            if not r["ok"]:
                out.append(dict(ok=False, pi=i, error=r.get("error"), params=dict(split=split, p0=p0))); continue
            v = np.load(r["path"])
            s = r["summary"] or {}
            out.append(dict(ok=True, pi=i, path=r["path"], reported=self.P.score(v),
                            params=dict(split=split, p0=p0, budget_s=budget_s, executor="modal", nev=s.get("nev"),
                                        start_score=s.get("start_score"), from_checkpoint=r.get("from_checkpoint", False))))
        return out, cpu

    def certify(self, parent_path, tag):
        return self._local.certify(parent_path, tag)
