"""Problem scout: extract the public replay of MILO's third-autocorrelation record from its project page.

Usage (from the repo root):  .venv/bin/python selection/scout_milo_replay.py

MILO (arXiv:2609.38349) claims 1.4488860143 on EinsteinArena's third autocorrelation inequality, on 25,600 grid
points. The construction itself is not published. The project page (https://jprithwish.github.io/MILO/) embeds a
data file for its interactive demo, `static/einstein-run.js`, which holds a *plot-resolution* copy of the record:
3,200 of the 25,600 values (one in eight), rounded to 6 decimals, plus the same for the arena entry the run
started from (BasinHopper, 1.4515829).

This script downloads that file, parses it as JSON only (it is untrusted data and is never executed), and writes:

- selection/.cache/scout/third-autocorrelation-inequality/milo/einstein-run.js       (raw download)
- .../milo/milo_{record,start}_3200.json        the 3,200 values as published
- .../milo/milo_{record,start}_x8_25600.json    the same, block-repeated x8 back to 25,600 values

The decimated copy is NOT MILO's construction: scored as given it is far from the claimed value (see the printed
scores). It is only a starting point that sits near MILO's basin; `selection/scout_third_autocorr.py refine`
re-optimizes from it. Anything obtained that way derives from MILO's published data and must be credited to it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / ".cache" / "scout" / "third-autocorrelation-inequality" / "milo"
URL = "https://jprithwish.github.io/MILO/static/einstein-run.js"
META = ROOT / "scout_milo_replay.json"


def score(values: np.ndarray) -> float:
    """The EinsteinArena verifier's formula (dx = 0.5 / n)."""
    n = len(values)
    return float(2 * n * np.convolve(values, values).max() / values.sum() ** 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, headers={"User-Agent": "research-client/1.0"}, timeout=60)
    resp.raise_for_status()
    raw = resp.text
    (OUT_DIR / "einstein-run.js").write_text(raw)
    data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])  # `window.X = {...}`: keep the object only
    c3 = data["problems"]["c3"]
    meta = {
        "source_url": URL, "page_url": "https://jprithwish.github.io/MILO/",
        "accessed_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "claimed_record": c3["ours"], "claimed_record_date": c3["ours_when"], "record_n": c3["record_n"],
        "prior": c3["prior"], "prior_who": c3["prior_who"], "start_who": c3["start_who"],
        "run_started": c3["t0"], "running_min": c3["running_min"], "n_harness_evals": c3["n_evals"],
        "n_sub_record_constructions": c3["n_discoveries"],
        "label": "claim (read on the project page); the full construction is not published",
        "published_copy": {},
    }
    for name in ("start", "record"):
        v = np.asarray(c3[name], dtype=np.float64)
        factor = c3[f"{name}_n"] // len(v)
        (OUT_DIR / f"milo_{name}_{len(v)}.json").write_text(json.dumps({"values": v.tolist()}))
        up = OUT_DIR / f"milo_{name}_x{factor}_{len(v) * factor}.json"
        up.write_text(json.dumps({"values": np.repeat(v, factor).tolist()}))
        meta["published_copy"][name] = {"n_published": len(v), "n_original": c3[f"{name}_n"],
                                        "score_as_published": score(v), "upsampled_file": str(up.relative_to(ROOT.parent))}
        print(f"{name}: {len(v)} of {c3[f'{name}_n']} values published; scored as given: {score(v)!r}")
    META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"claimed record {c3['ours']!r} on {c3['ours_when']} (n = {c3['record_n']}); wrote {META}")


if __name__ == "__main__":
    main()
