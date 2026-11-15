import modal, json
from pathlib import Path
fn = modal.Function.from_name("revolutionaries-autoloop-stage", "run_stage")
code = '''
import os, urllib.request
def refine(v, score, budget_s, rng):
    out = {}
    try:
        urllib.request.urlopen("https://einsteinarena.com", timeout=5); out["net"] = "OPEN"
    except Exception as e:
        out["net"] = "blocked:" + type(e).__name__
    out["env"] = sorted(os.environ)
    out["problems"] = os.path.exists("/root/problems")
    raise RuntimeError(repr(out))
'''
vsrc = Path("problems/circles-rectangle/verifier_pinned.py").read_text()
r = fn.remote(dict(job="probe", key="circles", shape=[21,3], direction="maximize", seed=0, budget=20.0), vsrc, code, None)
print(json.dumps(r["self_check"])); print(r.get("summary", {}).get("error") or r.get("err_tail"))
