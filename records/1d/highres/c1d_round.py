"""Round a float64 .npy vector to integers (max |v| = 2^56 by default), write {"values": [...]} JSON, check exactly.

    c1d_round.py SOLUTION.npy OUT.json [--bits 56]
"""
import json, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import c1d_exact as e

v = np.load(sys.argv[1]).astype(np.float64)
bits = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 56
ints = [int(x) for x in np.rint(v / np.abs(v).max() * 2.0 ** bits)]
json.dump({"values": ints}, open(sys.argv[2], "w"), separators=(",", ":"))
r = e.check(ints)
print(json.dumps(dict(out=sys.argv[2], n=r["n"], exact=float(r["score"]), beats_record=bool(r["beats"]),
                      gap_to_record=float(r["score"] - e.RECORD))))
