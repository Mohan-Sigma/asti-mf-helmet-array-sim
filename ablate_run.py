"""Bearing-weight width ablation (Supplement Table S6): N=1000 scenes per lean, five objects."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ladtz_ablate.py")).read().split("for phi in (0,20,40)")[0]
exec(src)
out = {phi: run(phi, 5, 1000, seed=500 + phi) for phi in (0, 20, 40)}
print(out)
json.dump({"ntrial": 1000, "nobj": 5, "inversion_pct": out}, open(os.path.join(HERE, "ablate_results.json"), "w"), indent=1)
