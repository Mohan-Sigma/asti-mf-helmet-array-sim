# ASTI-MF simulation code (IEEE Sensors Journal submission)

Reproduces every number, table and figure in
"Helmet-Mounted Acoustic Array and Lean-Path Threat Prioritization for Motorcycle Riders"
and Supplementary Material S1.

## Files
| File | Purpose |
|---|---|
| `doa_sim.py` | Helmet-array DoA Monte Carlo (rigid-sphere scattering, wind + diffuse noise, reflections; DAS / MVDR / SRP-PHAT) |
| `ladtz_sim.py` | Scene generator and ground truth for threat prioritization (also the original bearing-weighted model) |
| `lpct_sim.py` | Lean-path conflict-time (LP-CT) vs baselines, plus lean-noise and lean-bias sweeps |
| `ladtz_ablate.py`, `ablate_run.py` | Bearing-weight width ablation (Table S6) |
| `make_figures.py` | Figures 1-4 and the graphical abstract from the result files |
| `*.json` | Result files used in the paper |

## Reproduce
```
pip install -r requirements.txt
python doa_sim.py 300 ground 1     # ~5 min -> doa_ground_s1.json
python doa_sim.py 300 urban 2      # ~5 min -> doa_urban_s2.json
python lpct_sim.py 1000            # ~4 min -> lpct_results.json
python ablate_run.py               # ~2 min -> ablate_results.json
python make_figures.py             # -> fig/
```
Seeds are fixed in the scripts (see Supplement S1, Section S7). Results were produced with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11.

## License
MIT (see LICENSE).

## Scope
Simulation only. No hardware measurements. See Supplement S1, Section S8, for limitations.
