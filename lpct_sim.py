"""Lean-path conflict-time (LP-CT) prioritisation vs baselines.

Uses the scene generator in ladtz_sim.py. Prioritisers:
  RTTC    1/TTC_radial (closing speed / range), lean-agnostic baseline
  BW      bearing-weighted RTTC (original LADTZ formulation: Gaussian weight
          alpha=1.5, sigma=38+0.55|phi| deg around the lean-path centroid)
  LPCT-L  conflict time against the predicted path, curvature from lean:
          kappa = g*tan(phi_m)/v^2   (lean-only, no yaw-rate sensor)
  LPCT-Y  as LPCT-L, curvature from yaw rate: kappa = omega_z/v

Experiments
  main      lean {0,20,40} x N_obj {3,5,8}, nominal noise (lean 0.8 deg)
  noise     lean 40, N_obj 5, lean-noise sigma {0.5,1,2,3,5} deg
  bias      lean 40, N_obj 5, lean bias {-5,-3,0,+3,+5} deg (e.g. rider
            hang-off / tyre-width effect not captured by g*tan(phi))
Output: lpct_results.json (counts, Wilson 95 % CIs).
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, json, sys
sys.path.insert(0, HERE)
import ladtz_sim as L

NAMES = ["RTTC", "BW", "LPCT-L", "LPCT-Y"]


def conflict_score(pm, velm, kappa, v):
    ego = L.path_xy(kappa, v * L.TT)
    traj = pm[None] + velm[None] * L.TT[:, None]
    idx = np.nonzero(np.linalg.norm(traj - ego, axis=1) < L.DCONF)[0]
    if not len(idx):
        return 0.0
    return 1.0 / max(L.TT[idx[0]], L.DT)


def trial(rng, phi_deg, nobj, lean_sd=0.8, lean_bias=0.0):
    v, phi, kappa, objs, tc = L.scene(rng, phi_deg, nobj)
    gt = int(np.argmin(tc))
    sgn = np.sign(phi) if phi != 0 else 1.0
    phi_m = phi + sgn * np.deg2rad(lean_bias) + np.deg2rad(rng.normal(0, lean_sd))
    k_lean = L.G * np.tan(phi_m) / v ** 2
    k_yaw = kappa * (1 + rng.normal(0, 0.03))
    sc = {n: [] for n in NAMES}
    for o in objs:
        p = o[0]; vel = (o[1] - o[0]) / L.DT
        r = np.linalg.norm(p) * (1 + rng.normal(0, 0.02))
        th = np.arctan2(p[1], p[0]) + np.deg2rad(rng.normal(0, 1.0))
        pm = r * np.array([np.cos(th), np.sin(th)])
        velm = vel + rng.normal(0, 0.3, 2)
        u = max(-(pm @ (velm - np.array([v, 0.0]))) / r, 0) / r
        sc["RTTC"].append(u)
        sc["BW"].append(u * L.ladtz_weight(th, k_yaw, v, phi_m))
        sc["LPCT-L"].append(conflict_score(pm, velm, k_lean, v))
        sc["LPCT-Y"].append(conflict_score(pm, velm, k_yaw, v))
    return {n: int(np.argmax(s) != gt) for n, s in sc.items()}


def cell(seed, ntr, **kw):
    rng = np.random.default_rng(seed)
    cnt = dict.fromkeys(NAMES, 0)
    for _ in range(ntr):
        for n, e in trial(rng, **kw).items():
            cnt[n] += e
    return {n: dict(k=c, n=ntr, p=L.wilson(c, ntr)[0], lo=L.wilson(c, ntr)[1], hi=L.wilson(c, ntr)[2]) for n, c in cnt.items()}


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = {"ntrial_per_cell": ntr, "main": {}, "noise": {}, "bias": {}}
    seed = 100
    for phi in (0, 20, 40):
        for nobj in (3, 5, 8):
            seed += 1
            out["main"][f"{phi}_{nobj}"] = r = cell(seed, ntr, phi_deg=phi, nobj=nobj)
            print("main", phi, nobj, {n: round(100 * v["p"], 1) for n, v in r.items()}, flush=True)
    for sd in (0.5, 1, 2, 3, 5):
        seed += 1
        out["noise"][str(sd)] = r = cell(seed, ntr, phi_deg=40, nobj=5, lean_sd=sd)
        print("noise", sd, {n: round(100 * v["p"], 1) for n, v in r.items()}, flush=True)
    for b in (-5, -3, 0, 3, 5):
        seed += 1
        out["bias"][str(b)] = r = cell(seed, ntr, phi_deg=40, nobj=5, lean_bias=b)
        print("bias", b, {n: round(100 * v["p"], 1) for n, v in r.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "lpct_results.json"), "w"), indent=1)
