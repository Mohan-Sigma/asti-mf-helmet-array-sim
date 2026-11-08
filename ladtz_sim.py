"""LADTZ priority-inversion Monte Carlo (kinematic, 2-D, ego frame).

Ego: steady-state cornering at speed v ~ U(40, 90) km/h and lean phi in
{0, 20, 40} deg; path curvature kappa = g*tan(phi)/v^2 (turn direction random).
Right-hand traffic, lane width 3.5 m; oncoming lane on the left of the ego lane.

Object types (sampled per object):
  lead      same lane ahead, slower, same direction
  oncoming  adjacent lane, opposite direction; 20 % drift into ego lane
  crossing  side-road vehicle crossing the ego path at a random arc point
  static    static obstacle near the lane (|d| <= 3 m)
  tangent   static object on the straight-ahead tangent line (off the road
            in a curve) -- the case a straight-line model over-prioritises

Ground truth: both trajectories propagated for 6 s (dt = 0.05 s); an object is
in conflict if separation < 2.0 m; criticality = time of first conflict.
Only scenes with >= 1 conflict object are scored.

Prioritisers (measurements with noise: bearing 1 deg, range 2 %, velocity
0.3 m/s, lean 0.8 deg, yaw-rate-derived curvature 3 %):
  baseline   1/TTC_radial (closing speed / range), lean-agnostic
  ladtz      baseline x W_LADTZ (alpha=1.5, sigma=38+0.55|phi| deg,
             mu = mean bearing of predicted path points over T_LA=2.5 s)
  pathttc    time to closest approach between the object (constant velocity)
             and the predicted curved ego path (reference, not in the paper)

Metric: priority inversion = top-ranked object != ground-truth most critical.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, json, sys

G = 9.81
DT, TH, DCONF = 0.05, 6.0, 2.0
LANE = 3.5
T_LA, ALPHA = 2.5, 1.5
TT = np.arange(0, TH + 1e-9, DT)


def path_xy(kappa, s, d=0.0):
    if abs(kappa) < 1e-6:
        return np.stack([s, np.full_like(s, d)], -1)
    x = np.sin(kappa * s) / kappa
    y = (1 - np.cos(kappa * s)) / kappa
    nx, ny = -np.sin(kappa * s), np.cos(kappa * s)
    return np.stack([x + d * nx, y + d * ny], -1)


def heading(kappa, s):
    return kappa * s


def sample_object(rng, kappa, v, kind):
    t = TT
    if kind == "lead":
        s0 = rng.uniform(10, 60); vo = v * rng.uniform(0.3, 0.95)
        traj = path_xy(kappa, s0 + vo * t)
    elif kind == "oncoming":
        s0 = rng.uniform(30, 120); vo = rng.uniform(40, 90) / 3.6
        d = np.full_like(t, LANE)
        if rng.random() < 0.2:
            d = np.maximum(LANE - rng.uniform(0.6, 1.5) * t, rng.uniform(-0.5, 1.0))
        s = s0 - vo * t
        traj = np.stack([path_xy(kappa, np.array([si]), di)[0] for si, di in zip(s, d)])
    elif kind == "crossing":
        sx = rng.uniform(15, 80); side = rng.choice([-1, 1])
        p = path_xy(kappa, np.array([sx]))[0]
        h = heading(kappa, sx)
        nrm = np.array([-np.sin(h), np.cos(h)]) * side          # direction to start point
        vo = rng.uniform(20, 50) / 3.6
        # time-of-arrival jitter so that only some cross in front of the ego
        tarr = sx / v * rng.uniform(0.6, 1.4)
        start = p + nrm * vo * tarr
        traj = start[None] - nrm[None] * vo * t[:, None]
    elif kind == "static":
        s0 = rng.uniform(10, 80); d = rng.uniform(-3, 3)
        traj = np.repeat(path_xy(kappa, np.array([s0]), d), len(t), 0)
    elif kind == "tangent":
        r = rng.uniform(20, 80); lat = rng.uniform(-1, 1)
        traj = np.repeat(np.array([[r, lat]]), len(t), 0)
    return traj


KINDS = ["lead", "oncoming", "crossing", "static", "tangent"]
KIND_P = [0.2, 0.25, 0.2, 0.15, 0.2]


def ladtz_weight(theta, kappa_m, v, phi_m):
    s = v * np.arange(DT, T_LA + 1e-9, DT)
    pts = path_xy(kappa_m, s)
    mu = np.mean(np.arctan2(pts[:, 1], pts[:, 0]))
    sig = np.deg2rad(38 + 0.55 * abs(np.rad2deg(phi_m)))
    dlt = np.angle(np.exp(1j * (theta - mu)))
    return 1 + ALPHA * np.exp(-dlt ** 2 / (2 * sig ** 2))


def scene(rng, phi_deg, nobj):
    while True:
        v = rng.uniform(40, 90) / 3.6
        phi = np.deg2rad(phi_deg) * rng.choice([-1, 1])
        kappa = G * np.tan(phi) / v ** 2
        ego = path_xy(kappa, v * TT)
        objs = [sample_object(rng, kappa, v, rng.choice(KINDS, p=KIND_P)) for _ in range(nobj)]
        tc = []
        for o in objs:
            dist = np.linalg.norm(o - ego, axis=1)
            idx = np.nonzero(dist < DCONF)[0]
            tc.append(TT[idx[0]] if len(idx) else np.inf)
        tc = np.array(tc)
        if np.isfinite(tc).any() and np.all(tc[np.isfinite(tc)] > 0):
            return v, phi, kappa, objs, tc


def prioritise(rng, v, phi, kappa, objs):
    phi_m = phi + np.deg2rad(rng.normal(0, 0.8))
    kappa_m = kappa * (1 + rng.normal(0, 0.03))
    base, lad, pth = [], [], []
    ego_vel = np.array([v, 0.0])
    s_pred = v * TT
    ego_pred = path_xy(kappa_m, s_pred)
    for o in objs:
        p = o[0]; vel = (o[1] - o[0]) / DT
        r = np.linalg.norm(p) * (1 + rng.normal(0, 0.02))
        th = np.arctan2(p[1], p[0]) + np.deg2rad(rng.normal(0, 1.0))
        pm = r * np.array([np.cos(th), np.sin(th)])
        velm = vel + rng.normal(0, 0.3, 2)
        rel = velm - ego_vel
        rdot = (pm @ rel) / r
        u = max(-rdot, 0) / r                                 # 1/TTC_radial
        base.append(u)
        lad.append(u * ladtz_weight(th, kappa_m, v, phi_m))
        traj = pm[None] + velm[None] * TT[:, None]
        dist = np.linalg.norm(traj - ego_pred, axis=1)
        idx = np.nonzero(dist < DCONF)[0]
        pth.append(1.0 / TT[idx[0]] if len(idx) and TT[idx[0]] > 0 else (1e3 if len(idx) else 0.0))
    return np.array(base), np.array(lad), np.array(pth)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(7)
    out = {}
    for phi in (0, 20, 40):
        for nobj in (3, 5, 8):
            inv = {"baseline": 0, "ladtz": 0, "pathttc": 0}
            for _ in range(ntr):
                v, ph, k, objs, tc = scene(rng, phi, nobj)
                gt = int(np.argmin(tc))
                for name, sc in zip(inv, prioritise(rng, v, ph, k, objs)):
                    inv[name] += int(np.argmax(sc) != gt)
            out[f"{phi}_{nobj}"] = {m: wilson(c, ntr) for m, c in inv.items()}
            print(phi, nobj, {m: f"{100*w[0]:.1f}% [{100*w[1]:.1f},{100*w[2]:.1f}]" for m, w in out[f'{phi}_{nobj}'].items()}, flush=True)
    json.dump({"ntrial_per_cell": ntr, "results": out}, open(os.path.join(HERE, "ladtz_results.json"), "w"))
