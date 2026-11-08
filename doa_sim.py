"""ASTI-MF helmet-array DoA Monte Carlo.

Array: 10 MEMS microphones on the surface of a 22 cm-diameter rigid sphere
(helmet-shell proxy): 8 equatorial at 45 deg spacing (8.4 cm chord), 1 crown,
1 chin (front, -40 deg elevation).

Propagation: far-field plane waves scattered by a rigid sphere (spherical-
harmonic series, Rafaely 2015). Optional specular ground reflection (rider ear
height 1.2 m, |rho| = 0.9) and, in the "urban" condition, one building
reflection (random azimuth, rho 0.3-0.7, extra path delay 2-20 ms).

Source: synthetic two-tone wail siren, fundamental swept 650-1500 Hz plus
2nd/3rd harmonics. Noise: 70 % spatially uncorrelated 1/f wind noise + 30 %
spherically diffuse 1/f traffic noise (by band power). SNR is per microphone
over the 500 Hz-4 kHz analysis band.

Snapshot: 25 ms (1200 samples @ 48 kHz), STFT 256/128 -> 8 frames.

Estimators (same snapshot, same 2 deg x 10 deg grid):
  DAS, MVDR (10 % diagonal loading), SRP-PHAT -- all with rigid-sphere steering
  SRP-PHAT-FF -- SRP-PHAT with free-field steering (model mismatch baseline)
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, json, sys
from numpy.fft import rfft, irfft, rfftfreq
from scipy.special import spherical_jn, spherical_yn, eval_legendre

FS, C, R = 48000, 343.0, 0.11
NFFT, HOP = 256, 128
DUR = 0.025
BAND = (500.0, 4000.0)
H_EAR = 1.2
NMAX = 25


def array_geometry():
    pos = [[R * np.cos(np.deg2rad(45 * k)), R * np.sin(np.deg2rad(45 * k)), 0.0] for k in range(8)]
    pos.append([0, 0, R])
    el = np.deg2rad(-40)
    pos.append([R * np.cos(el), 0, R * np.sin(el)])
    return np.array(pos)


def unit(az, el):
    az, el = np.asarray(az), np.asarray(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def _bn(ka):
    """Rigid-sphere radial terms b_n(ka) for n=0..NMAX, shape (F, N)."""
    n = np.arange(NMAX + 1)[None, :]
    ka = np.maximum(ka, 1e-6)[:, None]
    j, jd = spherical_jn(n, ka), spherical_jn(n, ka, True)
    h = j - 1j * spherical_yn(n, ka)                     # h^(2): outgoing for e^{+iwt}
    hd = jd - 1j * spherical_yn(n, ka, True)
    return (2 * n + 1) * (1j) ** n * (j - jd / hd * h)


def sphere_tf(freqs, mics, U):
    """Pressure at mic positions for unit plane waves from directions U.
    freqs (F,), mics (M,3), U (G,3) -> (F,G,M)."""
    b = _bn(2 * np.pi * freqs / C * R)                    # (F,N)
    cos = np.clip((U @ (mics / np.linalg.norm(mics, axis=1, keepdims=True)).T), -1, 1)  # (G,M)
    P = np.stack([eval_legendre(n, cos) for n in range(NMAX + 1)], -1)                # (G,M,N)
    return np.einsum("fn,gmn->fgm", b, P)


def freefield_tf(freqs, mics, U):
    return np.exp(1j * 2 * np.pi * freqs[:, None, None] / C * (U @ mics.T)[None])


def apply_tf(sig, tf_fn, mics, u, extra_delay=0.0, gain=1.0):
    n = len(sig)
    S = rfft(sig, 2 * n)
    f = rfftfreq(2 * n, 1 / FS)
    H = tf_fn(f, mics, u[None])[:, 0, :]                  # (F,M)
    X = gain * S[:, None] * H * np.exp(-2j * np.pi * f * extra_delay)[:, None]
    return irfft(X, 2 * n, axis=0)[:n].T                  # (M,n)


def siren(n, rng):
    t = np.arange(n) / FS
    f0 = 1075 + 425 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t + rng.uniform(0, 2 * np.pi))
    ph = 2 * np.pi * np.cumsum(f0) / FS
    return np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)


def colored(nch, n, rng):
    W = rfft(rng.standard_normal((nch, n)), axis=1)
    f = rfftfreq(n, 1 / FS); f[0] = f[1]
    return irfft(W / np.sqrt(f)[None, :], n, axis=1)


_POOL = {}


def diffuse(mics, n, rng, ndir=48, npool=512):
    """Spherically diffuse noise: ndir random plane waves drawn from a fixed
    pool of npool quasi-uniform directions (TFs cached on the nominal array)."""
    if n not in _POOL:
        k = np.arange(npool) + 0.5
        z = 1 - 2 * k / npool; a = np.pi * (1 + 5 ** 0.5) * k
        U = np.stack([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z], 1)
        f = rfftfreq(2 * n, 1 / FS)
        _POOL[n] = sphere_tf(f, array_geometry(), U)       # (F,G,M)
    H = _POOL[n]
    idx = rng.choice(H.shape[1], ndir, replace=False)
    S = rfft(colored(ndir, n, rng), 2 * n, axis=1)        # (ndir,F)
    X = np.einsum("df,fdm->fm", S, H[:, idx, :])
    return irfft(X, 2 * n, axis=0)[:n].T


def band_power(x):
    X = rfft(x, axis=-1); f = rfftfreq(x.shape[-1], 1 / FS)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return np.mean(np.abs(X[..., m]) ** 2)


def stft(x):
    win = np.hanning(NFFT)
    nfr = 1 + (x.shape[1] - NFFT) // HOP
    fr = np.stack([x[:, i * HOP:i * HOP + NFFT] * win for i in range(nfr)], 1)
    return rfft(fr, axis=-1)


AZ = np.deg2rad(np.arange(0, 360, 2.0))
EL = np.deg2rad(np.array([-10, 0, 10, 20, 30]))
AA, EE = np.meshgrid(AZ, EL, indexing="ij")
GRID_U = unit(AA.ravel(), EE.ravel())
GRID_AZ = AA.ravel()


def spectra(X, A, Aphat):
    M, T, F = X.shape
    Xf = np.transpose(X, (2, 0, 1))
    Rm = Xf @ np.conj(np.transpose(Xf, (0, 2, 1))) / T
    Xp = Xf / (np.abs(Xf) + 1e-12)
    Rp = Xp @ np.conj(np.transpose(Xp, (0, 2, 1))) / T
    out = {}
    out["DAS"] = np.real(np.einsum("fgm,fmn,fgn->g", np.conj(A), Rm, A))
    load = 0.1 * np.real(np.trace(Rm, axis1=1, axis2=2))[:, None, None] / M
    Ri = np.linalg.inv(Rm + load * np.eye(M)[None])
    out["MVDR"] = np.sum(1.0 / np.maximum(np.real(np.einsum("fgm,fmn,fgn->fg", np.conj(A), Ri, A)), 1e-12), 0)
    out["SRP-PHAT"] = np.real(np.einsum("fgm,fmn,fgn->g", np.conj(Aphat["sphere"]), Rp, Aphat["sphere"]))
    out["SRP-PHAT-FF"] = np.real(np.einsum("fgm,fmn,fgn->g", np.conj(Aphat["ff"]), Rp, Aphat["ff"]))
    return out


METHODS = ("DAS", "MVDR", "SRP-PHAT", "SRP-PHAT-FF")


def run(snrs, ntrial, condition, seed):
    rng = np.random.default_rng(seed)
    mics0 = array_geometry()
    n = int(DUR * FS)
    freqs = rfftfreq(NFFT, 1 / FS)
    bm = (freqs >= BAND[0]) & (freqs <= BAND[1])
    A = sphere_tf(freqs[bm], mics0, GRID_U)
    A = A / np.linalg.norm(A, axis=2, keepdims=True)
    Aph = {"sphere": np.exp(1j * np.angle(A)),
           "ff": freefield_tf(freqs[bm], mics0, GRID_U)}
    res = {s: {m: [] for m in METHODS} for s in snrs}
    for snr in snrs:
        for _ in range(ntrial):
            az = rng.uniform(0, 2 * np.pi); el = np.deg2rad(rng.uniform(-10, 20))
            mics = mics0 + rng.normal(0, 0.002, mics0.shape)
            src = siren(n + 2000, rng)
            s = apply_tf(src[2000:], sphere_tf, mics, unit(az, el))
            if condition in ("ground", "urban"):
                dg = 2 * H_EAR * max(np.sin(el), 0.02) / C
                s += apply_tf(src[2000:], sphere_tf, mics, unit(az, -abs(el) - 0.02), dg, 0.9)
            if condition == "urban":
                dw = rng.uniform(0.002, 0.020)
                k = int(round(dw * FS))
                s += apply_tf(src[2000 - k:2000 - k + n], sphere_tf, mics,
                              unit(rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(-5, 15))), 0.0,
                              rng.uniform(0.3, 0.7))
            wind = colored(len(mics), n, rng)
            dif = diffuse(mics, n, rng)
            wind *= np.sqrt(0.7 / band_power(wind)); dif *= np.sqrt(0.3 / band_power(dif))
            noise = wind + dif
            s *= np.sqrt(10 ** (snr / 10) * band_power(noise) / band_power(s))
            X = stft(s + noise)[:, :, bm]
            for m, P in spectra(X, A, Aph).items():
                e = np.rad2deg(np.angle(np.exp(1j * (GRID_AZ[np.argmax(P)] - az))))
                res[snr][m].append(abs(e))
    return res


def summarize(e):
    e = np.array(e)
    boot = np.random.default_rng(1).choice(e, (2000, len(e))).mean(1)
    return dict(mean=float(e.mean()), ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                median=float(np.median(e)), p90=float(np.percentile(e, 90)), within10=float(np.mean(e <= 10)), n=len(e))


if __name__ == "__main__":
    ntrial = int(sys.argv[1]); condition = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    snrs = [-15, -10, -5, 0, 5, 10, 20]
    res = run(snrs, ntrial, condition, seed)
    summ = {s: {m: summarize(v) for m, v in d.items()} for s, d in res.items()}
    for s in snrs:
        print(f"SNR {s:>4}: " + "  ".join(f"{m} {summ[s][m]['mean']:6.2f}/{summ[s][m]['median']:5.2f}" for m in METHODS))
    json.dump({"condition": condition, "ntrial": ntrial, "seed": seed, "summary": summ,
               "raw": {str(s): d for s, d in res.items()}},
              open(os.path.join(HERE, f"doa_{condition}_s{seed}.json"), "w"))
