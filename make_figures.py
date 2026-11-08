"""Figures for the IEEE Sensors Journal manuscript, generated only from the
result files written by doa_sim.py and lpct_sim.py."""
import json, os, numpy as np, matplotlib
HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import sys
sys.path.insert(0, HERE)
from doa_sim import array_geometry

OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1})
W1 = 3.45  # IEEE single-column width (in)
COL = {"DAS": "#1f4e79", "MVDR": "#c55a11", "SRP-PHAT": "#548235", "SRP-PHAT-FF": "#7f7f7f",
       "RTTC": "#7f7f7f", "BW": "#c55a11", "LPCT-L": "#1f4e79", "LPCT-Y": "#548235"}
LAB = {"DAS": "DAS (sphere)", "MVDR": "MVDR (sphere)", "SRP-PHAT": "SRP-PHAT (sphere)",
       "SRP-PHAT-FF": "SRP-PHAT (free field)", "RTTC": "Radial TTC (baseline)",
       "BW": "Bearing-weighted (LADTZ v1)", "LPCT-L": "LP-CT, lean-derived $\\kappa$",
       "LPCT-Y": "LP-CT, yaw-rate $\\kappa$"}


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec="#333333", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.0, linespacing=1.1)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(W1, 2.35)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(2.25, 6.35, "Helmet node", ha="center", fontsize=7.5, weight="bold")
    ax.text(7.75, 6.35, "Chassis node", ha="center", fontsize=7.5, weight="bold")
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 4.3, 6.0, boxstyle="round,pad=0.02", fc="none", ec="#1f4e79", lw=0.8, ls="--"))
    ax.add_patch(FancyBboxPatch((5.6, 0.15), 4.3, 6.0, boxstyle="round,pad=0.02", fc="none", ec="#548235", lw=0.8, ls="--"))
    hb = "#deebf7"; cb = "#e2f0d9"
    box(ax, 0.35, 4.75, 3.8, 1.1, "10-mic MEMS array\non helmet shell", hb)
    box(ax, 0.35, 3.25, 3.8, 1.15, "MCU DSP: STFT +\nDoA spectrum (rigid-\nsphere steering)", hb)
    box(ax, 0.35, 1.9, 3.8, 1.0, "Helmet IMU (head pose)", hb)
    box(ax, 0.35, 0.4, 3.8, 1.15, "Spatial audio +\ndirectional haptics", hb)
    box(ax, 5.85, 4.75, 3.8, 1.1, "Radar, LiDAR,\ncameras (tracks)", cb)
    box(ax, 5.85, 3.25, 3.8, 1.15, "Lean, yaw rate,\nspeed (IMU/ECU)", cb)
    box(ax, 5.85, 1.6, 3.8, 1.3, "Edge computer: track\nfusion + LP-CT\nprioritization", cb)
    box(ax, 5.85, 0.4, 3.8, 0.85, "Alert manager", cb)
    ax.annotate("", xy=(5.6, 3.9), xytext=(4.4, 3.9), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(4.4, 0.95), xytext=(5.6, 0.95), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(5.0, 4.25, "DoA,\nIMU", ha="center", fontsize=5.0)
    ax.text(5.0, 1.2, "alert", ha="center", fontsize=5.0)
    ax.text(5.0, 2.55, "Wi-Fi\nlink", ha="center", fontsize=5.0, style="italic")
    fig.savefig(f"{OUT}/fig1_architecture.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_array():
    m = array_geometry() * 100
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.0))
    for ax, (i, j, ti, xl, yl) in zip(axs, [(0, 1, "(a) top view", "x (cm)", "y (cm)"), (0, 2, "(b) side view", "x (cm)", "z (cm)")]):
        ax.add_patch(Circle((0, 0), 11, fc="#f2f2f2", ec="#555555", lw=0.6))
        ax.scatter(m[:8, i], m[:8, j], s=14, c="#1f4e79", zorder=3, label="equatorial")
        ax.scatter(m[8, i], m[8, j], s=18, marker="^", c="#c55a11", zorder=3, label="crown")
        ax.scatter(m[9, i], m[9, j], s=18, marker="s", c="#548235", zorder=3, label="chin")
        ax.annotate("", xy=(15, 0), xytext=(11.5, 0), arrowprops=dict(arrowstyle="->", lw=0.7))
        ax.text(15.2, 0.8, "fwd", fontsize=6)
        ax.set_xlim(-14, 17); ax.set_ylim(-14, 14); ax.set_aspect("equal")
        ax.set_title(ti, fontsize=7); ax.set_xlabel(xl); ax.set_ylabel(yl)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig2_array.png", dpi=300); plt.close(fig)


def fig_doa():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.1), sharey=True)
    for ax, (fn, ti) in zip(axs, [("doa_ground_s1.json", "(a) open road"), ("doa_urban_s2.json", "(b) urban (+ wall reflection)")]):
        d = json.load(open(os.path.join(HERE, fn)))["summary"]
        snr = sorted(int(s) for s in d)
        for k, mth in enumerate(["DAS", "MVDR", "SRP-PHAT", "SRP-PHAT-FF"]):
            mu = np.array([d[str(s)][mth]["mean"] for s in snr])
            lo = np.array([d[str(s)][mth]["ci_lo"] for s in snr]); hi = np.array([d[str(s)][mth]["ci_hi"] for s in snr])
            x = np.array(snr) + (k - 1.5) * 0.35
            ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=COL[mth], marker="os^D"[k], ms=2.6, lw=0.9,
                        capsize=1.2, elinewidth=0.6, label=LAB[mth], ls="--" if mth == "SRP-PHAT-FF" else "-")
        ax.set_yscale("log"); ax.set_title(ti, fontsize=7); ax.set_xlabel("SNR per microphone (dB)")
        ax.grid(alpha=0.25, which="both", lw=0.4); ax.set_xticks([-10, 0, 10, 20])
    axs[0].set_ylabel("Mean |azimuth error| (deg)")
    axs[0].legend(loc="upper right", fontsize=5.4, frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig3_doa.png", dpi=300); plt.close(fig)


def fig_lpct():
    r = json.load(open(os.path.join(HERE, "lpct_results.json")))
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    names = ["RTTC", "BW", "LPCT-L", "LPCT-Y"]
    for k, n in enumerate(names):
        p = [100 * r["main"][f"{phi}_5"][n]["p"] for phi in (0, 20, 40)]
        lo = [100 * r["main"][f"{phi}_5"][n]["lo"] for phi in (0, 20, 40)]
        hi = [100 * r["main"][f"{phi}_5"][n]["hi"] for phi in (0, 20, 40)]
        x = np.arange(3) + (k - 1.5) * 0.2
        ax.bar(x, p, 0.19, color=COL[n], label=LAB[n], yerr=[np.subtract(p, lo), np.subtract(hi, p)],
               error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(range(3)); ax.set_xticklabels(["0°", "20°", "40°"]); ax.set_xlabel("Lean angle")
    ax.set_ylabel("Priority inversion (%)"); ax.set_ylim(0, 75); ax.set_title("(a) $N_{obj}$ = 5", fontsize=7); ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax = axs[1]
    for key, xs, lab, mk in (("bias", [-5, -3, 0, 3, 5], "lean bias (deg)", "o"),):
        for n in ("RTTC", "LPCT-L", "LPCT-Y"):
            p = np.array([100 * r[key][str(b)][n]["p"] for b in xs])
            lo = np.array([100 * r[key][str(b)][n]["lo"] for b in xs]); hi = np.array([100 * r[key][str(b)][n]["hi"] for b in xs])
            ax.errorbar(xs, p, yerr=[p - lo, hi - p], color=COL[n], marker=mk, ms=2.5, lw=0.9, capsize=1, elinewidth=0.5)
        ax.set_xlabel("Lean-angle bias (deg)")
    ax.set_title("(b) 40° lean, bias sweep", fontsize=7); ax.set_ylim(0, 75); ax.grid(alpha=0.25, lw=0.4)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=5.5, frameon=False)
    fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 1)); fig.savefig(f"{OUT}/fig4_lpct.png", dpi=300); plt.close(fig)


def fig_graphical_abstract():
    d = json.load(open(os.path.join(HERE, "doa_ground_s1.json")))["summary"]
    r = json.load(open(os.path.join(HERE, "lpct_results.json")))
    fig, axs = plt.subplots(1, 2, figsize=(4.2, 1.9))
    snr = sorted(int(s) for s in d)
    for mth in ("DAS", "SRP-PHAT", "SRP-PHAT-FF"):
        axs[0].plot(snr, [d[str(s)][mth]["mean"] for s in snr], marker="o", ms=2.5, color=COL[mth], label=LAB[mth],
                    ls="--" if mth == "SRP-PHAT-FF" else "-")
    axs[0].set_yscale("log"); axs[0].set_xlabel("SNR (dB)"); axs[0].set_ylabel("DoA error (deg)")
    axs[0].legend(fontsize=5, frameon=False); axs[0].set_title("Helmet-array DoA", fontsize=7); axs[0].grid(alpha=0.25, lw=0.4)
    for k, n in enumerate(["RTTC", "BW", "LPCT-L"]):
        axs[1].bar(np.arange(3) + (k - 1) * 0.27, [100 * r["main"][f"{p}_5"][n]["p"] for p in (0, 20, 40)], 0.26,
                   color=COL[n], label=LAB[n])
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["0°", "20°", "40°"]); axs[1].set_xlabel("Lean angle")
    axs[1].set_ylabel("Priority inversion (%)"); axs[1].legend(fontsize=5, frameon=False); axs[1].set_ylim(0, 85)
    axs[1].set_title("Lean-path prioritization", fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/graphical_abstract.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_array(); fig_doa(); fig_lpct(); fig_graphical_abstract()
    print(sorted(os.listdir(OUT)))
