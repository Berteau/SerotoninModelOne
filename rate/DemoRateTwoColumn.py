import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate.RateTwoColumnParams import buildDefaultParams
from rate.RateTwoColumnSimulation import RateTwoColumnSimulation

'''
Runs the rate two-region sensory-loss simulation and writes paper-style
figures for validating the rate reformulation against the methods draft:

  rate_twocolumn_rates.png   - region-A firing rates (Input / Pyramidal / FS /
                               LTS) across the four epochs, matching the draft's
                               Figure "Neural activity in sensory loss
                               simulations" layout.
  rate_twocolumn_influence.png - relative influence of Input alpha (negative)
                               vs Input beta (positive) on alpha pyramidal
                               firing, matching the draft's "Relative influence"
                               figure. Beta's influence should rise in epoch 3
                               and persist into epoch 4.
  rate_twocolumn_weights.png - mean remapping-synapse (S_B -> P_A) weight over
                               time, showing bounded potentiation locked in
                               during epoch 3.

Usage: python3 rate/DemoRateTwoColumn.py
'''

FIG_DIR = os.path.join(os.curdir, "figures", "rate_twocolumn")


def smooth(series, win_ms, tau):
    # Boxcar smoothing with edge-extension (mode="nearest"), NOT zero-padding.
    # np.convolve(mode="same") zero-pads the ends, which drags the first/last
    # ~half-window of samples toward zero and creates a spurious ramp at t=0 in
    # otherwise-flat traces; uniform_filter1d(mode="nearest") holds the edge
    # value instead.
    from scipy.ndimage import uniform_filter1d
    n = max(1, int(win_ms / tau))
    s = np.asarray(series, dtype=float)
    if len(s) < n:
        return s
    return uniform_filter1d(s, size=n, mode="nearest")


def epoch_lines(ax, boundaries):
    for b in boundaries[:-1]:
        ax.axvline(b, color="0.6", ls="--", lw=1)


def plot_rates(sim, outdir, tau):
    boundaries = sim.epochBoundaries
    pops = [("InputA", "Sensory Input (S_alpha)"),
            ("pyramidalsA", "Pyramidal (P_alpha)"),
            ("fastSpikingsA", "Fast Spiking (F_alpha)"),
            ("lowThresholdsA", "Low-Threshold Spiking (L_alpha)")]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, (pop, title) in zip(axes, pops):
        rec = sim.network.populations[pop].rateRecord
        t = np.arange(len(rec)) * tau
        ax.plot(t, smooth(rec, 20.0, tau), color="C0")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Rate (Hz)")
        epoch_lines(ax, boundaries)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Rate model: region-A activity across sensory-loss epochs")
    fig.tight_layout()
    path = os.path.join(outdir, "rate_twocolumn_rates.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_influence(sim, outdir, tau):
    pa = sim.network.populations["pyramidalsA"]
    inputA = sim.network.populations["InputA"]
    inputB = sim.network.populations["InputB"]
    # Leave-one-out ablation influence of each input on P_alpha's commanded rate.
    infA = np.array(pa.ablationInfluence[inputA])
    infB = np.array(pa.ablationInfluence[inputB])
    t = np.arange(len(infA)) * tau
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, -smooth(infA, 20.0, tau), color="C3", label="Input alpha (own, plotted negative)")
    ax.plot(t, smooth(infB, 20.0, tau), color="C0", label="Input beta (cross-modal, positive)")
    ax.axhline(0, color="0.7", lw=0.8)
    epoch_lines(ax, sim.epochBoundaries)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Ablation influence on P_alpha rate (Hz)")
    ax.set_title("Relative influence of Inputs alpha/beta on alpha pyramidal firing (leave-one-out)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = os.path.join(outdir, "rate_twocolumn_influence.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_weights(sim, outdir, tau):
    w = sim.weightHistory
    t = np.arange(len(w)) * tau
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, w, color="C2")
    epoch_lines(ax, sim.epochBoundaries)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean remapping weight (S_B -> P_A)")
    ax.set_title("Cross-modal remapping synapse potentiation (bounded)")
    fig.tight_layout()
    path = os.path.join(outdir, "rate_twocolumn_weights.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    random.seed(1)
    np.random.seed(1)
    params = buildDefaultParams()

    outdir = os.path.join(FIG_DIR, datetime.utcnow().isoformat())
    os.makedirs(outdir, exist_ok=True)

    print("Running rate two-region sensory-loss simulation "
          "(popCount=%d, epoch=%dms)..." % (params["popCount"], params["epochDurationMs"]))
    sim = RateTwoColumnSimulation(params)
    sim.run()

    tau = params["tau"]
    paths = [plot_rates(sim, outdir, tau),
             plot_influence(sim, outdir, tau),
             plot_weights(sim, outdir, tau)]
    print("Wrote:")
    for p in paths:
        print("  " + p)


if __name__ == "__main__":
    main()
