import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
Shared plotting helpers for the rate two-region experiments (sensory-loss and
pharmacological). The plot functions are epoch-count-agnostic (they read
sim.epochBoundaries), so they work for both the 4-epoch and 3-epoch protocols.
'''


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


def plot_rates(sim, outdir, tau, prefix, title):
    pops = [("InputA", "Sensory Input (S_alpha)"),
            ("pyramidalsA", "Pyramidal (P_alpha)"),
            ("fastSpikingsA", "Fast Spiking (F_alpha)"),
            ("lowThresholdsA", "Low-Threshold Spiking (L_alpha)")]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, (pop, subtitle) in zip(axes, pops):
        rec = sim.network.populations[pop].rateRecord
        t = np.arange(len(rec)) * tau
        ax.plot(t, smooth(rec, 20.0, tau), color="C0")
        ax.set_title(subtitle, loc="left", fontsize=10)
        ax.set_ylabel("Rate (Hz)")
        epoch_lines(ax, sim.epochBoundaries)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(outdir, prefix + "_rates.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_influence(sim, outdir, tau, prefix, title):
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
    ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = os.path.join(outdir, prefix + "_influence.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_weights(sim, outdir, tau, prefix, title):
    w = sim.weightHistory
    t = np.arange(len(w)) * tau
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, w, color="C2")
    epoch_lines(ax, sim.epochBoundaries)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean remapping weight (S_B -> P_A)")
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(outdir, prefix + "_weights.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
