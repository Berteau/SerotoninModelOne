import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate.plotting import smooth, epoch_lines

'''
Plotting for the retinotopic grand-plan sensory-loss experiment. Each figure
overlays the plasticity condition against the no-plasticity control so the
causal role of the cross-modal remapping synapses is visible directly.

Panels:
  1. V1 firing rate in fully-deprived vs intact columns.
  2. Stream currents into deprived-column V1: visual (bottom-up, lost),
     audio (cross-modal, the remapping readout), top-down (Category feedback).
  3. Mean audio -> V1 remapping-synapse weight.
'''

EPOCH_LABELS = ["baseline", "loss", "5HT + plasticity", "return"]


def _time(sim):
    return np.arange(len(sim.remapWeight)) * sim.tau


def _epoch_labels(ax, sim):
    epoch_lines(ax, sim.epochBoundaries)
    for i, b in enumerate([0] + list(sim.epochBoundaries[:-1])):
        if i < len(EPOCH_LABELS):
            ax.text(b + 5, ax.get_ylim()[1], EPOCH_LABELS[i], fontsize=7,
                    va="top", ha="left", color="0.4")


def plot_experiment(sim, control, outdir, title_prefix):
    tau = sim.tau
    t = _time(sim)
    win = 20.0
    os.makedirs(outdir, exist_ok=True)
    paths = []

    # --- Panel 1: rates ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, smooth(sim.rateDeprived, win, tau), color="C3", label="deprived V1 (plasticity)")
    ax.plot(t, smooth(sim.rateIntact, win, tau), color="C0", label="intact V1 (plasticity)")
    if control is not None:
        ax.plot(t, smooth(control.rateDeprived, win, tau), color="C3", ls="--",
                label="deprived V1 (control, no plasticity)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("V1 firing rate (Hz)")
    ax.set_title("%s: V1 firing rate, deprived vs intact columns" % title_prefix)
    _epoch_labels(ax, sim); ax.legend(fontsize="small")
    fig.tight_layout()
    p = os.path.join(outdir, "rates.png"); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # --- Panel 2: stream currents into deprived V1 ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, smooth(sim.visualCurrent, win, tau), color="C0", label="visual bottom-up (lost at loss)")
    ax.plot(t, smooth(sim.audioCurrent, win, tau), color="C2", label="audio cross-modal (remapping)")
    ax.plot(t, smooth(sim.topDownCurrent, win, tau), color="C1", label="top-down (Category feedback)")
    if control is not None:
        ax.plot(t, smooth(control.audioCurrent, win, tau), color="C2", ls="--",
                label="audio cross-modal (control)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Raw stream current into deprived V1")
    ax.set_title("%s: input streams to deprived-column V1" % title_prefix)
    _epoch_labels(ax, sim); ax.legend(fontsize="small")
    fig.tight_layout()
    p = os.path.join(outdir, "streams.png"); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # --- Panel 3: remapping weight ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, sim.remapWeight, color="C2", label="plasticity")
    if control is not None:
        ax.plot(t, control.remapWeight, color="0.5", ls="--", label="control (no plasticity)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Mean audio -> V1 weight")
    ax.set_title("%s: cross-modal remapping-synapse weight" % title_prefix)
    _epoch_labels(ax, sim); ax.legend(fontsize="small")
    fig.tight_layout()
    p = os.path.join(outdir, "remap_weight.png"); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    return paths


def epoch_means(sim):
    # Convenience: per-epoch means of the key series for a compact text summary.
    tau = sim.tau
    dur = sim.epochDurationMs
    out = {}
    for name in ("rateDeprived", "rateIntact", "audioCurrent", "visualCurrent",
                 "topDownCurrent", "remapWeight"):
        arr = np.asarray(getattr(sim, name), dtype=float)
        vals = []
        for e in range(4):
            s = int(e * dur / tau); en = int((e + 1) * dur / tau)
            seg = arr[s:en]
            vals.append(float(np.nanmean(seg)) if len(seg) else float("nan"))
        out[name] = vals
    return out
