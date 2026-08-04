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
    # Intact columns only exist when the lesion spares part of the field (a
    # scotoma). Under full blindness every column is deprived, so the intact
    # series is all-NaN -- skip it (and its legend entry) rather than draw a
    # phantom line.
    intact = np.asarray(sim.rateIntact, dtype=float)
    has_intact = np.any(np.isfinite(intact))
    if has_intact:
        ax.plot(t, smooth(sim.rateIntact, win, tau), color="C0", label="intact V1 (plasticity)")
    if control is not None:
        ax.plot(t, smooth(control.rateDeprived, win, tau), color="C3", ls="--",
                label="deprived V1 (control; == plasticity until 5HT epoch)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("V1 firing rate (Hz)")
    title_tail = "deprived vs intact columns" if has_intact else "deprived columns (no intact columns under full blindness)"
    ax.set_title("%s: V1 firing rate, %s" % (title_prefix, title_tail))
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

    # --- Panel 4: retinotopic V1 maps (edge input + per-epoch V1 activity) ---
    p = plot_maps(sim, outdir, title_prefix)
    if p is not None:
        paths.append(p)

    return paths


def plot_maps(sim, outdir, title_prefix):
    # Edge-detected input + per-epoch V1 column-rate maps, with the lesion
    # outlined, so the deprived region visibly darkens at loss and partially
    # re-lights after cross-modal remapping.
    if not sim.v1MapsByEpoch:
        return None
    epochs = sorted(sim.v1MapsByEpoch)
    maps = [sim.v1MapsByEpoch[e] for e in epochs]
    vmax = max(1e-6, max(float(np.nanmax(m)) for m in maps))
    n = len(epochs) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    axes[0].imshow(sim.inputGrid, cmap="magma")
    axes[0].set_title("edge input (V1 bottom-up)", fontsize=9)
    for ax, e, m in zip(axes[1:], epochs, maps):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title("V1 rate: %s" % EPOCH_LABELS[e - 1], fontsize=9)
        _outline_lesion(ax, sim.silencedMap)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01, label="Hz")
    fig.suptitle("%s: retinotopic V1 activity across epochs (lesion outlined)" % title_prefix)
    path = os.path.join(outdir, "v1_maps.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def _outline_lesion(ax, silencedMap):
    # Draw a thin outline around the silenced (lesioned) columns.
    G = silencedMap.shape[0]
    for y in range(G):
        for x in range(G):
            if not silencedMap[y, x]:
                continue
            if y == 0 or not silencedMap[y - 1, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="red", lw=1.2)
            if y == G - 1 or not silencedMap[y + 1, x]:
                ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color="red", lw=1.2)
            if x == 0 or not silencedMap[y, x - 1]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="red", lw=1.2)
            if x == G - 1 or not silencedMap[y, x + 1]:
                ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color="red", lw=1.2)


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
