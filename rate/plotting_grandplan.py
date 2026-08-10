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

    # --- Panel 5: ART classifier readout (only when the ART secondary area ran) ---
    if getattr(sim, "artMatch", None):
        p = plot_art(sim, outdir, title_prefix)
        if p is not None:
            paths.append(p)

    # --- Panel 6: emergent homeostatic controller (only when it ran) ---
    hd = getattr(sim, "homeostaticDrive", None)
    if hd and np.any(np.asarray(hd, dtype=float) > 1e-6):
        p = plot_homeostatic(sim, outdir, title_prefix)
        if p is not None:
            paths.append(p)

    return paths


def plot_homeostatic(sim, outdir, title_prefix):
    # Emergent controller: deprived firing (left axis) and the latent drive h
    # (right axis) over time, showing that the drop -> hyperactivity -> settle
    # trajectory and the response ramp/relaxation emerge from the activity deficit
    # -- the only imposed event is the loss.
    tau = sim.tau
    rate = np.asarray(sim.rateDeprived, dtype=float)
    h = np.asarray(sim.homeostaticDrive, dtype=float)
    t = np.arange(len(rate)) * tau
    fig, axL = plt.subplots(figsize=(9, 4))
    axL.plot(t, smooth(rate, 40.0, tau), color="C3", label="deprived V1 rate")
    if sim.A_set is not None:
        axL.axhline(sim.A_set, color="C3", ls=":", lw=1, label="baseline set-point")
    axL.set_xlabel("Time (ms)"); axL.set_ylabel("V1 firing rate (Hz)", color="C3")
    axL.tick_params(axis="y", labelcolor="C3")
    axR = axL.twinx()
    axR.plot(t, h, color="C0", label="homeostatic drive h")
    axR.set_ylabel("controller drive h", color="C0"); axR.set_ylim(0, 1.05)
    axR.tick_params(axis="y", labelcolor="C0")
    epoch_lines(axL, sim.epochBoundaries)
    axL.set_title("%s: emergent homeostatic response (only the loss is imposed)" % title_prefix)
    l1, la = axL.get_legend_handles_labels(); l2, lb = axR.get_legend_handles_labels()
    axL.legend(l1 + l2, la + lb, fontsize="small", loc="upper right")
    fig.tight_layout()
    path = os.path.join(outdir, "homeostatic.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_art(sim, outdir, title_prefix):
    # ART secondary-area readout over the experiment: the one-hot decision
    # (class vs reject) and the match/confidence, so hallucination (confident
    # classification without veridical input) and recovery are visible.
    tau = sim.tau
    cls = np.asarray(sim.artClass, dtype=float)     # class label, or -1 (reject)
    match = np.asarray(sim.artMatch, dtype=float)
    t = np.arange(len(cls)) * tau
    classNames = getattr(sim.inputSet, "classNames", None)
    true_lbl = getattr(sim, "trueLabel", None)

    fig, (axd, axm) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    # Decision strip: reject at -1, classes at 0,1,...
    axd.plot(t, cls, color="0.3", lw=1.0, drawstyle="steps-post")
    axd.scatter(t[::50], cls[::50], c=["0.6" if v < 0 else ("C2" if v == true_lbl else "C3")
                                       for v in cls[::50]], s=8, zorder=3)
    if true_lbl is not None:
        axd.axhline(true_lbl, color="C2", ls=":", lw=1, label="true class")
    ncls = (len(classNames) if classNames else int(np.nanmax(cls)) + 1)
    axd.set_yticks([-1] + list(range(ncls)))
    axd.set_yticklabels(["reject"] + (classNames if classNames else [str(i) for i in range(ncls)]))
    axd.set_ylabel("ART decision"); axd.set_ylim(-1.5, ncls - 0.5)
    epoch_lines(axd, sim.epochBoundaries); axd.legend(fontsize="small", loc="upper right")
    axd.set_title("%s: ART secondary-area readout (classify / reject)" % title_prefix)

    axm.plot(t, match, color="C0", label="match / confidence")
    vig = sim.params.get("artVigilance", None)
    if vig is not None:
        axm.axhline(vig, color="C3", ls="--", lw=1, label="vigilance (reject below)")
    axm.set_ylabel("ART match"); axm.set_xlabel("Time (ms)"); axm.set_ylim(0, 1)
    epoch_lines(axm, sim.epochBoundaries); axm.legend(fontsize="small", loc="lower right")

    fig.tight_layout()
    path = os.path.join(outdir, "art_readout.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


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
