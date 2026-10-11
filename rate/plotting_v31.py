import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
Plots + metrics for the v3.1 sequence experiment. The classifier is a moving
target (slow continuous learning), so everything is measured over time:

  n_clusters(t)          do new categories appear?
  ARI vs visual class    alignment of the assignment to the ORIGINAL visual
  ARI vs sound-clip ID    labels (should fall) vs the audio structure (should
                          rise) as audio comes to dominate.
  hallucination rate     deprived-phase presentations mapped to a stale
                          WRONG-visual-class cluster.
  rate + h               to relate emergence to the fading intrinsic drive.
'''


def adjusted_rand_index(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    n = len(a)
    if n == 0:
        return float("nan")
    cont = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        cont[ua[x], ub[y]] += 1
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_ij = comb2(cont).sum()
    sum_i = comb2(cont.sum(1)).sum()
    sum_j = comb2(cont.sum(0)).sum()
    total = comb2(n)
    expected = sum_i * sum_j / total if total else 0.0
    maxi = 0.5 * (sum_i + sum_j)
    denom = maxi - expected
    if denom == 0:
        return 1.0
    return float((sum_ij - expected) / denom)


def _seg_windows(lo, hi, win, step):
    # Sliding windows within [lo, hi); if the segment is shorter than win, emit
    # one window covering the whole segment (so a short baseline still yields a
    # point). Never spans across the segment boundary (the loss).
    n = hi - lo
    if n <= 0:
        return
    if n < win:
        yield lo, hi
        return
    i = lo
    while i + win <= hi:
        yield i, i + win
        i += step


def compute_series(blob, win=40, step=10):
    cluster = np.asarray(blob["assignedCluster"])
    mapped = np.asarray(blob["mappedLabel"])
    truec = np.asarray(blob["trueClass"])
    clip = np.asarray(blob["clipIdRec"])
    t = np.asarray(blob["recTime"], dtype=float)
    loss_i = blob["lossPresentationIndex"]
    n = len(cluster)
    li = loss_i if (loss_i is not None and 0 < loss_i < n) else n
    out = {"t_mid": [], "ari_visual": [], "ari_clip": [], "halluc": []}
    # Compute windows SEPARATELY within the sighted baseline [0, li) and the
    # post-loss segment [li, n), so no window straddles the loss.
    for seg_lo, seg_hi in ((0, li), (li, n)):
        for lo, hi in _seg_windows(seg_lo, seg_hi, win, step):
            cl = cluster[lo:hi]; mp = mapped[lo:hi]; tc = truec[lo:hi]; cp = clip[lo:hi]
            out["t_mid"].append(float(np.mean(t[lo:hi])))
            out["ari_visual"].append(adjusted_rand_index(cl, tc))
            out["ari_clip"].append(adjusted_rand_index(cl, cp))
            real = mp >= 0
            wrong = real & (mp != tc)
            out["halluc"].append(float(np.sum(wrong)) / max(1, len(mp)))
    for k in out:
        out[k] = np.asarray(out[k], dtype=float)
    out["loss_t"] = t[loss_i] if (loss_i is not None and loss_i < len(t)) else None
    return out


def plot_v31(blob, outdir, title_prefix):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    s = compute_series(blob)
    t = np.asarray(blob["recTime"], dtype=float)
    loss_t = s["loss_t"]

    def _mark(ax):
        # Shade the sighted baseline (pre-loss) and annotate both phases, so the
        # pre-deprivation period is unmistakable and the "deprived" label does not
        # bleed into the sighted stretch.
        if loss_t is not None and loss_t > 0:
            ax.axvspan(0, loss_t, color="0.85", alpha=0.5, zorder=0)
            ax.axvline(loss_t, color="0.4", ls="--", lw=1)
            y = ax.get_ylim()[1]
            ax.text(loss_t * 0.5, y, "sighted baseline", fontsize=7, va="top",
                    ha="center", color="0.4")
            ax.text(loss_t, y, " deprived →", fontsize=7, va="top",
                    ha="left", color="0.4")

    # 1. cluster count + rate + h
    fig, axL = plt.subplots(figsize=(9, 4))
    axL.plot(t, blob["nClustersSeq"], color="C4", label="n_clusters")
    axL.set_xlabel("Time (ms)"); axL.set_ylabel("n clusters", color="C4")
    axL.tick_params(axis="y", labelcolor="C4")
    axR = axL.twinx()
    axR.plot(t, blob["rateDeprivedSeq"], color="C3", lw=1, alpha=0.8, label="V1 rate (deprived region)")
    axR.plot(t, np.asarray(blob["homeostaticDriveSeq"]) * (np.nanmax(blob["rateDeprivedSeq"]) or 1),
             color="C0", lw=1, ls=":", label="h (scaled)")
    axR.set_ylabel("V1 rate (Hz) / h", color="C3"); axR.tick_params(axis="y", labelcolor="C3")
    _mark(axL)
    axL.set_title("%s: cluster count vs deprived-region activity" % title_prefix)
    l1, la = axL.get_legend_handles_labels(); l2, lb = axR.get_legend_handles_labels()
    axL.legend(l1 + l2, la + lb, fontsize="small", loc="upper left")
    fig.tight_layout(); pth = os.path.join(outdir, "clusters.png")
    fig.savefig(pth, dpi=110); plt.close(fig); paths.append(pth)

    # 2. ARI vs visual class and vs clip id
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s["t_mid"], s["ari_visual"], color="C1", label="ARI vs visual class (old labels)")
    ax.plot(s["t_mid"], s["ari_clip"], color="C2", label="ARI vs sound-clip ID (audio structure)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Adjusted Rand Index"); ax.set_ylim(-0.1, 1.05)
    _mark(ax)
    ax.set_title("%s: what the clusters track over time" % title_prefix)
    ax.legend(fontsize="small"); fig.tight_layout()
    pth = os.path.join(outdir, "ari.png"); fig.savefig(pth, dpi=110); plt.close(fig); paths.append(pth)

    # 3. hallucination rate
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s["t_mid"], s["halluc"], color="C3", label="hallucination rate (wrong visual class)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("fraction of presentations"); ax.set_ylim(0, 1.0)
    _mark(ax)
    ax.set_title("%s: hallucination (stale wrong-class readout)" % title_prefix)
    ax.legend(fontsize="small"); fig.tight_layout()
    pth = os.path.join(outdir, "hallucination.png"); fig.savefig(pth, dpi=110); plt.close(fig); paths.append(pth)

    return paths, s
