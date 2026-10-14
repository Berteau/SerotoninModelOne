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


def separability(features, labels):
    """Within- vs between-class RMS distance in feature space. ratio > 1 means
    classes are, on average, farther apart than samples within a class (good)."""
    X = np.asarray(features, dtype=float); y = np.asarray(labels)
    classes = sorted(set(y.tolist()))
    cm = {c: X[y == c].mean(0) for c in classes}
    within = np.mean([np.sqrt(((X[y == c] - cm[c]) ** 2).mean()) for c in classes if np.any(y == c)])
    pairs = [(i, j) for i in classes for j in classes if i < j]
    between = np.mean([np.sqrt(((cm[i] - cm[j]) ** 2).mean()) for i, j in pairs]) if pairs else float("nan")
    return float(within), float(between), float(between / within) if within else float("nan")


def plot_baseline_clusters(blob, outpath, title_prefix=""):
    """PCA (first 2 PCs) scatter of the SIGHTED-BASELINE V1 features (pre-loss),
    colour = assigned cluster, marker = true visual class. Shows directly how well
    the classifier's clusters separate the classes before deprivation, for tuning
    vigilance. Returns outpath (or None if features were not recorded)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    feats = blob.get("featureSeq")
    if not feats:
        return None
    li = blob["lossPresentationIndex"] or len(feats)
    X = np.asarray(feats[:li], dtype=float)
    cl = np.asarray(blob["assignedCluster"][:li])
    tc = np.asarray(blob["trueClass"][:li])
    if len(X) < 3:
        return None
    classNames = blob.get("classNames") or [str(i) for i in sorted(set(tc.tolist()))]
    vig = blob.get("params", {}).get("artVigilance", None)

    # PCA via SVD on mean-centred features
    mu = X.mean(0); Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    ev = (S ** 2); ev = ev / ev.sum() if ev.sum() else ev

    wi, be, ratio = separability(X, tc)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]
    clusters = sorted(set(cl.tolist()))
    cmap = plt.get_cmap("tab10" if len(clusters) <= 10 else "tab20")
    cof = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cls in sorted(set(tc.tolist())):
        for cu in clusters:
            m = (tc == cls) & (cl == cu)
            if not m.any():
                continue
            ax.scatter(proj[m, 0], proj[m, 1], color=[cof[cu]],
                       marker=markers[cls % len(markers)], s=42,
                       edgecolor="k", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("PC1 (%.0f%% var)" % (100 * ev[0]))
    ax.set_ylabel("PC2 (%.0f%% var)" % (100 * ev[1]))
    kind = blob.get("classifier", "")
    ttl = "%s baseline V1 features: colour=cluster, marker=class" % (title_prefix + ":" if title_prefix else "")
    detail = ("vigilance=%.2f  " % vig) if (vig is not None and "ART" in kind) else ""
    sub = "%sdistinct baseline clusters=%d  between/within RMS=%.2f" % (detail, len(clusters), ratio)
    ax.set_title(ttl + "\n" + sub, fontsize=10)

    class_handles = [Line2D([0], [0], marker=markers[cls % len(markers)], color="w",
                            markerfacecolor="0.6", markeredgecolor="k",
                            label="class: %s" % classNames[cls]) for cls in sorted(set(tc.tolist()))]
    cluster_handles = [Patch(facecolor=cof[cu], edgecolor="k",
                             label="cluster %d" % cu if cu >= 0 else "reject") for cu in clusters]
    leg1 = ax.legend(handles=class_handles, loc="upper left", fontsize=8, title="true class")
    ax.add_artist(leg1)
    ax.legend(handles=cluster_handles, loc="upper right", fontsize=8, title="assigned cluster")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120); plt.close(fig)
    return outpath
