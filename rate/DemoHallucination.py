import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rate.RetinotopicGrandPlanParams import buildGrandPlanParams
from rate.RetinotopicGrandPlanSimulation import RetinotopicGrandPlanSimulation
from rate.StructuredInput import ImageFolderInputSet, StructuredInputSet

'''
Visualize the "hallucination": the percept that TOP-DOWN feedback alone paints on
V1. We run the ART sensory-loss experiment (which, after remapping, stably
recognizes the wrong category -- "person" for a horse stimulus), take the ART
category recognized during the RETURN epoch, and drive V1 with ONLY that
category's top-down template (bottom-up disconnected -- Sulfaro's sensory
disconnection), sweeping the top-down synaptic gain.

At the influence the model actually exhibits (gain x1) the percept is faint (top-
down is diluted by the normalized mixture -- Sulfaro's account of why imagery is
dim); amplified, the hallucinated category emerges vividly on V1, and its spatial
structure is the (mis)recognized category, not the veridical stimulus.

Usage:
  python3 rate/DemoHallucination.py --grid 10 --cpc 8 --mode blind \
      --images <classified_stimuli/classified_stimuli> --gains 1 2 4 8
'''

FIG_DIR = os.path.join(os.curdir, "figures", "rate_grandplan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--cpc", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--mode", choices=["scotoma", "blind"], default="blind")
    ap.add_argument("--images", default=None)
    ap.add_argument("--gains", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--settle", type=float, default=400.0, help="top-down render settle (ms)")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0)
    params = buildGrandPlanParams(gridSize=args.grid, cellsPerColumn=args.cpc, categoryCount=2)
    params["epochDurationMs"] = args.epoch
    params["warmupMs"] = args.warmup
    params["useART"] = True

    if args.images:
        data = ImageFolderInputSet(args.images, gridSize=args.grid, audioBins=params["audioBins"],
                                   imagesPerClass=5, seed=0,
                                   minRateHz=params["minRateHz"], maxRateHz=params["maxRateHz"])
        names = data.classNames
    else:
        data = StructuredInputSet(gridSize=args.grid, audioBins=params["audioBins"],
                                  classCount=2, seed=0,
                                  minRateHz=params["minRateHz"], maxRateHz=params["maxRateHz"])
        names = [str(i) for i in range(2)]
    stim = data.stimuli[0]

    print("Running %s ART experiment (grid=%d) to obtain the return-epoch percept..."
          % (args.mode, args.grid))
    sim = RetinotopicGrandPlanSimulation(params, stim, data, mode=args.mode, plasticityEnabled=True)
    sim.run()
    net, art = sim.network, sim.art

    # Modal winning ART category during the return epoch (epoch index 3).
    tau, dur = params["tau"], params["epochDurationMs"]
    s, e = int(3 * dur / tau), int(4 * dur / tau)
    cats = [c for c in sim.artCategory[s:e] if c is not None]
    if not cats:
        print("Return epoch rejected throughout; no stable percept to render.")
        return
    win = Counter(cats).most_common(1)[0][0]
    winName = names[art.labels[win]]
    print("Return-epoch percept: category %d = '%s' (true stimulus '%s')"
          % (win, winName, names[stim.label]))

    template = art.category_template(win)
    percepts = [net.renderTopDownPercept(template, settleMs=args.settle, gain=g) for g in args.gains]
    print("percept mean Hz by gain:", dict(zip(args.gains, [round(float(p.mean()), 2) for p in percepts])))

    G = args.grid
    ncol = 2 + len(args.gains)
    fig, ax = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.5))
    ax[0].imshow(stim.visualGrid, cmap="magma"); ax[0].set_title("veridical input\n(%s)" % names[stim.label], fontsize=9)
    ax[1].imshow(template.reshape(G, G), cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("ART '%s' template\n(top-down feedback)" % winName, fontsize=9)
    vmax = max(1e-6, max(p.max() for p in percepts))
    for a, g, p in zip(ax[2:], args.gains, percepts):
        im = a.imshow(p, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title("V1 hallucination\ntop-down x%g" % g, fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[2:].tolist(), fraction=0.02, pad=0.01, label="Hz")
    fig.suptitle("Pure top-down percept: what V1 renders under the '%s' hallucination (%s)"
                 % (winName, args.mode), y=1.03)

    stamp = args.stamp or datetime.utcnow().isoformat()
    outdir = os.path.join(FIG_DIR, stamp, args.mode)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "hallucination.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
