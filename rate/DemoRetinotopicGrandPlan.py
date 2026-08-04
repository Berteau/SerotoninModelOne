import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from datetime import datetime

import numpy as np

from rate.RetinotopicGrandPlanParams import buildGrandPlanParams
from rate.RetinotopicGrandPlanSimulation import RetinotopicGrandPlanSimulation
from rate.StructuredInput import StructuredInputSet, ImageFolderInputSet
from rate.plotting_grandplan import plot_experiment, epoch_means, EPOCH_LABELS

'''
Runs the retinotopic grand-plan sensory-loss experiment for both loss modes
(scotoma and full blindness), each paired with a no-plasticity control, and
writes the rate / stream-current / remapping-weight figures plus a text summary.

Usage:
  python3 rate/DemoRetinotopicGrandPlan.py [--grid 6] [--cpc 6] [--epoch 500]
                                           [--warmup 200] [--mode both|scotoma|blind]

Grid defaults to 6 for a runnable first look; the target architecture is 10.
'''

FIG_DIR = os.path.join(os.curdir, "figures", "rate_grandplan")


def run_condition(params, data, stim, mode, plasticity):
    random.seed(0); np.random.seed(0)
    sim = RetinotopicGrandPlanSimulation(params, stim, data, mode=mode,
                                         plasticityEnabled=plasticity)
    return sim.run()


def summarize(mode, sim, control):
    em = epoch_means(sim)
    emc = epoch_means(control)
    print("\n=== %s ===" % mode.upper())
    print("  deprived columns: %d   intact columns: %d"
          % (sum(sim.deprivedColumns),
             sum(1 for c in sim.silencedColumns if not c)))
    hdr = "  %-16s " % "epoch" + " ".join("%10s" % e for e in EPOCH_LABELS)
    print(hdr)
    def row(label, vals):
        print("  %-16s " % label + " ".join("%10.2f" % v for v in vals))
    row("rate deprived", em["rateDeprived"])
    row("rate intact", em["rateIntact"])
    row("visual I", em["visualCurrent"])
    row("audio I (remap)", em["audioCurrent"])
    row("top-down I", em["topDownCurrent"])
    row("remap W", em["remapWeight"])
    row("remap W (ctrl)", emc["remapWeight"])
    row("audio I (ctrl)", emc["audioCurrent"])
    row("rate depr (ctrl)", emc["rateDeprived"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--cpc", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mode", choices=["both", "scotoma", "blind"], default="both")
    ap.add_argument("--images", default=None,
                    help="root folder of labeled image subfolders (person/ horse/); "
                         "if omitted, synthetic class-distinct patterns are used")
    ap.add_argument("--stamp", default=None, help="output subdir name (default: UTC timestamp)")
    args = ap.parse_args()

    params = buildGrandPlanParams(gridSize=args.grid, cellsPerColumn=args.cpc, categoryCount=2)
    params["epochDurationMs"] = args.epoch
    params["warmupMs"] = args.warmup

    if args.images:
        data = ImageFolderInputSet(args.images, gridSize=args.grid, audioBins=params["audioBins"],
                                   imagesPerClass=1, seed=0,
                                   minRateHz=params["minRateHz"], maxRateHz=params["maxRateHz"])
        print("Using real images from %s: classes %s" % (args.images, data.classNames))
    else:
        data = StructuredInputSet(gridSize=args.grid, audioBins=params["audioBins"],
                                  classCount=2, seed=0,
                                  minRateHz=params["minRateHz"], maxRateHz=params["maxRateHz"])
    stim = data.stimuli[0]

    stamp = args.stamp or datetime.utcnow().isoformat()
    modes = ["scotoma", "blind"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print("Running %s (grid=%d, cpc=%d, epoch=%dms)..."
              % (mode, args.grid, args.cpc, args.epoch))
        sim = run_condition(params, data, stim, mode, plasticity=True)
        control = run_condition(params, data, stim, mode, plasticity=False)
        summarize(mode, sim, control)
        outdir = os.path.join(FIG_DIR, stamp, mode)
        paths = plot_experiment(sim, control, outdir, mode.capitalize())
        print("  wrote:")
        for p in paths:
            print("    " + p)


if __name__ == "__main__":
    main()
