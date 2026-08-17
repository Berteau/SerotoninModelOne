import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random
import numpy as np

from rate.RetinotopicGrandPlanParams import buildGrandPlanParams
from rate.RetinotopicGrandPlanSimulation import RetinotopicGrandPlanSimulation
from rate.StructuredInput import ImageFolderInputSet

'''
Run ONE emergent (option-B) sensory-loss simulation (one mode x one plasticity
condition) and pickle its recorded data, so the four sims of the experiment can
run as independent parallel processes (each ~90 min at 5000 ms epochs) rather
than sequentially. ART is OFF here (classifier kept separate per the current
plan); the emergent RATE trajectory is ART-independent.

Usage: python3 rate/EmergentRunOne.py <mode> <plast 0|1> <out.pkl> <images> [grid] [epoch]
'''

FIELDS = ["tau", "epochDurationMs", "epochBoundaries", "A_set",
          "rateDeprived", "rateIntact", "audioCurrent", "visualCurrent",
          "topDownCurrent", "remapWeight", "homeostaticDrive",
          "v1MapsByEpoch", "inputGrid", "silencedMap", "deprivedMap",
          "silencedColumns", "deprivedColumns"]


def main():
    mode = sys.argv[1]
    plast = bool(int(sys.argv[2]))
    out = sys.argv[3]
    images = sys.argv[4]
    grid = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    epoch = int(sys.argv[6]) if len(sys.argv) > 6 else 5000

    random.seed(0); np.random.seed(0)
    p = buildGrandPlanParams(gridSize=grid, cellsPerColumn=8, categoryCount=2)
    p["epochDurationMs"] = epoch
    p["warmupMs"] = 300
    p["emergent5HT"] = True
    p["gamma_p"] = p["gamma_p"] / 10.0          # >=10x lower plasticity rate
    data = ImageFolderInputSet(images, gridSize=grid, audioBins=p["audioBins"],
                               imagesPerClass=3, seed=0,
                               minRateHz=p["minRateHz"], maxRateHz=p["maxRateHz"])
    sim = RetinotopicGrandPlanSimulation(p, data.stimuli[0], data, mode=mode,
                                         plasticityEnabled=plast)
    # Per-cell rate history is unused here and dominates memory (~7 GB at 10x10);
    # switch it off so the four sims fit and can run in parallel.
    sim.network.setCellHistoryRecording(False)

    def _blob(completedEpoch):
        b = {f: getattr(sim, f) for f in FIELDS}
        b["params"] = p
        b["classNames"] = data.classNames
        b["trueLabel"] = data.stimuli[0].label
        b["mode"] = mode
        b["plasticity"] = plast
        b["completedEpoch"] = completedEpoch      # 4 == full run
        return b

    def _dump(b, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(b, fh)
        os.replace(tmp, path)                      # atomic: a restart can't leave a half-file

    # Resilience: after each epoch, atomically overwrite <out> with the partial
    # state so a worker restart mid-run loses at most one epoch. The final write
    # (completedEpoch=4) is identical in shape, so the plotter needs no changes.
    sim.checkpointFn = lambda s, ep: _dump(_blob(ep), out)

    sim.run()
    _dump(_blob(4), out)
    print("wrote %s  (mode=%s plast=%s, deprived final rate=%.2f)"
          % (out, mode, plast, sim.rateDeprived[-1] if sim.rateDeprived else float("nan")))


if __name__ == "__main__":
    main()
