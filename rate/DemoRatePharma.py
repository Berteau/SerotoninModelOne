import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime

import numpy as np

from rate.RateTwoColumnParams import buildDefaultParams
from rate.RatePharmaSimulation import RatePharmaSimulation
from rate.plotting import plot_rates, plot_influence, plot_weights

'''
Runs the rate pharmacological (5HT2A-agonist / psilocybin) simulation and writes
the same three figure types as the sensory-loss demo, over the pharma protocol's
three epochs (baseline / 5HT2A agonism + weak plasticity / return). Mean-only:
the deterministic rate model captures the mean pharmacological behaviour (Input
alpha keeps its dominant influence) but not the paper's variability finding.

Usage: python3 rate/DemoRatePharma.py
'''

FIG_DIR = os.path.join(os.curdir, "figures", "rate_pharma")


def main():
    random.seed(1)
    np.random.seed(1)
    params = buildDefaultParams()

    outdir = os.path.join(FIG_DIR, datetime.utcnow().isoformat())
    os.makedirs(outdir, exist_ok=True)

    print("Running rate pharmacological (5HT2A-agonist) simulation "
          "(popCount=%d, epoch=%dms)..." % (params["popCount"], params["epochDurationMs"]))
    sim = RatePharmaSimulation(params)
    sim.run()

    tau = params["tau"]
    paths = [
        plot_rates(sim, outdir, tau, "rate_pharma",
                   "Rate model: region-A activity across pharmacological (5HT2A-agonist) epochs"),
        plot_influence(sim, outdir, tau, "rate_pharma",
                       "Relative influence of Inputs alpha/beta on alpha pyramidal firing (pharma, leave-one-out)"),
        plot_weights(sim, outdir, tau, "rate_pharma",
                     "Cross-modal S_B -> P_A weight under 5HT2A agonism (bounded)"),
    ]
    print("Wrote:")
    for p in paths:
        print("  " + p)


if __name__ == "__main__":
    main()
