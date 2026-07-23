import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime

import numpy as np

from rate.RateTwoColumnParams import buildDefaultParams
from rate.RateTwoColumnSimulation import RateTwoColumnSimulation
from rate.plotting import plot_rates, plot_influence, plot_weights

'''
Runs the rate two-region sensory-loss simulation and writes paper-style figures
validating the rate reformulation against the methods draft: region-A firing
rates across the four epochs, the leave-one-out relative influence of Input
alpha vs beta on P_alpha, and the bounded remapping-synapse potentiation.

Usage: python3 rate/DemoRateTwoColumn.py
'''

FIG_DIR = os.path.join(os.curdir, "figures", "rate_twocolumn")


def main():
    random.seed(1)
    np.random.seed(1)
    params = buildDefaultParams()

    outdir = os.path.join(FIG_DIR, datetime.utcnow().isoformat())
    os.makedirs(outdir, exist_ok=True)

    print("Running rate two-region sensory-loss simulation "
          "(popCount=%d, epoch=%dms)..." % (params["popCount"], params["epochDurationMs"]))
    sim = RateTwoColumnSimulation(params)
    sim.run()

    tau = params["tau"]
    paths = [
        plot_rates(sim, outdir, tau, "rate_twocolumn",
                   "Rate model: region-A activity across sensory-loss epochs"),
        plot_influence(sim, outdir, tau, "rate_twocolumn",
                       "Relative influence of Inputs alpha/beta on alpha pyramidal firing (leave-one-out)"),
        plot_weights(sim, outdir, tau, "rate_twocolumn",
                     "Cross-modal remapping synapse potentiation (bounded)"),
    ]
    print("Wrote:")
    for p in paths:
        print("  " + p)


if __name__ == "__main__":
    main()
