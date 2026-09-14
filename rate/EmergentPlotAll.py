import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import types

import numpy as np

from rate.plotting_grandplan import plot_experiment, epoch_means

'''
Load the four EmergentRunOne pickles (scotoma/blind x plasticity/control),
pair each mode's plasticity run with its no-plasticity control, and regenerate
the emergent figures (rates / streams / remap_weight / v1_maps / homeostatic)
plus a compact per-epoch text summary. ART panels are skipped automatically:
these runs carry no artMatch (classifier kept separate).

Usage: python3 rate/EmergentPlotAll.py <pkl_dir> <out_dir>
  expects pkl_dir/{scotoma,blind}_{plast,ctrl}.pkl
'''


def load(path):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    sim = types.SimpleNamespace(**blob)
    # plot_art reads sim.inputSet.classNames; give it a light shim.
    sim.inputSet = types.SimpleNamespace(classNames=blob.get("classNames"))
    return sim


def main():
    pkl_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    for mode in ("scotoma", "blind"):
        plast_path = os.path.join(pkl_dir, "%s_plast.pkl" % mode)
        ctrl_path = os.path.join(pkl_dir, "%s_ctrl.pkl" % mode)
        if not os.path.exists(plast_path):
            print("missing %s -- skipping mode %s" % (plast_path, mode))
            continue
        sim = load(plast_path)
        control = load(ctrl_path) if os.path.exists(ctrl_path) else None
        title = "Emergent 5HT (%s)" % mode
        outdir = os.path.join(out_dir, mode)
        paths = plot_experiment(sim, control, outdir, title)
        print("\n=== %s ===" % title)
        for p in paths:
            print("  wrote", p)

        # compact per-epoch text summary
        em = epoch_means(sim)
        print("  per-epoch means (plasticity run):")
        print("    %-14s %8s %8s %8s %8s" %
              ("field", "baseline", "loss", "epoch3", "return"))
        for name in ("rateDeprived", "audioCurrent", "topDownCurrent", "remapWeight"):
            vals = em[name]
            print("    %-14s %8.2f %8.2f %8.2f %8.2f" % (name, *vals))
        h = np.asarray(sim.homeostaticDrive, dtype=float)
        rate = np.asarray(sim.rateDeprived, dtype=float)
        if sim.A_set:
            post = rate[len(rate) // 4:]  # after baseline epoch
            print("    A_set=%.2f  trough=%.2f  peak=%.2f  final=%.2f  h_peak=%.2f h_final=%.2f  W_final=%.1f"
                  % (sim.A_set, float(np.nanmin(post)), float(np.nanmax(post)),
                     float(rate[-1]), float(np.nanmax(h)), float(h[-1]),
                     float(sim.remapWeight[-1])))


if __name__ == "__main__":
    main()
