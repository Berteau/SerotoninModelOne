import dill
import os
import sys

from absl import app
from absl import flags
from absl import logging 
from copy import deepcopy
from params import loadParams
from simulation.ScalingTestSimulation import ScalingTestSimulation
from simulation.ScalingTestSimulation import ScalingTestSimulationPlotter

sys.setrecursionlimit(10000)

FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "PLOT", "Mode to execute. PLOT to run and plot an output, wihout saving an intermediate simulation. RUN for running the simulation & saving a pickled output, WRITE to read a pickled output and generate figures.")
flags.
flags.DEFINE_string("picklejar", "scalingSimulation.pickle", "Filename to write simulation to or read simulation from.")

def runSim():
    logging.info("Running a new ScalingTestSimulation pass")
    params = loadParams()
    sim = ScalingTestSimulation(params)

    sim.run()

    return sim

def writeSim(sim):
    with open(FLAGS.picklejar, "wb") as pickleJar:
        logging.info("Writing a TwoColumnSimulation pass to %s" % FLAGS.picklejar)
        dill.dump(deepcopy(sim), pickleJar)

def readSim():
    with open(FLAGS.picklejar, "rb") as pickleJar:
        logging.info("Reading a TwoColumnSimulation pass from %s" % FLAGS.picklejar)
        sim = dill.load(pickleJar)
        return sim

def plotSim(sim):
    logging.info("Generating plots for TwoColumnSimulation")
    ScalingTestSimulationPlotter(sim).plotColumns()

def main(argv):
    if FLAGS.mode == "PLOT":
        sim = runSim()
        plotSim(sim)
    if FLAGS.mode == "RUN":
        sim = runSim()
        writeSim(sim)
    if FLAGS.mode == "WRITE":
        sim = readSim()
        plotSim(sim)

if __name__ == '__main__':
    app.run(main)
