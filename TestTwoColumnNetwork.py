import dill
import sys
from absl import app
from absl import flags
from absl import logging
from copy import deepcopy

from TwoColumnSimulation import *

sys.setrecursionlimit(10000)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mode",
    "PLOT",
    "Mode to execute. PLOT to run and plot an output, wihout saving an intermediate simulation. RUN for running the simulation & saving a pickled output, WRITE to read a pickled output and generate figures.")
flags.DEFINE_string(
    "picklejar",
    "two_column_simulation.pickle",
    "Filename to write simulation to or read simulation from.")


def loadParams():
    params = {}
    # Time Parameters
    params["maxTime"] = 100  # 1000
    params["tau"] = 0.1

    # Population and Connection Parameters
    params["popCount"] = 20

    params["pyramidalSelfExcitationWeight"] = 25
    params["pyramidalToPyramidalWeight"] = 50
    params["pyramidalToPyramidalLikelihood"] = 0.3
    params["PyramidalsToFSWeight"] = 35
    params["FSToPyramidalsWeight"] = -100
    params["PyramidalsToLTSWeight"] = 20  # 500
    params["LTStoFSWeight"] = -50
    params["LTStoPyramidalsWeight"] = -50

    # Regular
    # Input Paramters
    params["inputWeightA"] = 50
    params["inputWeightB"] = 50
    params["rateA"] = 1
    params["rateB"] = 1
    params["inputWeightAB"] = 25
    params["crossModalABLikelihood"] = 0.5
    params["inputWeightBA"] = 25  # 7500
    params["crossModalBALikelihood"] = 0.5

    # Diffuse Neurotransmitter Paramters
    params["serotoninLevelA"] = 10
    params["serotoninLevelB"] = 10
    params["Somatic5HT2AWeight"] = 85
    params["Somatic5HT2AWeightLTS"] = 81
    params["Somatic5HT1AWeight"] = -80
    params["Axonal5HT2AWeight"] = 0.4
    params["Axonal5HT1AWeight"] = -0.4

    return params


def runSim():
    logging.info("Running a new TwoColumnSimulation pass")
    params = loadParams()
    sim = TwoColumnSimulation(params)

    sim.run()

    return sim


def writeSim(sim):
    with open(FLAGS.picklejar, "wb") as pickleJar:
        logging.info(
            "Writing a TwoColumnSimulation pass to %s" %
            FLAGS.picklejar)
        dill.dump(deepcopy(sim), pickleJar)


def readSim():
    with open(FLAGS.picklejar, "rb") as pickleJar:
        logging.info(
            "Reading a TwoColumnSimulation pass from %s" %
            FLAGS.picklejar)
        sim = dill.load(pickleJar)
        return sim


def plotSim(sim):
    logging.info("Generating plots for TwoColumnSimulation")
    sim.plotColumns()


def main(argv):
    if FLAGS.mode == "PLOT":
        sim = runSim()
        plotSim(sim)
    if FLAGS.mode == "RUN":
        sim = runSim()
        writeSim(wim)
    if FLAGS.mode == "WRITE":
        sim = readSim()
        plotSim(sim)


if __name__ == '__main__':
    app.run(main)
