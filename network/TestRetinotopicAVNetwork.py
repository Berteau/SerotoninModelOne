import sys
import random
import tempfile
import os

# Allow running this script directly (python3 network/TestRetinotopicAVNetwork.py)
# regardless of the current working directory, since Python only puts the
# script's own directory on sys.path, not the repo root the other packages
# (sensory, simulation, ...) live under.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)

from sensory.SyntheticFixtures import writeSyntheticGreyscaleVideo, writeSyntheticBeepScore
from sensory.VideoFrameSource import VideoFrameSource
from sensory.BeepAudioSource import BeepAudioSource
from simulation.RetinotopicAVSimulation import RetinotopicAVSimulation

'''
Self-verifying test for RetinotopicAVNetwork / RetinotopicAVSimulation.

Mirrors the phase progression demonstrated by TestTwoColumnNetworkManyTimes /
TwoColumnSimulation (full input -> sensory loss -> serotonin rise with
plasticity -> remapped, stable state), but on the 100-column retinotopic /
20-column tonotopic network, driven by a real (synthetic) greyscale video and
beep score. It runs both required scenarios - a partial scotoma and a
full-field visual loss, with audio left active throughout both - and asserts
that the relevant "remapping" synapses actually potentiated.

This is a runnable script rather than a pytest suite, consistent with the
rest of this repository's Test*.py files. It exits with status 0 on success
and 1 if any assertion fails.
'''

random.seed(1234)
np.random.seed(1234)

def buildParams():
    params = {}
    params["popCount"] = 8
    params["gridRows"] = 10
    params["gridCols"] = 10
    params["numAudioColumns"] = 20
    params["tau"] = 0.1
    params["frameDurationMs"] = 5
    params["phaseDurationMs"] = 20

    params["serotoninLevelV"] = 10
    params["serotoninLevelA"] = 10
    params["remapSerotoninLevel"] = 30
    params["remapPlasticityCp"] = 90

    # The pooled connection weights below (pyramidalSelfExcitationWeight etc.)
    # are divided by popCount to get a per-synapse weight (see
    # RetinotopicAVNetwork.gaussWeightWithChanceFactory). SerotoninAVNetwork's
    # values were tuned for popCount=40; reusing them unscaled at a much
    # smaller popCount makes each synapse disproportionately strong and drives
    # the network into pathological, near-saturated firing. popCount/40 alone
    # (0.2 here) overcorrects and leaves the network too quiet to fire once
    # lesioned columns lose their own input, so this uses an empirically
    # tuned factor that keeps activity in a healthy middle ground.
    weightScale = 0.4

    params["baselineRateV"] = 5
    params["baselineRateA"] = 5
    params["inputWeightV"] = 1500 * weightScale
    params["inputWeightA"] = 1500 * weightScale
    params["inputWeightVA"] = 800 * weightScale
    params["inputWeightAV"] = 800 * weightScale
    params["crossModalLikelihood"] = 0.3
    params["lateralVisualWeight"] = 1500 * weightScale
    params["lateralVisualLikelihood"] = 0.3
    params["lateralAudioWeight"] = 1500 * weightScale
    params["lateralAudioLikelihood"] = 0.3

    params["Somatic5HT2AWeight"] = 20
    params["Somatic5HT1AWeight"] = -5
    params["Axonal5HT2AWeight"] = 0.2
    params["Axonal5HT1AWeight"] = -0.2
    params["pyramidalSelfExcitationWeight"] = 15000 * weightScale
    params["PyramidalsToFSWeight"] = 50000 * weightScale
    params["FSToPyramidalsWeight"] = -30000 * weightScale
    params["PyramidalsToLTSWeight"] = 80000 * weightScale
    return params

def runScenario(lesionMode, videoPath, beepPath, params):
    video = VideoFrameSource(videoPath, gridRows=params["gridRows"], gridCols=params["gridCols"])
    audio = BeepAudioSource(beepPath, numBands=params["numAudioColumns"])
    sim = RetinotopicAVSimulation(params, video, audio, lesionMode=lesionMode)
    sim.run()
    return sim

def checkNetworkShape(sim, failures):
    if len(sim.network.visualColumnNames) != 100:
        failures.append("Expected 100 visual columns (10x10), got %d" % len(sim.network.visualColumnNames))
    if len(sim.network.audioColumnNames) != 20:
        failures.append("Expected 20 tonotopic audio columns, got %d" % len(sim.network.audioColumnNames))

def checkRemapping(lesionMode, sim, failures):
    if len(sim.remappingAxons) == 0:
        failures.append("[%s] No remapping axons were found to potentiate" % lesionMode)
        return

    priorMean = float(np.mean(sim.weightsPrior))
    postMean = float(np.mean(sim.weightsPost))
    increasedCount = sum(1 for prior, post in zip(sim.weightsPrior, sim.weightsPost) if post > prior)
    fraction = increasedCount / len(sim.remappingAxons)

    print("[%s] remapping synapses: %d, mean weight prior=%.4f post=%.4f, fraction increased=%.2f" %
          (lesionMode, len(sim.remappingAxons), priorMean, postMean, fraction))

    # LTD (a small constant decay) fires on every presynaptic spike, while LTP
    # (the bigger, timing-dependent term) only rewards synapses whose spikes
    # closely preceded a postsynaptic spike. So remapping shows up as a net
    # rise in mean weight, carried by a subset of well-timed synapses - not as
    # a majority of individual synapses increasing.
    if postMean <= priorMean * 1.001:
        failures.append("[%s] Mean remapping weight did not increase (prior=%.4f, post=%.4f)" % (lesionMode, priorMean, postMean))
    if fraction <= 0:
        failures.append("[%s] No individual remapping synapse potentiated" % lesionMode)

def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmpDir:
        videoPath = writeSyntheticGreyscaleVideo(os.path.join(tmpDir, "test_video.mp4"))
        beepPath = writeSyntheticBeepScore(os.path.join(tmpDir, "test_beep.csv"))
        params = buildParams()

        for lesionMode in ("scotoma", "full"):
            print("Running %s scenario..." % lesionMode)
            sim = runScenario(lesionMode, videoPath, beepPath, params)
            checkNetworkShape(sim, failures)
            checkRemapping(lesionMode, sim, failures)

    if failures:
        print("FAILED:")
        for failure in failures:
            print("  - " + failure)
        sys.exit(1)
    else:
        print("PASSED: retinotopic/tonotopic remapping occurred in both scenarios.")

if __name__ == "__main__":
    main()
