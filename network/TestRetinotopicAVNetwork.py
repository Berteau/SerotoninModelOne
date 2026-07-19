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
from network.RetinotopicAVParams import buildDefaultParams

'''
Self-verifying test for RetinotopicAVNetwork / RetinotopicAVSimulation.

Mirrors the phase progression demonstrated by TestTwoColumnNetworkManyTimes /
TwoColumnSimulation (full input -> sensory loss -> serotonin rise with
plasticity -> remapped, stable state), but on the 100-column retinotopic /
20-column tonotopic network, driven by a real (synthetic) greyscale video and
beep score. It runs both required scenarios - a partial scotoma and a
full-field visual loss, with audio left active throughout both.

For each scenario it runs two conditions from the same random seed (so the
network topology and initial state are identical): the real remapping
condition, and a control where phase3/phase4 still run (same duration, same
muting) but serotonin never rises and plasticity never switches on. This
network has enough recurrent excitation that overall activity drifts upward
over time on its own (visible even within phase1, before any lesion or
plasticity change) - so the pass criteria below don't just check that the
deprived column's activity went up, they check that (a) the remapping
synapses' weight only moves in the real condition, and (b) the real
condition's recovery specifically outstrips the control's, isolating the
effect of remapping from generic drift.

This is a runnable script rather than a pytest suite, consistent with the
rest of this repository's Test*.py files. It exits with status 0 on success
and 1 if any assertion fails.
'''

SEED = 1234

buildParams = buildDefaultParams

def runScenario(lesionMode, videoPath, beepPath, params, enableRemapping):
    random.seed(SEED)
    np.random.seed(SEED)
    video = VideoFrameSource(videoPath, gridRows=params["gridRows"], gridCols=params["gridCols"])
    audio = BeepAudioSource(beepPath, numBands=params["numAudioColumns"])
    sim = RetinotopicAVSimulation(params, video, audio, lesionMode=lesionMode, enableRemapping=enableRemapping)
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

def checkControlHasNoWeightChange(lesionMode, controlSim, failures):
    # Without ever setting plasticity=True, boutonSpike()/postSynapticSpikeFeedback()
    # can't touch weight at all - this should hold exactly, and guards against
    # a regression that enables plasticity unconditionally somewhere.
    if controlSim.weightsPrior != controlSim.weightsPost:
        failures.append("[%s control] Remapping weight changed even though plasticity was never enabled" % lesionMode)

def checkRemappingOutstripsDrift(lesionMode, experimentalSim, controlSim, failures):
    # The real causal claim: with the same network, same stimuli, and the
    # same generic activity drift in both conditions, the lesioned site
    # should recover further with remapping enabled than without it.
    lesionedCoord = experimentalSim.lesionedColumns[0]
    experimentalActivity = experimentalSim.activitySnapshots["4_remapped"][lesionedCoord]
    controlActivity = controlSim.activitySnapshots["4_remapped"][lesionedCoord]

    print("[%s] lesioned-site phase4 activity: remapping=%.4f control (drift only)=%.4f" %
          (lesionMode, experimentalActivity, controlActivity))

    if experimentalActivity <= controlActivity:
        failures.append(
            "[%s] Remapping condition's recovered activity (%.4f) did not exceed the no-plasticity control's (%.4f) - "
            "the apparent recovery may just be generic network drift, not remapping" %
            (lesionMode, experimentalActivity, controlActivity))

def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmpDir:
        videoPath = writeSyntheticGreyscaleVideo(os.path.join(tmpDir, "test_video.mp4"))
        beepPath = writeSyntheticBeepScore(os.path.join(tmpDir, "test_beep.csv"))
        params = buildParams()

        for lesionMode in ("scotoma", "full"):
            print("Running %s scenario (remapping)..." % lesionMode)
            experimentalSim = runScenario(lesionMode, videoPath, beepPath, params, enableRemapping=True)
            checkNetworkShape(experimentalSim, failures)
            checkRemapping(lesionMode, experimentalSim, failures)

            print("Running %s scenario (control, no plasticity)..." % lesionMode)
            controlSim = runScenario(lesionMode, videoPath, beepPath, params, enableRemapping=False)
            checkControlHasNoWeightChange(lesionMode, controlSim, failures)

            checkRemappingOutstripsDrift(lesionMode, experimentalSim, controlSim, failures)

    if failures:
        print("FAILED:")
        for failure in failures:
            print("  - " + failure)
        sys.exit(1)
    else:
        print("PASSED: remapping-specific recovery (beyond generic drift) occurred in both scenarios.")

if __name__ == "__main__":
    main()
