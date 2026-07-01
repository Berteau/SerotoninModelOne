import sys
import os

# Allow running this script directly (python3 network/DemoRetinotopicAVNetwork.py)
# regardless of the current working directory - see TestRetinotopicAVNetwork.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from absl import flags
import matplotlib
matplotlib.use("Agg")  # write figures to disk without needing a display
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("lesion_mode", "scotoma", "Which lesion scenario to run: 'scotoma' or 'full'.")
flags.DEFINE_string("video_path", None, "Greyscale video file to use as visual input. If unset, a small synthetic clip is generated and saved alongside the figures.")
flags.DEFINE_string("beep_path", None, "Beep score CSV (time_ms,frequency_hz,amplitude) to use as audio input. If unset, a small synthetic score is generated and saved alongside the figures.")
flags.DEFINE_string("figures_directory", os.path.join(os.curdir, "figures"), "Directory to write input fixtures and output plots into.")
FLAGS(sys.argv)

from network.RetinotopicAVParams import buildDefaultParams
from sensory.SyntheticFixtures import writeSyntheticGreyscaleVideo, writeSyntheticBeepScore
from sensory.VideoFrameSource import VideoFrameSource
from sensory.BeepAudioSource import BeepAudioSource
from simulation.RetinotopicAVSimulation import RetinotopicAVSimulation

'''
Runs one RetinotopicAVSimulation scenario end to end and writes out:
  - the video/beep input files actually used (synthetic ones are generated
    here instead of a throwaway temp dir, so they can be inspected/replayed)
  - weight_over_time.png: mean weight of the "remapping" synapses across all
    four phases, with phase boundaries marked
  - activity_traces.png: firing rate over time for a few representative
    columns (the lesioned site, and either a spared neighbor or the
    cross-modal audio pathway, depending on lesion mode)
  - activity_snapshots.png: a 1x4 spatial map of visual-grid activity at the
    end of each phase, showing the scotoma appear and then refill

Usage: python3 network/DemoRetinotopicAVNetwork.py [--lesion_mode=scotoma|full]
'''

def phaseBoundariesMs(sim):
    d = sim.params["phaseDurationMs"]
    return [d, 2 * d, 3 * d]

def selectExampleColumns(sim):
    network = sim.network
    examples = [("lesioned %s" % str(sim.lesionedColumns[0]), "pyramidals" + network.visualColumnNames[sim.lesionedColumns[0]])]
    if sim.lesionMode == "scotoma":
        examples.append(("spared neighbor %s" % str(sim.spareNeighborColumns[0]), "pyramidals" + network.visualColumnNames[sim.spareNeighborColumns[0]]))
        farColumn = (0, 0)
        examples.append(("driven corner %s" % str(farColumn), "pyramidals" + network.visualColumnNames[farColumn]))
    else:
        edgeColumn = (0, 0)
        examples.append(("lesioned edge (cross-modal) %s" % str(edgeColumn), "pyramidals" + network.visualColumnNames[edgeColumn]))
        examples.append(("audio band 0", "pyramidals" + network.audioColumnNames[0]))
    return examples

def plotWeightHistory(sim, outputDir):
    times = [t for t, _ in sim.weightHistory]
    weights = [w for _, w in sim.weightHistory]
    plt.figure()
    plt.plot(times, weights)
    for boundary in phaseBoundariesMs(sim):
        plt.axvline(boundary, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean remapping-synapse weight")
    plt.title("Remapping synapse weight over time (%s)" % sim.lesionMode)
    path = os.path.join(outputDir, "weight_over_time.png")
    plt.savefig(path)
    plt.close()
    return path

def plotActivityTraces(sim, outputDir):
    plt.figure()
    for label, popName in selectExampleColumns(sim):
        rateRecord = sim.network.populations[popName].rateRecord
        times = [i * sim.tau for i in range(len(rateRecord))]
        plt.plot(times, rateRecord, label=label)
    for boundary in phaseBoundariesMs(sim):
        plt.axvline(boundary, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (spikes/sec, trailing 50ms window)")
    plt.title("Pyramidal activity traces (%s)" % sim.lesionMode)
    plt.legend(fontsize="small")
    path = os.path.join(outputDir, "activity_traces.png")
    plt.savefig(path)
    plt.close()
    return path

def plotActivitySnapshots(sim, outputDir):
    names = list(sim.activitySnapshots.keys())
    grids = list(sim.activitySnapshots.values())
    vmax = max(grid.max() for grid in grids) if grids else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, axes = plt.subplots(1, len(grids), figsize=(4 * len(grids), 4))
    if len(grids) == 1:
        axes = [axes]
    im = None
    for ax, name, grid in zip(axes, names, grids):
        im = ax.pcolor(grid, vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.invert_yaxis()
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.suptitle("Visual grid activity by phase (%s)" % sim.lesionMode)
    path = os.path.join(outputDir, "activity_snapshots.png")
    plt.savefig(path)
    plt.close()
    return path

def main():
    outputDir = os.path.join(FLAGS.figures_directory, "retinotopic_demo", datetime.utcnow().isoformat())
    os.makedirs(outputDir, exist_ok=True)

    videoPath = FLAGS.video_path or writeSyntheticGreyscaleVideo(os.path.join(outputDir, "input_video.mp4"))
    beepPath = FLAGS.beep_path or writeSyntheticBeepScore(os.path.join(outputDir, "input_beep.csv"))
    print("Visual input file: %s" % videoPath)
    print("Audio input file:  %s" % beepPath)

    params = buildDefaultParams()
    video = VideoFrameSource(videoPath, gridRows=params["gridRows"], gridCols=params["gridCols"])
    audio = BeepAudioSource(beepPath, numBands=params["numAudioColumns"])
    sim = RetinotopicAVSimulation(params, video, audio, lesionMode=FLAGS.lesion_mode)

    print("Running '%s' scenario (popCount=%d, phaseDurationMs=%d)..." %
          (FLAGS.lesion_mode, params["popCount"], params["phaseDurationMs"]))
    sim.run()

    weightPath = plotWeightHistory(sim, outputDir)
    tracePath = plotActivityTraces(sim, outputDir)
    snapshotPath = plotActivitySnapshots(sim, outputDir)

    print("Wrote:")
    print("  " + weightPath)
    print("  " + tracePath)
    print("  " + snapshotPath)

if __name__ == "__main__":
    main()
