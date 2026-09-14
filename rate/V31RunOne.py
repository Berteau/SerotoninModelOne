import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import random

import numpy as np

from rate.RetinotopicGrandPlanParams import buildGrandPlanParams
from rate.RetinotopicV31Simulation import RetinotopicV31Simulation
from rate.StructuredInput import (StructuredInputSet, ImageFolderInputSet,
                                  SequencePresenter, AudioStimulusBank,
                                  PairedSequencePresenter)

'''
Run ONE v3.1 simulation (one mode x one plasticity condition x one classifier)
and pickle its per-presentation records + classifier snapshots. The two learning
rates are set here at run time: --remap-lr (audio->V1 plasticity, gamma_p) and
--classifier-lr (slow category learning). Classifier is switchable via
--classifier {FUZZY_ART,FUZZY_ARTMAP,KNN}. See V3.1_Plan.md.

If --images is omitted, a synthetic >=3-class set is used (no image files needed)
so the pipeline is runnable/testable stand-alone.
'''

FIELDS = ["recTime", "assignedCluster", "mappedLabel", "trueClass", "clipIdRec",
          "confidence", "rateDeprivedSeq", "rateIntactSeq", "homeostaticDriveSeq",
          "remapWeightSeq", "audioCurrentSeq", "visualCurrentSeq", "nClustersSeq",
          "snapshots", "lossPresentationIndex", "A_set", "clipClassAssociation",
          "presentationMs", "baselineMs", "postLossMs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--visual", default=None, help="visual_stimuli root (per-class image subfolders); pair with --audio")
    ap.add_argument("--audio", default=None, help="audio_stimuli root (per-class .txt band-power subfolders); pair with --visual")
    ap.add_argument("--images", default=None, help="single image root + SYNTHETIC audio (legacy path); omit for synthetic")
    ap.add_argument("--audio-normalize", default="minmax", choices=["minmax", "peak", "none"])
    ap.add_argument("--mode", default="blind", choices=["blind", "scotoma"])
    ap.add_argument("--plasticity", type=int, default=1)
    ap.add_argument("--classifier", default="FUZZY_ART", choices=["FUZZY_ART", "FUZZY_ARTMAP", "KNN"])
    ap.add_argument("--classifier-lr", type=float, default=0.05)
    ap.add_argument("--remap-lr", type=float, default=None, help="gamma_p; default = params/10")
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--cpc", type=int, default=8)
    ap.add_argument("--classes", type=int, default=3, help="synthetic only")
    ap.add_argument("--images-per-class", type=int, default=3)
    ap.add_argument("--presentation-ms", type=float, default=100.0)
    ap.add_argument("--baseline-ms", type=float, default=2000.0)
    ap.add_argument("--postloss-ms", type=float, default=20000.0)
    ap.add_argument("--sound-bank-size", type=int, default=None)
    ap.add_argument("--match-prob", type=float, default=0.25)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    audioBank = None
    if bool(args.visual) != bool(args.audio):
        ap.error("--visual and --audio must be given together (real paired stimuli)")

    if args.visual and args.audio:
        # Real paired stimuli: images + real per-class audio band-power vectors.
        audioBank = AudioStimulusBank(args.audio, normalize=args.audio_normalize)
        data = ImageFolderInputSet(args.visual, gridSize=args.grid,
                                   audioBins=audioBank.audioBins,
                                   imagesPerClass=args.images_per_class, seed=args.seed)
        nClasses = len(data.classNames)
        audioBins = audioBank.audioBins
    elif args.images:
        data = ImageFolderInputSet(args.images, gridSize=args.grid, audioBins=20,
                                   imagesPerClass=args.images_per_class, seed=args.seed)
        nClasses = len(data.classNames)
        audioBins = 20
    else:
        data = StructuredInputSet(gridSize=args.grid, audioBins=20,
                                  classCount=args.classes, seed=args.seed)
        nClasses = args.classes
        audioBins = 20

    p = buildGrandPlanParams(gridSize=args.grid, cellsPerColumn=args.cpc, categoryCount=nClasses)
    p["audioBins"] = audioBins           # network AudioInput size must match the clips
    p["emergent5HT"] = True
    p["classifierType"] = args.classifier
    p["classifierLearningRate"] = args.classifier_lr
    p["gamma_p"] = args.remap_lr if args.remap_lr is not None else p["gamma_p"] / 10.0
    p["presentationMs"] = args.presentation_ms
    p["baselineMs"] = args.baseline_ms
    p["postLossMs"] = args.postloss_ms
    p["snapshotEvery"] = args.snapshot_every
    p["soundClassMatchProb"] = args.match_prob

    if audioBank is not None:
        presenter = PairedSequencePresenter(data, audioBank,
                                            matchProb=args.match_prob, seed=args.seed)
    else:
        presenter = SequencePresenter(data, soundBankSize=args.sound_bank_size,
                                      matchProb=args.match_prob, seed=args.seed)

    sim = RetinotopicV31Simulation(p, presenter, mode=args.mode,
                                   plasticityEnabled=bool(args.plasticity), seed=args.seed)
    sim.run()

    blob = {f: getattr(sim, f) for f in FIELDS}
    blob["params"] = p
    blob["classNames"] = getattr(data, "classNames", [str(i) for i in range(nClasses)])
    blob["mode"] = args.mode
    blob["plasticity"] = bool(args.plasticity)
    blob["classifier"] = args.classifier
    blob["audioBins"] = audioBins
    blob["soundBankSize"] = (presenter.bank.size if hasattr(presenter, "bank")
                             else len(audioBank.flat))
    blob["audioSource"] = "real" if audioBank is not None else "synthetic"
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh)
    n = len(sim.assignedCluster)
    print("wrote %s (classifier=%s mode=%s plast=%s): %d presentations, "
          "final n_clusters=%d, clip-class assoc=%.3f"
          % (args.out, args.classifier, args.mode, bool(args.plasticity), n,
             sim.nClustersSeq[-1] if sim.nClustersSeq else -1,
             sim.clipClassAssociation if sim.clipClassAssociation is not None else float("nan")))


if __name__ == "__main__":
    main()
