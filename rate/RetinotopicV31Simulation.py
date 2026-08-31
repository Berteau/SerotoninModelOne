import numpy as np

from rate.RetinotopicGrandPlanSimulation import RetinotopicGrandPlanSimulation
from rate.Classifiers import make_classifier
from rate.ARTClassifier import REJECT

'''
v3.1 experiment: emergent option-B network + homeostatic controller, but driven
by a SHUFFLED SEQUENCE of multi-class images paired with loosely-correlated
discrete sound clips, and read out by a SWITCHABLE, slowly-learning unsupervised
classifier (FUZZY_ART / FUZZY_ARTMAP / KNN).

The scientific target is the emergence of NEW, audio-driven categories in the
classifier once audio comes to dominate the deprived cortex -- and the transient
hallucination (stale visual categories forced onto absent input) that precedes
it. The classifier is a pure READOUT here (no top-down feedback into V1); see
V3.1_Plan.md.

Two decoupled learning rates: remapLearningRate (audio->V1 plasticity, drives
recovery) and classifierLearningRate (slow category learning). Plasticity is on
from the sighted baseline so cross-modal drive can build; the classifier learns
slowly at every presentation.

Per-presentation records (one entry per image presentation, not per step):
  timeMs, assignedCluster, mappedLabel, trueClass, clipId, confidence,
  rateDeprived, rateIntact, homeostaticDrive, remapWeight, audioCurrent,
  visualCurrent, nClusters
plus classifier snapshots every snapshotEvery presentations.
'''

from rate.NormalizedMixtureNeuron import STREAM_TOPDOWN


class RetinotopicV31Simulation(RetinotopicGrandPlanSimulation):
    def __init__(self, params, presenter, mode="blind", plasticityEnabled=True, seed=0):
        p = dict(params)
        p["useART"] = False                      # classifier managed here, not in the network
        p["emergent5HT"] = True                  # reuse the emergent controller
        super().__init__(p, presenter.inputSet.stimuli[0], presenter.inputSet,
                         mode=mode, plasticityEnabled=plasticityEnabled)
        self.presenter = presenter
        self.seed = seed

        self.presentationMs = float(p.get("presentationMs", 100.0))
        self.baselineMs = float(p.get("baselineMs", 2000.0))
        self.postLossMs = float(p.get("postLossMs", 20000.0))
        self.snapshotEvery = int(p.get("snapshotEvery", 20))
        self._aEMA = None
        self._aEMAtau = float(p.get("homeostaticActivityTau", 500.0))

        # Switchable classifier over the V1 feature (length nCols).
        self.classifier = make_classifier(
            kind=p.get("classifierType", "FUZZY_ART"),
            dim=self.network.nCols,
            learningRate=p.get("classifierLearningRate", 0.05),
            vigilance=p.get("artVigilance", 0.75),
            activityFloor=p.get("artActivityFloor", 0.05),
            knnDistanceVigilance=p.get("knnDistanceVigilance", 0.30),
            seed=seed)

        # Records (per presentation).
        self.recTime = []
        self.assignedCluster = []
        self.mappedLabel = []
        self.trueClass = []
        self.clipIdRec = []
        self.confidence = []
        self.rateDeprivedSeq = []
        self.rateIntactSeq = []
        self.homeostaticDriveSeq = []
        self.remapWeightSeq = []
        self.audioCurrentSeq = []
        self.visualCurrentSeq = []
        self.nClustersSeq = []
        self.snapshots = []          # (presentationIndex, snapshot dict)
        self.lossPresentationIndex = None
        self.clipClassAssociation = None

    # --- controller reads an EMA of deprived activity (sequence smooths noise) ---
    def _deprivedActivity(self):
        inst = float(np.mean([c.rate for c in self.deprivedCells])) if self.deprivedCells else 0.0
        if self._aEMA is None:
            self._aEMA = inst
        else:
            dt = self._ctrlEvery * self.tau
            self._aEMA += (dt / self._aEMAtau) * (inst - self._aEMA)
        return self._aEMA

    # --- classifier pretraining on clean-V1 features (vision + canonical clip) ---
    def _pretrainClassifier(self, settleMs):
        steps = int(round(settleMs / self.tau))
        X, y = [], []
        for st in self.presenter.inputSet.stimuli:
            self.network.resetActivity()
            self.network.setVisualRates(self.presenter.inputSet.visualRates(st))
            clip = self.presenter.bank.canonical_clip(st.label)
            self.network.setAudioRates(self.presenter.audioRates(clip))
            for _ in range(steps):
                self.network.step()
            X.append(self.network.v1FeatureVector())
            y.append(st.label)
        self.classifier.pretrain(X, y)

    # --- one image presentation ---
    def _present(self, stimIndex, clipId, tMs, deprived, learn):
        self.network.setVisualRates(self.presenter.visualRates(stimIndex))
        if deprived:
            self.network.setVisualScotoma(self.silencedColumns)   # re-mask after setVisualRates
        self.network.setAudioRates(self.presenter.audioRates(clipId))
        steps = int(round(self.presentationMs / self.tau))
        for _ in range(steps):
            if self._controllerActive:
                self._ctrlCount += 1
                if self._ctrlCount % self._ctrlEvery == 0:
                    self._controllerStep()
            self.network.step()

        feat = self.network.v1FeatureVector()
        cluster, label, conf = self.classifier.observe(feat)
        if learn:
            self.classifier.learn(feat)

        sc = self.deprivedCells
        self.recTime.append(tMs)
        self.assignedCluster.append(cluster)
        self.mappedLabel.append(label)
        self.trueClass.append(self.presenter.true_label(stimIndex))
        self.clipIdRec.append(clipId)
        self.confidence.append(conf)
        self.rateDeprivedSeq.append(float(np.mean([c.rate for c in sc])) if sc else float("nan"))
        self.rateIntactSeq.append(float(np.mean([c.rate for c in self.intactCells])) if self.intactCells else float("nan"))
        self.homeostaticDriveSeq.append(self.homeostaticH)
        self.remapWeightSeq.append(float(np.mean([a.weight for a in self.plasticAxons])) if self.plasticAxons else float("nan"))
        self.audioCurrentSeq.append(float(np.mean([c.lastSourceCurrent.get(self._audioPop, 0.0) for c in sc])) if sc else float("nan"))
        self.visualCurrentSeq.append(float(np.mean([c.lastStreamCurrent[STREAM_TOPDOWN] for c in sc])) if sc else float("nan"))
        self.nClustersSeq.append(self.classifier.n_clusters)

    def run(self):
        p = self.params
        # Turn off the heavy per-cell history (unused; memory).
        self.network.setCellHistoryRecording(False)

        # Pretrain the classifier on clean V1, then warm up the network.
        self._pretrainClassifier(p.get("artPretrainSettleMs", 200.0))
        self.network.resetActivity()
        self._stepFor(self.warmupMs, record=False)

        # Plasticity ON from the sighted baseline (cross-modal drive can build).
        if self.plasticityEnabled:
            for a in self.plasticAxons:
                a.enablePlasticity(p["gamma_p"], p["gamma_d"], p["plasticityThreshold"],
                                   ceilingFactor=p["plasticityCeilingFactor"])

        nBaseline = int(round(self.baselineMs / self.presentationMs))
        nPost = int(round(self.postLossMs / self.presentationMs))
        nTotal = nBaseline + nPost
        seq = self.presenter.generate_sequence(nTotal, seed=self.seed)
        self.clipClassAssociation = self.presenter.measuredClipClassAssociation(seq)

        t = 0.0
        # ---- baseline (sighted) ----
        for i in range(nBaseline):
            si, clip = seq[i]
            self._present(si, clip, t, deprived=False, learn=True)
            if i % self.snapshotEvery == 0:
                self.snapshots.append((i, self.classifier.snapshot()))
            t += self.presentationMs
        # set-point from baseline deprived activity
        base = [r for r in self.rateDeprivedSeq if not np.isnan(r)]
        self.A_set = float(np.mean(base[len(base) // 2:])) if base else None

        # ---- impose loss; activate controller ----
        self.network.setVisualScotoma(self.silencedColumns)
        self._controllerActive = True
        self.lossPresentationIndex = nBaseline
        self._lossTimeMs = t

        # ---- post-loss (deprivation -> recovery -> long stable window) ----
        for i in range(nBaseline, nTotal):
            si, clip = seq[i]
            self._present(si, clip, t, deprived=True, learn=True)
            if i % self.snapshotEvery == 0:
                self.snapshots.append((i, self.classifier.snapshot()))
            t += self.presentationMs

        if self.plasticityEnabled:
            for a in self.plasticAxons:
                a.disablePlasticity()
        self.snapshots.append((nTotal - 1, self.classifier.snapshot()))
        return self
