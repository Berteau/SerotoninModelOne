import numpy as np

from rate.RetinotopicGrandPlanNetwork import RetinotopicGrandPlanNetwork, pretrainART
from rate.NormalizedMixtureNeuron import STREAM_TOPDOWN
from rate.ARTClassifier import FuzzyART, REJECT

'''
Sensory-loss experiment on the retinotopic grand-plan architecture, mirroring
the validated two-region sensory-loss protocol but on the Sulfaro-normalized V1.

Two loss modes:
  "scotoma"  a contiguous patch of visual columns is silenced (audio intact,
             the rest of the visual field intact);
  "blind"    every visual column is silenced (full visual loss, audio intact).

Four epochs (matching the two-region experiment):
  1 baseline          visual + audio on, serotonin baseline
  2 loss              masked visual columns -> 0 (stay 0 thereafter)
  3 serotonin + plast serotonin raised, plasticity ON for the cross-modal
                      audio -> V1 remapping synapses
  4 return            serotonin back to baseline, plasticity OFF

A no-plasticity control (plasticityEnabled=False) runs the identical protocol
with plasticity never switched on, isolating the causal role of the remapping
synapses in any recovery.

Readout population: because V1 receptive fields overlap (topographicRadius), a
silenced column at the edge of the lesion still receives visual current from its
intact neighbors, so it is not truly deprived. The primary readout is therefore
taken over FULLY-DEPRIVED columns -- silenced columns whose entire receptive
field (all columns within topographicRadius) is also silenced, so no visual
current leaks in. For full blindness every column is fully deprived; for a
scotoma it is the lesion interior. Intact = columns still receiving their own
visual input.

Recorded per step (means over the FULLY-DEPRIVED columns' V1 pyramidals unless noted):
  rateDeprived       V1 firing rate in fully-deprived columns
  rateIntact         V1 firing rate in intact columns
  audioCurrent       raw cross-modal (audio) drive into deprived-column V1
  visualCurrent      raw visual drive into deprived-column V1 (-> ~0 after loss)
  topDownCurrent     raw top-down (Category feedback) drive into deprived V1
  remapWeight        mean audio -> V1 remapping-synapse weight

In Sulfaro terms, a silenced column loses its bottom-up visual drive and becomes
top-down dominated (the hallucination-prone regime); serotonin-gated potentiation
of the cross-modal audio synapses restores an ascending (bottom-up) drive, and
the question is whether that restoration persists into epoch 4.
'''


class RetinotopicGrandPlanSimulation:
    def __init__(self, params, stimulus, inputSet, mode="scotoma",
                 plasticityEnabled=True):
        self.params = params
        self.tau = params["tau"]
        self.epochDurationMs = params["epochDurationMs"]
        self.warmupMs = params.get("warmupMs", 300.0)
        self.mode = mode
        self.plasticityEnabled = plasticityEnabled

        self.network = RetinotopicGrandPlanNetwork(self.tau, params)
        self.inputSet = inputSet
        self.stimulus = stimulus
        self.remapping = self.network.audioRemappingAxons()   # AudioInput -> V1pyr

        # If the secondary area is ART, build and pre-train the classifier on the
        # V1 representations of the labeled stimuli, then activate it. Done before
        # presenting the experiment stimulus so training sees clean bottom-up V1.
        self.art = None
        if self.network.useART:
            self.art = FuzzyART(dim=self.network.nCols, vigilance=params["artVigilance"],
                                alpha=0.01, beta=1.0, activity_floor=params["artActivityFloor"])
            pretrainART(self.network, inputSet, self.art,
                        settleMs=params.get("artPretrainSettleMs", 200.0))
            self.network.setARTActive(True)

        # Present the stimulus (held fixed across epochs; only the mask changes).
        self.network.resetActivity()
        self.network.setVisualRates(inputSet.visualRates(stimulus))
        self.network.setAudioRates(inputSet.audioRates(stimulus))

        # Column masks: silenced (visual set to 0) and fully-deprived (silenced
        # with the entire receptive field silenced, so no visual leaks in).
        self.silencedColumns = self._buildMask(mode)
        self.deprivedColumns = self._deprivedColumns(self.silencedColumns)
        self.deprivedCells, self.intactCells = self._partitionV1Cells()

        # Axons that undergo plasticity / cross-modal serotonin modulation. Under
        # Realistic-5HT the manipulation is confined to the DEPRIVED region, so we
        # use only the audio->V1 axons that target deprived cells (for full
        # blindness that is all of them); the legacy path uses all remapping axons.
        if params.get("realistic5HT", False):
            deprivedSet = set(self.deprivedCells)
            self.plasticAxons = [a for a in self.remapping if a.target in deprivedSet]
        else:
            self.plasticAxons = self.remapping

        # Retinotopic geometry for plotting.
        G = self.network.G
        self.inputGrid = np.asarray(stimulus.visualGrid, dtype=float)
        self.silencedMap = np.array(self.silencedColumns).reshape(G, G)
        self.deprivedMap = np.array(self.deprivedColumns).reshape(G, G)

        # Records
        self.epochBoundaries = []
        self.rateDeprived = []
        self.rateIntact = []
        self.audioCurrent = []
        self.visualCurrent = []
        self.topDownCurrent = []
        self.remapWeight = []
        self.v1MapsByEpoch = {}   # epoch index -> G x G V1 column-rate snapshot
        # ART decision time series (only populated when useART). artClass encodes
        # the decision numerically: the class label, or REJECT (-1) on reject.
        self.artClass = []
        self.artReject = []
        self.artMatch = []
        self.artCategory = []     # winning ART category index per step (None on reject)
        self.trueLabel = stimulus.label

        self._audioPop = self.network.populations["AudioInput"]
        self._visualPop = self.network.populations["VisualInput"]

    # ---- setup helpers ----

    def _buildMask(self, mode):
        G = self.network.G
        nCols = self.network.nCols
        mask = [False] * nCols
        if mode == "blind":
            return [True] * nCols
        # scotoma: silence a central patch large enough (ceil(2G/3)) that its
        # interior columns are fully deprived even with RF radius 1.
        h = max(1, -(-2 * G // 3))
        x0 = (G - h) // 2
        y0 = (G - h) // 2
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + h):
                mask[y * G + x] = True
        return mask

    def _deprivedColumns(self, silenced):
        # A silenced column is fully deprived iff every column within the V1
        # receptive-field radius is also silenced (no visual current leaks in).
        G = self.network.G
        radius = self.params["topographicRadius"]
        deprived = [False] * self.network.nCols
        for col in range(self.network.nCols):
            if not silenced[col]:
                continue
            cx, cy = col % G, col // G
            allSilenced = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < G and 0 <= ny < G:
                        if not silenced[ny * G + nx]:
                            allSilenced = False
                            break
                if not allSilenced:
                    break
            deprived[col] = allSilenced
        return deprived

    def _partitionV1Cells(self):
        cpc = self.network.cellsPerColumn
        deprived, intact = [], []
        for cell in self.network.populations["V1pyr"].cells:
            col = cell._idx // cpc
            if self.deprivedColumns[col]:
                deprived.append(cell)
            elif not self.silencedColumns[col]:
                intact.append(cell)
            # silenced-but-not-fully-deprived (penumbra) columns are excluded
            # from both readouts.
        return deprived, intact

    # ---- recording ----

    def _record(self):
        sc = self.deprivedCells
        self.rateDeprived.append(np.mean([c.rate for c in sc]) if sc else float("nan"))
        self.rateIntact.append(np.mean([c.rate for c in self.intactCells]) if self.intactCells else float("nan"))
        self.audioCurrent.append(np.mean([c.lastSourceCurrent.get(self._audioPop, 0.0) for c in sc]) if sc else float("nan"))
        self.visualCurrent.append(np.mean([c.lastSourceCurrent.get(self._visualPop, 0.0) for c in sc]) if sc else float("nan"))
        self.topDownCurrent.append(np.mean([c.lastStreamCurrent[STREAM_TOPDOWN] for c in sc]) if sc else float("nan"))
        self.remapWeight.append(float(np.mean([a.weight for a in self.plasticAxons])) if self.plasticAxons else float("nan"))
        if self.network.useART:
            d = self.network.lastART
            self.artReject.append(bool(d["reject"]))
            self.artClass.append(REJECT if d["reject"] else int(d["label"]))
            self.artMatch.append(float(d["match"]))
            self.artCategory.append(None if d["reject"] else int(d["category"]))

    # ---- run ----

    def _stepFor(self, durationMs, record=True):
        for _ in np.arange(0.0, durationMs, self.tau):
            self.network.step()
            if record:
                self._record()

    def _resetRecords(self):
        self.epochBoundaries = []
        for key in ("rateDeprived", "rateIntact", "audioCurrent",
                    "visualCurrent", "topDownCurrent", "remapWeight",
                    "artClass", "artReject", "artMatch", "artCategory"):
            setattr(self, key, [])

    def v1ColumnMap(self):
        # G x G map of per-column mean V1 pyramidal rate (retinotopic snapshot).
        G = self.network.G
        cpc = self.network.cellsPerColumn
        cells = self.network.populations["V1pyr"].cells
        m = np.array([np.mean([cells[c * cpc + k].rate for k in range(cpc)])
                      for c in range(G * G)])
        return m.reshape(G, G)

    def _endEpoch(self, epoch):
        self.epochBoundaries.append(self.epochDurationMs * epoch)
        # Retinotopic V1 snapshot at the end of this epoch, so we can show the
        # deprived region darken at loss and partially re-light after remapping.
        self.v1MapsByEpoch[epoch] = self.v1ColumnMap()

    def warmup(self):
        self._stepFor(self.warmupMs, record=False)
        self._resetRecords()

    def epoch1_baseline(self):
        self._stepFor(self.epochDurationMs)
        self._endEpoch(1)

    def epoch2_loss(self):
        self.network.setVisualScotoma(self.silencedColumns)
        self._stepFor(self.epochDurationMs)
        self._endEpoch(2)

    def epoch3_serotonin_plasticity(self):
        p = self.params
        if p.get("realistic5HT", False):
            # Realistic (5HT2A/3A-consistent) deafferentation response, confined
            # to the deprived region + its cross-modal input:
            #  (i)  reduce somatic 5HT in deprived visual cortex,
            #  (ii) raise V1 gain there (disinhibition),
            #  (iii)raise 5HT on the cross-modal audio->V1 transmission,
            #  (iv) lower the plasticity threshold.
            self.network.setV1SomaticSerotonin(p["v1SerotoninDeprived"], self.deprivedCells)
            self.network.setV1BottomUpGain(p["v1BottomUpGainDeprived"], self.deprivedCells)
            self.network.setCrossModalAxonalSerotonin(p["crossModalSerotoninLevel"], self.plasticAxons)
            plasticThreshold = p["plasticityThresholdDeprived"]
        else:
            self.network.setSerotonin(p["remapSerotoninLevel"])
            plasticThreshold = p["plasticityThreshold"]

        if self.plasticityEnabled:
            for axon in self.plasticAxons:
                axon.enablePlasticity(p["gamma_p"], p["gamma_d"], plasticThreshold,
                                      ceilingFactor=p["plasticityCeilingFactor"])
        self._stepFor(self.epochDurationMs)
        for axon in self.plasticAxons:
            axon.disablePlasticity()
        self._endEpoch(3)

    def epoch4_return(self):
        p = self.params
        if p.get("realistic5HT", False):
            # Restore the deprived region and its cross-modal input to baseline.
            self.network.setV1SomaticSerotonin(p["serotoninLevel"], self.deprivedCells)
            self.network.setV1BottomUpGain(1.0, self.deprivedCells)
            self.network.setCrossModalAxonalSerotonin(p["serotoninLevel"], self.plasticAxons)
        else:
            self.network.setSerotonin(p["serotoninLevel"])
        self._stepFor(self.epochDurationMs)
        self._endEpoch(4)

    def run(self):
        self.warmup()
        self.epoch1_baseline()
        self.epoch2_loss()
        self.epoch3_serotonin_plasticity()
        self.epoch4_return()
        return self
