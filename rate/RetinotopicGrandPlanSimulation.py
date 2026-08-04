import numpy as np

from rate.RetinotopicGrandPlanNetwork import RetinotopicGrandPlanNetwork
from rate.NormalizedMixtureNeuron import STREAM_TOPDOWN

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

        # Present the stimulus (held fixed across epochs; only the mask changes).
        self.network.setVisualRates(inputSet.visualRates(stimulus))
        self.network.setAudioRates(inputSet.audioRates(stimulus))

        # Column masks: silenced (visual set to 0) and fully-deprived (silenced
        # with the entire receptive field silenced, so no visual leaks in).
        self.silencedColumns = self._buildMask(mode)
        self.deprivedColumns = self._deprivedColumns(self.silencedColumns)
        self.deprivedCells, self.intactCells = self._partitionV1Cells()

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
        self.remapWeight.append(float(np.mean([a.weight for a in self.remapping])) if self.remapping else float("nan"))

    # ---- run ----

    def _stepFor(self, durationMs, record=True):
        for _ in np.arange(0.0, durationMs, self.tau):
            self.network.step()
            if record:
                self._record()

    def _resetRecords(self):
        self.epochBoundaries = []
        for key in ("rateDeprived", "rateIntact", "audioCurrent",
                    "visualCurrent", "topDownCurrent", "remapWeight"):
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
        self.network.setSerotonin(self.params["remapSerotoninLevel"])
        if self.plasticityEnabled:
            for axon in self.remapping:
                axon.enablePlasticity(self.params["gamma_p"], self.params["gamma_d"],
                                      self.params["plasticityThreshold"],
                                      ceilingFactor=self.params["plasticityCeilingFactor"])
        self._stepFor(self.epochDurationMs)
        for axon in self.remapping:
            axon.disablePlasticity()
        self._endEpoch(3)

    def epoch4_return(self):
        self.network.setSerotonin(self.params["serotoninLevel"])
        self._stepFor(self.epochDurationMs)
        self._endEpoch(4)

    def run(self):
        self.warmup()
        self.epoch1_baseline()
        self.epoch2_loss()
        self.epoch3_serotonin_plasticity()
        self.epoch4_return()
        return self
