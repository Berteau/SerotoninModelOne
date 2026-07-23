from random import random, gauss

from rate.RatePopulation import RatePopulation
from rate.NormalizedMixtureNeuron import (
    NormalizedMixtureNeuron, STREAM_BOTTOMUP, STREAM_TOPDOWN, STREAM_SELF)
from rate.RateDiffuseReceptor import RateSomaticReceptorFactory, RateAxonalReceptorFactory

'''
Retinotopic grand-plan architecture (rate-based), with the primary visual area
(V1) implementing the Sulfaro et al. (2023) normalized mixture of top-down and
bottom-up input at the current level (see NormalizedMixtureNeuron).

Areas
-----
  VisualInput   G*G retinotopic input cells (one per edge-detected pixel).
  AudioInput    A tonotopic input cells, r=0.25 correlated to the visual input
                (StructuredInput). Projects cross-modally to V1 -- these are the
                remapping synapses that potentiate under serotonin when vision
                is lost, exactly as in the validated two-region model.
  V1pyr         G*G*cellsPerColumn NormalizedMixtureNeuron pyramidals. Each
                cell's input current is the weight-normalized convex combination
                of its three streams:
                  bottom-up : VisualInput (topographic) + AudioInput (cross-modal)
                  self      : V1 recurrent excitation + local FS inhibition
                  top-down  : Category feedback (the secondary area)
  V1fs          G*G*fsPerColumn fast-spiking interneurons (local E/I loop).
  Category      K category cells forming the SECONDARY area: a competitive
                one-hot classifier with a reject/vigilance option (ART-style,
                first pass -- see below). Its winning category feeds back to V1
                as the top-down stream.

Streams and the Sulfaro knob
----------------------------
Each V1 pyramidal tags its inbound source populations into the three Sulfaro
streams via assignStream(). The feedforward:feedback weighting ratio
w_bottomup / w_topdown (params wBottomUp / wTopDown) is the perception knob:
>1 veridical, <1 hallucination-prone (Sulfaro Fig. 3). Serotonergic 5HT2A gain
on the bottom-up weight (setV1BottomUpGain) is Sulfaro's neurochemical bridge
(5HT2A dampens V1 bottom-up gain; Seillier 2017, Michaiel 2019).

Secondary classifier (first pass)
---------------------------------
The Category cells each hold a learned/handset template over V1 activity and
compete by mutual inhibition; a category "wins" only if its match exceeds a
vigilance threshold, otherwise the input is REJECTED (novel / potentially
hallucinated). This first pass uses fixed templates (one per class prototype) so
the top-down stream is well-defined and the V1 normalization can be exercised
end to end; the ART learning/reset dynamics and the person/horse label wiring
are deliberately left as a follow-up (they carry design choices -- number of
categories, vigilance, template learning rule -- best set with the user).
'''


class RetinotopicGrandPlanNetwork:
    def __init__(self, tau, params, name="RetinotopicGrandPlan"):
        self.tau = tau
        self.params = params
        self.name = name
        self.G = params["gridSize"]
        self.nCols = self.G * self.G
        self.cellsPerColumn = params["cellsPerColumn"]
        self.fsPerColumn = params["fsPerColumn"]
        self.audioBins = params["audioBins"]
        self.categoryCount = params["categoryCount"]
        self.populations = {}

        transmitters = {"5HT2A": params["serotoninLevel"], "5HT1A": params["serotoninLevel"]}
        somaP = [RateSomaticReceptorFactory("5HT2A", params["Somatic5HT2AWeight"]),
                 RateSomaticReceptorFactory("5HT1A", params["Somatic5HT1AWeight"])]
        axon5HT2A = [RateAxonalReceptorFactory("5HT2A", params["Axonal5HT2AWeight"])]
        self._axon5HT2A = axon5HT2A

        p = self.populations

        # ---- input generators ----
        p["VisualInput"] = RatePopulation(tau, None, self.nCols, [], {}, self, "VisualInput",
                                          isInput=True, inputRate=0.0)
        p["AudioInput"] = RatePopulation(tau, None, self.audioBins, [], {}, self, "AudioInput",
                                         isInput=True, inputRate=0.0)

        # ---- V1 area (mixture neurons) ----
        wB, wT, wS = params["wBottomUp"], params["wTopDown"], params["wSelf"]

        def mixtureFactory(tau_, cellType, nm, parentPop):
            return NormalizedMixtureNeuron(tau_, cellType, nm, parentPop=parentPop,
                                           w_bottomup=wB, w_topdown=wT, w_self=wS)

        p["V1pyr"] = RatePopulation(tau, "Pyramidal", self.nCols * self.cellsPerColumn,
                                    somaP, transmitters, self, "V1pyr",
                                    neuronFactory=mixtureFactory)
        p["V1fs"] = RatePopulation(tau, "FS", self.nCols * self.fsPerColumn,
                                   [], transmitters, self, "V1fs")
        # The leave-one-out ablation influence metric assumes additive currents;
        # under the normalized mixture the marginal of a source is not its raw
        # current, so we do not report that metric for V1 (see NormalizedMixtureNeuron).
        p["V1pyr"].trackInfluence = False

        # ---- secondary area: competitive classifier ----
        p["Category"] = RatePopulation(tau, "Pyramidal", self.categoryCount,
                                       somaP, transmitters, self, "Category")

        # Cache each cell's index within its population so the connectivity weight
        # functions are O(1) rather than O(n) (cells.index) inside the O(n^2) build.
        for pop in p.values():
            for i, cell in enumerate(pop.cells):
                cell._idx = i

        self._buildConnectivity(params)
        self._assignV1Streams()

    # ---- column index helpers ----

    def _colOf(self, cellIndex, perColumn):
        return cellIndex // perColumn

    def _colXY(self, col):
        return col % self.G, col // self.G

    # ---- connectivity ----

    def _buildConnectivity(self, params):
        p = self.populations
        G = self.G
        cpc = self.cellsPerColumn
        fpc = self.fsPerColumn
        neigh = params["topographicRadius"]

        # Bottom-up (topographic): VisualInput pixel (col) -> V1pyr cells in the
        # same column and columns within topographicRadius (the pooling / RF
        # analogue of Sulfaro's 13x13 ascending kernel).
        def visualToV1(source, target):
            srcCol = source._idx                    # input col == cell index
            tgtCol = self._colOf(target._idx, cpc)
            sx, sy = self._colXY(srcCol)
            tx, ty = self._colXY(tgtCol)
            if max(abs(sx - tx), abs(sy - ty)) <= neigh:
                w = params["visualWeight"] / cpc
                return gauss(w, abs(w) / 10)
            return None
        p["VisualInput"].addOutboundConnections(p["V1pyr"], visualToV1, self._axon5HT2A)

        # Cross-modal bottom-up: AudioInput -> V1pyr, sparse all-to-all (the
        # remapping synapses; weak at baseline, plastic under serotonin). Axonal
        # 5HT2A on these transmission axons, as in the two-region model.
        def audioToV1(source, target):
            if random() < params["audioToV1Likelihood"]:
                w = params["audioWeight"] / self.audioBins
                return gauss(w, abs(w) / 10)
            return None
        p["AudioInput"].addOutboundConnections(p["V1pyr"], audioToV1, self._axon5HT2A)

        # Self stream: V1 recurrent excitation within a column ...
        def v1RecurrentExc(source, target):
            si = source._idx
            ti = target._idx
            if self._colOf(si, cpc) == self._colOf(ti, cpc) and source is not target:
                w = params["v1RecurrentWeight"] / cpc
                return gauss(w, abs(w) / 10)
            return None
        p["V1pyr"].addOutboundConnections(p["V1pyr"], v1RecurrentExc)

        # ... and the local E/I loop: V1pyr -> V1fs (drive) and V1fs -> V1pyr
        # (inhibition), both within a column.
        def v1ToFs(source, target):
            si = source._idx
            ti = target._idx
            if self._colOf(si, cpc) == self._colOf(ti, fpc):
                w = params["v1ToFsWeight"] / cpc
                return gauss(w, abs(w) / 10)
            return None
        p["V1pyr"].addOutboundConnections(p["V1fs"], v1ToFs)

        def fsToV1(source, target):
            si = source._idx
            ti = target._idx
            if self._colOf(si, fpc) == self._colOf(ti, cpc):
                w = params["fsToV1Weight"] / fpc
                return gauss(w, abs(w) / 10)
            return None
        p["V1fs"].addOutboundConnections(p["V1pyr"], fsToV1)

        # Feedforward pooling into the secondary area: all V1pyr -> each Category
        # cell (the classifier reads the whole V1 map). Templates (per-category
        # selectivity) live in the weight sign/magnitude; here a mild positive
        # pooling weight, refined by template assignment in setCategoryTemplates.
        def v1ToCategory(source, target):
            w = params["v1ToCategoryWeight"] / (self.nCols * cpc)
            return gauss(w, abs(w) / 10)
        p["V1pyr"].addOutboundConnections(p["Category"], v1ToCategory)

        # Lateral competition among Category cells (winner-take-all via mutual
        # inhibition): each category inhibits the others.
        def categoryCompetition(source, target):
            if source is not target:
                w = params["categoryInhibitionWeight"]
                return gauss(w, abs(w) / 10)
            return None
        p["Category"].addOutboundConnections(p["Category"], categoryCompetition)

        # Top-down feedback: Category -> V1pyr. The winning category projects its
        # template back onto V1 as the top-down stream.
        def categoryToV1(source, target):
            w = params["categoryToV1Weight"] / self.categoryCount
            return gauss(w, abs(w) / 10)
        p["Category"].addOutboundConnections(p["V1pyr"], categoryToV1)

    def _assignV1Streams(self):
        # Tag each V1 pyramidal's inbound source populations into Sulfaro streams.
        p = self.populations
        streamByPop = {
            p["VisualInput"]: STREAM_BOTTOMUP,
            p["AudioInput"]: STREAM_BOTTOMUP,   # cross-modal ascending drive
            p["V1pyr"]: STREAM_SELF,            # recurrent excitation
            p["V1fs"]: STREAM_SELF,             # local inhibition (current-layer)
            p["Category"]: STREAM_TOPDOWN,      # descending feedback
        }
        for cell in p["V1pyr"].cells:
            for srcPop, stream in streamByPop.items():
                cell.assignStream(srcPop, stream)

    # ---- runtime controls ----

    def setVisualRates(self, rates):
        cells = self.populations["VisualInput"].cells
        for cell, r in zip(cells, rates):
            cell.setRate(float(r))

    def setAudioRates(self, rates):
        cells = self.populations["AudioInput"].cells
        for cell, r in zip(cells, rates):
            cell.setRate(float(r))

    def setVisualScotoma(self, mask):
        # mask: iterable of length nCols; True/1 -> that column's visual input
        # is silenced (rate 0). Used for the scotoma / full-blindness tests.
        cells = self.populations["VisualInput"].cells
        for cell, silenced in zip(cells, mask):
            if silenced:
                cell.setRate(0.0)

    def setSerotonin(self, level):
        transmitters = {"5HT2A": level, "5HT1A": level}
        for key in ("V1pyr", "V1fs", "Category"):
            self.populations[key].setDiffuseTransmitters(dict(transmitters))

    def set5HT2A(self, level):
        # Raise only 5HT2A (pharmacological / agonist), 5HT1A at baseline.
        base = self.params["serotoninLevel"]
        transmitters = {"5HT2A": level, "5HT1A": base}
        for key in ("V1pyr", "V1fs", "Category"):
            self.populations[key].setDiffuseTransmitters(dict(transmitters))

    def setV1BottomUpGain(self, gain):
        # Sulfaro neurochemical bridge: scale the bottom-up stream weight on every
        # V1 pyramidal (5HT2A dampening of bottom-up gain -> gain < 1).
        for cell in self.populations["V1pyr"].cells:
            cell.bottomUpGain = float(gain)

    def audioRemappingAxons(self):
        # The cross-modal AudioInput -> V1pyr synapses (analogue of S_B -> P_A).
        return self.populations["AudioInput"].outboundAxonsTo(self.populations["V1pyr"])

    def step(self):
        for population in self.populations.values():
            population.stepCells()
        for population in self.populations.values():
            population.stepOutputs()
