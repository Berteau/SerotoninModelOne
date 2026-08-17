from random import random, gauss

import numpy as np

from rate.RatePopulation import RatePopulation
from rate.NormalizedMixtureNeuron import (
    NormalizedMixtureNeuron, STREAM_BOTTOMUP, STREAM_TOPDOWN, STREAM_SELF)
from rate.RateDiffuseReceptor import RateSomaticReceptorFactory, RateAxonalReceptorFactory
from rate.ARTClassifier import REJECT

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

        # Secondary area: either the first-pass competitive Category layer, or
        # (useART) a Fuzzy ART classifier that reads the V1 map and drives a
        # per-column TopDown input population with the winning category's
        # template (content-specific feedback; zero on reject).
        self.useART = bool(params.get("useART", False))
        self.art = None
        self.artActive = False
        # V1 column rate (Hz) that maps to feature value 1.0 when normalizing the
        # V1 map into the ART [0,1] feature space; and the Hz the top-down
        # template's value 1.0 projects back as.
        self.artReferenceRate = float(params.get("artReferenceRate", 6.0))
        self.topDownDriveHz = float(params.get("topDownDriveHz", params["maxRateHz"]))
        self.lastART = {"label": None, "reject": True, "match": 0.0, "category": None}

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

        # ---- secondary area ----
        if self.useART:
            # Per-column top-down input population driven by the ART classifier.
            p["TopDown"] = RatePopulation(tau, None, self.nCols, [], {}, self, "TopDown",
                                          isInput=True, inputRate=0.0)
        else:
            # First-pass competitive classifier.
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

        if self.useART:
            # Top-down feedback: TopDown column c -> V1pyr cells in column c.
            # The ART classifier sets TopDown column rates each step to the
            # winning category's template (retinotopic, content-specific).
            def topDownToV1(source, target):
                if source._idx == self._colOf(target._idx, cpc):
                    w = params["categoryToV1Weight"] / cpc
                    return gauss(w, abs(w) / 10)
                return None
            p["TopDown"].addOutboundConnections(p["V1pyr"], topDownToV1)
        else:
            # Feedforward pooling into the secondary area: all V1pyr -> each
            # Category cell (the classifier reads the whole V1 map).
            def v1ToCategory(source, target):
                w = params["v1ToCategoryWeight"] / (self.nCols * cpc)
                return gauss(w, abs(w) / 10)
            p["V1pyr"].addOutboundConnections(p["Category"], v1ToCategory)

            # Lateral competition among Category cells (winner-take-all).
            def categoryCompetition(source, target):
                if source is not target:
                    w = params["categoryInhibitionWeight"]
                    return gauss(w, abs(w) / 10)
                return None
            p["Category"].addOutboundConnections(p["Category"], categoryCompetition)

            # Top-down feedback: Category -> V1pyr.
            def categoryToV1(source, target):
                w = params["categoryToV1Weight"] / self.categoryCount
                return gauss(w, abs(w) / 10)
            p["Category"].addOutboundConnections(p["V1pyr"], categoryToV1)

    def _assignV1Streams(self):
        # Tag each V1 pyramidal's inbound source populations into Sulfaro streams.
        p = self.populations
        topDownPop = p["TopDown"] if self.useART else p["Category"]
        streamByPop = {
            p["VisualInput"]: STREAM_BOTTOMUP,
            p["AudioInput"]: STREAM_BOTTOMUP,   # cross-modal ascending drive
            p["V1pyr"]: STREAM_SELF,            # recurrent excitation
            p["V1fs"]: STREAM_SELF,             # local inhibition (current-layer)
            topDownPop: STREAM_TOPDOWN,         # descending feedback
        }
        for cell in p["V1pyr"].cells:
            for srcPop, stream in streamByPop.items():
                cell.assignStream(srcPop, stream)

    def _serotoninTargets(self):
        # Populations carrying somatic serotonin receptors (exclude input pops).
        keys = ["V1pyr", "V1fs"]
        if not self.useART:
            keys.append("Category")
        return keys

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
        for key in self._serotoninTargets():
            self.populations[key].setDiffuseTransmitters(dict(transmitters))
        self._updateMixtureFromSerotonin(level)

    def set5HT2A(self, level):
        # Raise only 5HT2A (pharmacological / agonist), 5HT1A at baseline.
        base = self.params["serotoninLevel"]
        transmitters = {"5HT2A": level, "5HT1A": base}
        for key in self._serotoninTargets():
            self.populations[key].setDiffuseTransmitters(dict(transmitters))
        self._updateMixtureFromSerotonin(level)

    def _updateMixtureFromSerotonin(self, level5ht2a):
        # Sulfaro / Seillier-Michaiel neurochemical bridge: 5HT2A dampens V1's
        # bottom-up gain, i.e. lowers the ascending stream weight in the
        # normalized mixture (shifting the feedforward:feedback ratio toward
        # feedback). This SUPPLEMENTS the somatic (additive) and axonal
        # (transmission) serotonin effects; it is the only one that actually
        # moves the Sulfaro competition. Opt-in via serotoninShiftsMixture.
        #
        #   bottomUpGain(L) = clamp(1 - damp * (L - baseline)/baseline, floor, 1)
        #
        # so gain = 1 at baseline serotonin and falls toward `floor` as 5HT2A
        # rises. Note this partly opposes the axonal 5HT2A effect (which raises
        # bottom-up *transmission*); they act on different quantities (stream
        # weight vs axon efficacy) and coexist, as documented.
        if not self.params.get("serotoninShiftsMixture", False):
            return
        base = self.params["serotoninLevel"]
        damp = self.params.get("bottomUpGainDamp", 0.2)
        floor = self.params.get("minBottomUpGain", 0.1)
        gain = 1.0 - damp * max(0.0, (level5ht2a - base) / base)
        gain = max(floor, min(1.0, gain))
        self.setV1BottomUpGain(gain)

    def setV1BottomUpGain(self, gain, cells=None):
        # Scale the bottom-up stream weight on V1 pyramidals (all, or a subset --
        # e.g. the deprived region). Called by the serotonin coupling, by analyses
        # that sweep the gain, and by the Realistic-5HT manipulation.
        cells = self.populations["V1pyr"].cells if cells is None else cells
        for cell in cells:
            cell.bottomUpGain = float(gain)

    def setV1SomaticSerotonin(self, level, cells=None):
        # Set the somatic 5HT receptor level (both 5HT2A and 5HT1A) on V1
        # pyramidals (all, or a subset -- e.g. the deprived region), directly on
        # the receptors. Used by Realistic-5HT to REDUCE serotonin in deprived
        # visual cortex. Bypasses the population transmitter dict (which the
        # Realistic path does not read), so somatic V1 serotonin can move
        # independently of the cross-modal axonal serotonin.
        cells = self.populations["V1pyr"].cells if cells is None else cells
        for cell in cells:
            for receptor in cell.diffuseReceptors:
                receptor.setLevel(float(level))

    def setDeprivedIntrinsicDrive(self, offset, cells=None):
        # Deprivation-induced intrinsic-excitability homeostasis (Desai, Rutherford
        # & Turrigiano 1999): an input-INDEPENDENT tonic depolarization of the
        # deprived region. Unlike bottom-up gain (which amplifies an absent input)
        # or disinhibition (which removes an absent, activity-driven inhibition),
        # this can raise firing in a cortex that has lost its feedforward drive --
        # the source of the transient hyperactivity that provides the postsynaptic
        # activity Hebbian cross-modal remapping needs to bootstrap. Applied via
        # the neuron's external (injected) current, which is additive outside the
        # normalized mixture.
        cells = self.populations["V1pyr"].cells if cells is None else cells
        for cell in cells:
            cell.setInjectedCurrent(float(offset))

    def setCrossModalAxonalSerotonin(self, level, axons=None):
        # Set the axonal 5HT receptor level on the cross-modal AudioInput -> V1
        # remapping axons (all, or a subset), raising transmission (lowering
        # failure). Used by Realistic-5HT to INCREASE serotonin on the auditory
        # cross-modal input, independently of somatic V1 serotonin.
        axons = self.audioRemappingAxons() if axons is None else axons
        for axon in axons:
            for receptor in axon.axonalReceptors:
                receptor.setLevel(float(level))

    def audioRemappingAxons(self):
        # The cross-modal AudioInput -> V1pyr synapses (analogue of S_B -> P_A).
        return self.populations["AudioInput"].outboundAxonsTo(self.populations["V1pyr"])

    # ---- ART secondary area ----

    def v1ColumnRates(self):
        # Per-column mean V1 pyramidal rate (Hz), length nCols.
        cpc = self.cellsPerColumn
        cells = self.populations["V1pyr"].cells
        return np.array([np.mean([cells[c * cpc + k].rate for k in range(cpc)])
                         for c in range(self.nCols)])

    def v1FeatureVector(self):
        # Normalize the V1 column-rate map into the ART [0,1] feature space by a
        # FIXED reference rate (not a per-pattern max), so overall activity level
        # is meaningful: a globally suppressed (deprived) map yields low feature
        # values -> rejected by the ART activity gate, while a serotonin-inflated
        # map yields high values -> can be classified (a hallucination readout).
        return np.clip(self.v1ColumnRates() / self.artReferenceRate, 0.0, 1.0)

    def attachART(self, art):
        self.art = art

    def setARTActive(self, active):
        # When active, ART reads V1 and drives TopDown each step. Kept off during
        # classifier pre-training so V1 is purely bottom-up driven.
        self.artActive = bool(active)
        if not active:
            for cell in self.populations["TopDown"].cells:
                cell.setRate(0.0)

    def updateTopDownFromART(self):
        # ART reads the current V1 feature, classifies, and drives the TopDown
        # population with the winning category's template (0 on reject).
        x = self.v1FeatureVector()
        label, (cat, match) = self.art.classify(x, return_detail=True)
        reject = (label == REJECT) or (cat is None)
        if reject:
            template = np.zeros(self.nCols)
        else:
            template = self.art.category_template(cat)   # [0,1]^nCols
        rates = template * self.topDownDriveHz
        for cell, r in zip(self.populations["TopDown"].cells, rates):
            cell.setRate(float(r))
        self.lastART = {"label": None if reject else label,
                        "reject": bool(reject), "match": float(match), "category": cat}

    def renderTopDownPercept(self, template, settleMs, gain=1.0,
                             disconnectBottomUp=True):
        """Drive V1 with ONLY a top-down template and return the settled V1
        column-rate map -- the percept that top-down feedback alone paints on V1.

        template : per-column [0,1] pattern (e.g. an ART category template, from
                   art.category_template(j)).
        gain     : multiplies the top-down SYNAPTIC WEIGHT (a linear current
                   boost) to show the hallucination under stronger-than-observed
                   feedback influence. Note gain is applied to the weight, not the
                   TopDown firing rate, because the synaptic conductance saturates
                   in rate so rate scaling has little effect above ~30 Hz.
        disconnectBottomUp : silence the visual+audio inputs AND remove the
                   bottom-up stream from the V1 mixture (weight -> 0). This is
                   Sulfaro's "sensory disconnection" scenario: with ascending
                   input downweighted to zero, the normalized mixture renders the
                   pure top-down (imagined/hallucinated) percept, undiluted by the
                   empty bottom-up slot in the denominator.

        All modified state (ART-active flag, input rates, bottom-up stream
        weights, top-down axon weights) is restored before returning.
        """
        template = np.asarray(template, dtype=float)
        wasActive = self.artActive
        self.resetActivity()
        self.setARTActive(False)   # hold TopDown fixed; also zeros it, so set AFTER

        v1 = self.populations["V1pyr"].cells
        saved_wbu = [c.streamWeights[STREAM_BOTTOMUP] for c in v1]
        if disconnectBottomUp:
            for cell in self.populations["VisualInput"].cells:
                cell.setRate(0.0)
            for cell in self.populations["AudioInput"].cells:
                cell.setRate(0.0)
            for c in v1:
                c.streamWeights[STREAM_BOTTOMUP] = 0.0

        tdAxons = self.populations["TopDown"].outboundAxons
        saved_w = [a.weight for a in tdAxons]
        for a in tdAxons:
            a.weight *= gain
            a._recomputeEffWeight()

        rates = np.clip(template * self.topDownDriveHz, 0.0, None)
        for cell, r in zip(self.populations["TopDown"].cells, rates):
            cell.setRate(float(r))
        for _ in range(int(round(settleMs / self.tau))):
            self.step()               # artActive is False, so TopDown stays clamped
        percept = self.v1ColumnRates().reshape(self.G, self.G)

        # restore
        for a, w in zip(tdAxons, saved_w):
            a.weight = w; a._recomputeEffWeight()
        for c, w in zip(v1, saved_wbu):
            c.streamWeights[STREAM_BOTTOMUP] = w
        self.artActive = wasActive
        return percept

    def setCellHistoryRecording(self, enabled):
        # Turn per-cell rateRecord accumulation on/off across every population.
        # Off keeps long (10x10, hundreds of thousands of steps) runs within the
        # memory budget: the per-cell history is the dominant cost (~7 GB) and is
        # unused by the grand-plan metrics/plots (population-level rateRecord and
        # the recorded scalar series are untouched).
        for pop in self.populations.values():
            for cell in pop.cells:
                cell.recordCellHistory = bool(enabled)

    def resetActivity(self):
        # Zero all firing rates, conductances, and synaptic accumulators, for a
        # clean presentation (used between ART training images).
        for pop in self.populations.values():
            for cell in pop.cells:
                cell.rate = 0.0
                if hasattr(cell, "clearSynapticAccumulators"):
                    cell.clearSynapticAccumulators()
            for axon in pop.outboundAxons:
                axon.g_ampa = axon.g_nmda = axon.g_gaba = 0.0

    def step(self):
        if self.useART and self.art is not None and self.artActive:
            self.updateTopDownFromART()
        for population in self.populations.values():
            population.stepCells()
        for population in self.populations.values():
            population.stepOutputs()


def pretrainART(network, inputSet, art, settleMs):
    """Train the ART classifier on the V1 representations of the labeled stimuli.

    Each stimulus is presented (top-down inactive, so V1 is purely bottom-up
    driven), V1 is settled for settleMs, and ART learns (V1 feature -> label).
    The classifier is thus trained on the actual V1 activity patterns it will
    later classify, not on the raw edge images. Attaches the trained ART to the
    network but leaves it inactive; the caller enables it with setARTActive(True).
    """
    tau = network.tau
    network.setARTActive(False)
    steps = int(round(settleMs / tau))
    for st in inputSet.stimuli:
        network.resetActivity()
        network.setVisualRates(inputSet.visualRates(st))
        network.setAudioRates(inputSet.audioRates(st))
        for _ in range(steps):
            network.step()
        art.train_one(network.v1FeatureVector(), st.label)
    network.attachART(art)
    return art
