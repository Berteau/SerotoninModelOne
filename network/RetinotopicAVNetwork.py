import logging as log
import numpy as np
from model.Population import Population
from model.PoissonPopulation import PoissonPopulation
from network.Network import Network
from model.AxonalSerotoninReceptorFactory import AxonalSerotoninDiffuseReceptorFactory
from model.SomaticSerotoninReceptorFactory import SomaticSerotoninDiffuseReceptorFactory
from random import random, gauss

'''
A retinotopic / tonotopic expansion of SerotoninAVNetwork.

Instead of 3 visual + 2 audio columns wired (mostly) all-to-all, this network
builds a (gridRows x gridCols) grid of visual columns - one per pixel of a
downsampled greyscale video frame - and a 1D chain of numAudioColumns
tonotopic audio columns - one per frequency band of a beep score.

Each column (visual or audio) has the same internal structure as a column in
SerotoninAVNetwork: an Input population driving a Pyramidal population, which
in turn drives (and is driven by) Fast-Spiking and Low-Threshold populations.

Connectivity between columns is local/topographic rather than all-to-all:
  * Visual columns laterally connect (pyramidal -> pyramidal) to their
    4-connected grid neighbors. This is the substrate for retinotopic
    remapping: if a column's own input is silenced, its spared neighbors can
    still drive it through these lateral connections.
  * Audio columns laterally connect to their tonotopic chain neighbors
    (band i <-> band i+1).
  * Cross-modal connections are structured and sparse, mirroring how the
    original network only cross-connected its edge visual columns (V1, V3) to
    the two audio columns: here, the leftmost column of visual field connects
    to the low half of the tonotopic chain, and the rightmost column connects
    to the high half, one visual row per band.
'''

class RetinotopicAVNetwork(Network):
    def __init__(self, tau, parentSimulation, params, name):
        self.tau = tau
        self.parentSimulation = parentSimulation
        self.name = name
        self.populations = {}
        self.params = params

        self.popCount = params["popCount"]
        self.gridRows = params.get("gridRows", 10)
        self.gridCols = params.get("gridCols", 10)
        self.numAudioColumns = params.get("numAudioColumns", 20)

        self.serotoninLevelV = params["serotoninLevelV"]
        self.serotoninLevelA = params["serotoninLevelA"]
        self.baselineRateV = params["baselineRateV"]
        self.baselineRateA = params["baselineRateA"]
        self.inputWeightV = params["inputWeightV"]
        self.inputWeightA = params["inputWeightA"]
        self.crossModalWeightVA = params["inputWeightVA"]
        self.crossModalWeightAV = params["inputWeightAV"]
        self.crossModalLikelihood = params["crossModalLikelihood"]
        self.lateralVisualWeight = params["lateralVisualWeight"]
        self.lateralVisualLikelihood = params["lateralVisualLikelihood"]
        self.lateralAudioWeight = params["lateralAudioWeight"]
        self.lateralAudioLikelihood = params["lateralAudioLikelihood"]
        self.somaticSerotonin2AReceptorWeight = params["Somatic5HT2AWeight"]
        self.somaticSerotonin1AReceptorWeight = params["Somatic5HT1AWeight"]
        self.axonalSerotonin2AReceptorWeight = params["Axonal5HT2AWeight"]
        self.axonalSerotonin1AReceptorWeight = params["Axonal5HT1AWeight"]
        self.pyramidalSelfExcitationWeight = params["pyramidalSelfExcitationWeight"]

        # Set up cell parameters (identical to SerotoninAVNetwork)
        pyramidalParams = {"C": 100, "k": 3, "v_r": -60, "v_t": -50, "v_peak": 50,
                            "a": 0.01, "b": 5, "c": -60, "d": 400, "type": "Pyramidal"}
        fsParams = {"C": 20, "k": 1, "v_r": -55, "v_t": -40, "v_peak": 25,
                    "a": 0.15, "b": 8, "c": -55, "d": 200, "type": "FS"}
        ltsParams = {"C": 100, "k": 1, "v_r": -56, "v_t": -42, "v_peak": 40,
                     "a": 0.03, "b": 8, "c": -50, "d": 200, "type": "LTS"}
        inputVParams = {"spikeRate": self.baselineRateV, "type": "Poisson"}
        inputAParams = {"spikeRate": self.baselineRateA, "type": "Poisson"}

        # Diffuse receptor factories
        factorySomatic5HT2A = SomaticSerotoninDiffuseReceptorFactory("5HT2A", lambda x: self.somaticSerotonin2AReceptorWeight, [])
        factoryAxonal5HT2A = AxonalSerotoninDiffuseReceptorFactory("5HT2A", lambda x: self.axonalSerotonin2AReceptorWeight, [])

        self.transmittersV = {"5HT2A": self.serotoninLevelV, "5HT1A": self.serotoninLevelV}
        self.transmittersA = {"5HT2A": self.serotoninLevelA, "5HT1A": self.serotoninLevelA}

        # Build visual columns (retinotopic grid)
        self.visualColumnNames = {}
        for r in range(self.gridRows):
            for c in range(self.gridCols):
                colName = self.visualColumnName(r, c)
                self.visualColumnNames[(r, c)] = colName
                self._buildColumn(colName, inputVParams, pyramidalParams, fsParams, ltsParams,
                                   self.transmittersV, factorySomatic5HT2A)

        # Build audio columns (tonotopic chain)
        self.audioColumnNames = {}
        for k in range(self.numAudioColumns):
            colName = self.audioColumnName(k)
            self.audioColumnNames[k] = colName
            self._buildColumn(colName, inputAParams, pyramidalParams, fsParams, ltsParams,
                               self.transmittersA, factorySomatic5HT2A)

        # Within-column connectivity (feedforward + local E/I loop), same pattern for every column
        for colName in list(self.visualColumnNames.values()) + list(self.audioColumnNames.values()):
            self._wireColumnInternals(colName, factoryAxonal5HT2A)

        # Lateral visual-visual connectivity: 4-connected grid neighbors, pyramidal -> pyramidal
        for r in range(self.gridRows):
            for c in range(self.gridCols):
                sourceName = self.visualColumnNames[(r, c)]
                for (nr, nc) in self.gridNeighbors(r, c):
                    targetName = self.visualColumnNames[(nr, nc)]
                    self.populations["pyramidals" + sourceName].addOutboundConnections(
                        self.populations["pyramidals" + targetName],
                        self.gaussWeightWithChanceFactory(self.lateralVisualWeight, self.lateralVisualLikelihood), [], [])

        # Lateral audio-audio connectivity: tonotopic chain neighbors, pyramidal -> pyramidal
        for k in range(self.numAudioColumns):
            for neighborK in self._chainNeighbors(k, self.numAudioColumns):
                sourceName = self.audioColumnNames[k]
                targetName = self.audioColumnNames[neighborK]
                self.populations["pyramidals" + sourceName].addOutboundConnections(
                    self.populations["pyramidals" + targetName],
                    self.gaussWeightWithChanceFactory(self.lateralAudioWeight, self.lateralAudioLikelihood), [], [])

        # Cross-modal connectivity: left visual edge <-> low half of tonotopic chain,
        # right visual edge <-> high half of tonotopic chain, one visual row per band.
        self.leftEdgeBandForRow = self._edgeBandMapping(0, self.numAudioColumns // 2 - 1)
        self.rightEdgeBandForRow = self._edgeBandMapping(self.numAudioColumns // 2, self.numAudioColumns - 1)
        for r in range(self.gridRows):
            self._wireCrossModalPair(self.visualColumnNames[(r, 0)], self.audioColumnNames[self.leftEdgeBandForRow[r]], factoryAxonal5HT2A)
            self._wireCrossModalPair(self.visualColumnNames[(r, self.gridCols - 1)], self.audioColumnNames[self.rightEdgeBandForRow[r]], factoryAxonal5HT2A)

    # ---- Naming helpers ----

    def visualColumnName(self, r, c):
        return "V%02d_%02d" % (r, c)

    def audioColumnName(self, k):
        return "A%02d" % k

    # ---- Topology helpers ----

    def gridNeighbors(self, r, c):
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(nr, nc) for (nr, nc) in candidates if 0 <= nr < self.gridRows and 0 <= nc < self.gridCols]

    def _chainNeighbors(self, k, count):
        candidates = [k - 1, k + 1]
        return [n for n in candidates if 0 <= n < count]

    def _edgeBandMapping(self, bandStart, bandEnd):
        if self.gridRows == 1:
            return np.array([bandStart])
        return np.round(np.linspace(bandStart, bandEnd, self.gridRows)).astype(int)

    # ---- Construction helpers ----

    def _buildColumn(self, colName, inputParams, pyramidalParams, fsParams, ltsParams, transmitters, factorySomatic5HT2A):
        self.populations["Input" + colName] = PoissonPopulation(self.tau, inputParams, self.popCount, [], {}, self, self.name)
        self.populations["pyramidals" + colName] = Population(self.tau, pyramidalParams, self.popCount, [factorySomatic5HT2A], transmitters, self, colName + " Pyramidal Cells")
        self.populations["fastSpikings" + colName] = Population(self.tau, fsParams, self.popCount, [factorySomatic5HT2A], transmitters, self, colName + " Fast Spiking Cells")
        self.populations["lowThresholds" + colName] = Population(self.tau, ltsParams, self.popCount, [factorySomatic5HT2A], transmitters, self, colName + " Low Threshold Cells")

    def _wireColumnInternals(self, colName, factoryAxonal5HT2A):
        isVisual = colName.startswith("V")
        inputWeight = self.inputWeightV if isVisual else self.inputWeightA
        self.populations["Input" + colName].addOutboundConnections(
            self.populations["pyramidals" + colName], self.gaussWeightWithChanceFactory(inputWeight, 1.0), [], [factoryAxonal5HT2A])
        self.populations["pyramidals" + colName].addOutboundConnections(
            self.populations["pyramidals" + colName], self.selfTargetingGaussWeightFactory(self.pyramidalSelfExcitationWeight), [], [])
        self.populations["pyramidals" + colName].addOutboundConnections(
            self.populations["fastSpikings" + colName], self.gaussWeightWithChanceFactory(self.params["PyramidalsToFSWeight"], 1.0), [], [])
        self.populations["fastSpikings" + colName].addOutboundConnections(
            self.populations["pyramidals" + colName], self.gaussWeightWithChanceFactory(self.params["FSToPyramidalsWeight"], 1.0), [], [])
        self.populations["pyramidals" + colName].addOutboundConnections(
            self.populations["lowThresholds" + colName], self.gaussWeightWithChanceFactory(self.params["PyramidalsToLTSWeight"], 1.0), [], [])

    def _wireCrossModalPair(self, visualColName, audioColName, factoryAxonal5HT2A):
        self.populations["Input" + visualColName].addOutboundConnections(
            self.populations["pyramidals" + audioColName],
            self.gaussWeightWithChanceFactory(self.crossModalWeightVA, self.crossModalLikelihood), [], [factoryAxonal5HT2A])
        self.populations["Input" + audioColName].addOutboundConnections(
            self.populations["pyramidals" + visualColName],
            self.gaussWeightWithChanceFactory(self.crossModalWeightAV, self.crossModalLikelihood), [], [factoryAxonal5HT2A])

    # ---- Weight function factories (same math as SerotoninAVNetwork) ----

    def gaussWeightWithChanceFactory(self, inputWeight, chance):
        def weightFunction(source, target):
            if random() < chance:
                return gauss(inputWeight / self.popCount, ((inputWeight / self.popCount) / 10))
            else:
                return None
        return weightFunction

    def selfTargetingGaussWeightFactory(self, inputWeight):
        def weightFunction(source, target):
            if source is target:
                return gauss(inputWeight / self.popCount, ((inputWeight / self.popCount) / 10))
            else:
                return None
        return weightFunction

    # ---- Runtime drive / lesion / serotonin controls used by RetinotopicAVSimulation ----

    def setVisualInputRate(self, r, c, rateHz):
        self.populations["Input" + self.visualColumnNames[(r, c)]].setRate(rateHz)

    def setVisualInputRates(self, rateGrid):
        for r in range(self.gridRows):
            for c in range(self.gridCols):
                self.setVisualInputRate(r, c, rateGrid[r, c])

    def setAudioInputRate(self, k, rateHz):
        self.populations["Input" + self.audioColumnNames[k]].setRate(rateHz)

    def setAudioInputRates(self, rateVector):
        for k in range(self.numAudioColumns):
            self.setAudioInputRate(k, rateVector[k])

    def setSerotoninForColumns(self, columnNames, transmitters):
        for colName in columnNames:
            for popPrefix in ["pyramidals", "fastSpikings", "lowThresholds"]:
                population = self.populations[popPrefix + colName]
                population.setDiffuseTransmitters(transmitters)
                for inboundAxon in population.inboundAxons:
                    inboundAxon.updateDistalDiffuseTransmitters(transmitters)
                for outboundAxon in population.outboundAxons:
                    outboundAxon.updateProximalDiffuseTransmitters(transmitters)

    def getAxonsBetween(self, sourcePopName, targetPopName):
        sourcePop = self.populations[sourcePopName]
        targetPop = self.populations[targetPopName]
        return sourcePop.outboundAxonsByTargetPop.get(targetPop, [])

    def step(self):
        for popName, population in self.populations.items():
            population.stepCells()
        for popName, population in self.populations.items():
            population.stepOutputs()
