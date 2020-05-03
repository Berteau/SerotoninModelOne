from Population import Population
from PoissonPopulation import PoissonPopulation
from Network import Network
from AxonalSerotoninReceptorFactory import AxonalSerotoninReceptorFactory
from SomaticSerotoninReceptorFactory import SomaticSerotoningReceptorFactory
from random import random, gauss
import functools

# TODO: change imports to a "import <file>" format, then call them with
# <file>.<class>


"""
Build two columns of input, pyramidal, fast spiking, and low threshold neurons with apporpriate interconnections.

INA -> PYA
INB -> PYB
INA -> PYB
INB -> PYA

PYA -> PYA
PYA -> PYA
PYB -> PYB
PYB -> PYB

PYA -> FSA
FSA -> FSA
PYA -> LTA

PYB -> FSB
FSB -> PYB
PYB -> LTA

LTA -> PYB
LTA -> FSB

LTB -> PYA
LYB -> FSB
"""


class TwoColumnNetwork(Network):
    def __init__(self, tau, parentSimulation, params, name):
        self.tau = tau
        self.parentSimulation = parentSimulation
        self.name = name
        self.populations = {}
        self.params = params

        # try:
        self.popCount = params["popCount"]
        self.serotoninLevelA = params["serotoninLevelA"]
        self.serotoninLevelB = params["serotoninLevelB"]
        self.inputWeightA = params["inputWeightA"]
        self.rateA = params["rateA"]
        self.inputWeightB = params["inputWeightB"]
        self.rateB = params["rateB"]
        self.crossModalWeightAB = params["inputWeightAB"]
        self.crossModalWeightBA = params["inputWeightBA"]
        self.somaticSerotonin2AReceptorWeight = params["Somatic5HT2AWeight"]
        self.somaticSerotonin1AReceptorWeight = params["Somatic5HT1AWeight"]
        self.axonalSerotonin2AReceptorWeight = params["Axonal5HT2AWeight"]
        self.axonalSerotonin1AReceptorWeight = params["Axonal5HT1AWeight"]
        self.pyramidalSelfExcitationWeight = params["pyramidalSelfExcitationWeight"]
        self.LTStoFSWeight = params["LTStoFSWeight"]
        self.LTStoPyramidalWeight = params["LTStoFSWeight"]

        # Set up cell parameters
        pyramidalParams = {}
        pyramidalParams["C"] = 100
        pyramidalParams["k"] = 3
        pyramidalParams["v_r"] = -60
        pyramidalParams["v_t"] = -50
        pyramidalParams["v_peak"] = 50
        pyramidalParams["a"] = 0.02
        pyramidalParams["b"] = 0.2
        pyramidalParams["c"] = -65
        pyramidalParams["d"] = 8
        pyramidalParams["type"] = "Pyramidal"

        fsParams = {}
        fsParams["C"] = 20
        fsParams["k"] = 1
        fsParams["v_r"] = -55
        fsParams["v_t"] = -40
        fsParams["v_peak"] = 25
        fsParams["a"] = 0.1
        fsParams["b"] = 0.2
        fsParams["c"] = -65
        fsParams["d"] = 2
        fsParams["type"] = "FS"

        ltsParams = {}
        ltsParams["C"] = 100
        ltsParams["k"] = 1
        ltsParams["v_r"] = -56
        ltsParams["v_t"] = -42
        ltsParams["v_peak"] = 40
        ltsParams["a"] = 0.02
        ltsParams["b"] = 0.25
        ltsParams["c"] = -65
        ltsParams["d"] = 2
        ltsParams["type"] = "LTS"

        lambdaAParams = {}
        lambdaAParams["spikeRate"] = self.rateA
        lambdaAParams["type"] = "Poisson"

        lambdaBParams = {}
        lambdaBParams["spikeRate"] = self.rateB
        lambdaBParams["type"] = "Poisson"

        # Define defuse receptor factories
        factorySomatic5HT2A = SomaticSerotoninDiffuseReceptorFactory(
            "5HT2A", lambda x: self.somaticSerotonin2AReceptorWeight, [])
        factorySomatic5HT2ALTS = SomaticSerotoninDiffuseReceptorFactory(
            "5HT2A", lambda x: params["Somatic5HT2AWeightLTS"], [])
        factorySomatic5HT1A = SomaticSerotoninDiffuseReceptorFactory(
            "5HT1A", lambda x: self.somaticSerotonin1AReceptorWeight, [])
        factoryAxonal5HT2A = AxonalSerotoninDiffuseReceptorFactory(
            "5HT2A", lambda x: self.axonalSerotonin2AReceptorWeight, [])
        factoryAxonal5HT1A = AxonalSerotoninDiffuseReceptorFactory(
            "5HT1A", lambda x: self.axonalSerotonin1AReceptorWeight, [])

        # Define diffuse Transmitters for Area A
        transmittersA = {}
        transmittersA["5HT2A"] = self.serotoninLevelA
        transmittersA["5HT1A"] = self.serotoninLevelA

        # Define diffuse transmitters for Area B
        transmittersB = {}
        transmittersB["5HT2A"] = self.serotoninLevelB
        transmittersB["5HT1A"] = self.serotoninLevelB

        # Build the populations
        self.populations["InputA"] = PoissonPopulation(
            tau, lambdaAParams, self.popCount, [], transmittersA, self, name)
        self.populations["InputB"] = PoissonPopulation(
            tau, lambdaBParams, self.popCount, [], transmittersB, self, name)

        # [factorySomatic5HT2A, factorySomatic5HT1A]
        self.populations["pyramidalsA"] = Population(
            self.tau, pyramidalParams, self.popCount, [
                factorySomatic5HT2A, factorySomatic5HT1A], transmittersA, self, "Area A Pyramidal Cells")
        self.populations["fastSpikingsA"] = Population(
            self.tau,
            fsParams,
            self.popCount,
            [],
            transmittersA,
            self,
            "Area A Fast Spiking Cells")
        self.populations["lowThresholdsA"] = Population(
            self.tau, ltsParams, self.popCount, [
                factorySomatic5HT2ALTS, factorySomatic5HT1A], transmittersA, self, "Area A Low Threshold Cells")

        self.populations["pyramidalsB"] = Population(
            self.tau, pyramidalParams, self.popCount, [
                factorySomatic5HT2A, factorySomatic5HT1A], transmittersB, self, "Area B Pyramidal Cells")
        self.populations["fastSpikingsB"] = Population(
            self.tau,
            fsParams,
            self.popCount,
            [],
            transmittersA,
            self,
            "Area B Fast Spiking Cells")
        self.populations["lowThresholdsB"] = Population(
            self.tau, ltsParams, self.popCount, [
                factorySomatic5HT2ALTS, factorySomatic5HT1A], transmittersB, self, "Area B Low Threshold Cells")

        # Build connections

        # First, unimodal input
        self.populations["InputA"].addOutboundConnections(
            self.populations["pyramidalsA"], self.gaussWeightWithChanceFactory(
                self.inputWeightA, 1.0), [factoryAxonal5HT2A], [factoryAxonal5HT2A])
        self.populations["InputB"].addOutboundConnections(
            self.populations["pyramidalsB"], self.gaussWeightWithChanceFactory(
                self.inputWeightB, 1.0), [factoryAxonal5HT2A], [factoryAxonal5HT2A])

        # Then cross-modal input
        self.populations["InputA"].addOutboundConnections(
            self.populations["pyramidalsB"],
            self.gaussWeightWithChanceFactory(
                self.params["inputWeightAB"],
                self.params["crossModalABLikelihood"]),
            [],
            [factoryAxonal5HT2A])
        self.populations["InputB"].addOutboundConnections(
            self.populations["pyramidalsA"],
            self.gaussWeightWithChanceFactory(
                self.params["inputWeightBA"],
                self.params["crossModalBALikelihood"]),
            [],
            [factoryAxonal5HT2A])

        # Now we do connections that run within each column

        # Pyramidal self-excitation
        self.populations["pyramidalsA"].addOutboundConnections(
            self.populations["pyramidalsA"], self.selfTargetingGaussWeightFactory(
                self.params["pyramidalSelfExcitationWeight"]), [], [])
        self.populations["pyramidalsA"].addOutboundConnections(
            self.populations["pyramidalsA"],
            self.gaussWeightWithChanceFactory(
                self.params["pyramidalToPyramidalWeight"],
                self.params["pyramidalToPyramidalLikelihood"]))
        self.populations["pyramidalsB"].addOutboundConnections(
            self.populations["pyramidalsB"], self.selfTargetingGaussWeightFactory(
                self.params["pyramidalSelfExcitationWeight"]), [], [])
        self.populations["pyramidalsB"].addOutboundConnections(
            self.populations["pyramidalsB"],
            self.gaussWeightWithChanceFactory(
                self.params["pyramidalToPyramidalWeight"],
                self.params["pyramidalToPyramidalLikelihood"]))

        # Now Pyramidal to Fast-Spiking
        self.populations["pyramidalsA"].addOutboundConnections(
            self.populations["fastSpikingsA"], self.gaussWeightWithChanceFactory(
                self.params["PyramidalsToFSWeight"], 1.0), [], [])
        self.populations["pyramidalsB"].addOutboundConnections(
            self.populations["fastSpikingsB"], self.gaussWeightWithChanceFactory(
                self.params["PyramidalsToFSWeight"], 1.0), [], [])

        # Fast Spiking to Pyramidal
        self.populations["fastSpikingsA"].addOutboundConnections(
            self.populations["pyramidalsA"], self.gaussWeightWithChanceFactory(
                self.params["FSToPyramidalsWeight"], 1.0), [], [])
        self.populations["fastSpikingsB"].addOutboundConnections(
            self.populations["pyramidalsB"], self.gaussWeightWithChanceFactory(
                self.params["FSToPyramidalsWeight"], 1.0), [], [])

        # Pyramidal to Low-Threshold
        self.populations["pyramidalsA"].addOutboundConnections(
            self.populations["lowThresholdsA"], self.gaussWeightWithChanceFactory(
                self.params["PyramidalsToLTSWeight"], 1.0), [], [])
        self.populations["pyramidalsB"].addOutboundConnections(
            self.populations["lowThresholdsB"], self.gaussWeightWithChanceFactory(
                self.params["PyramidalsToLTSWeight"], 1.0), [], [])

        # Now Cross-Modal LTS Connections
        self.populations["lowThresholdsA"].addOutboundConnections(
            self.populations["fastSpikingsB"], self.gaussWeightWithChanceFactory(
                self.params["LTStoFSWeight"], 1.0), [], [])
        self.populations["lowThresholdsA"].addOutboundConnections(
            self.populations["pyramidalsB"], self.gaussWeightWithChanceFactory(
                self.params["LTStoPyramidalsWeight"], 1.0), [], [])
        self.populations["lowThresholdsB"].addOutboundConnections(
            self.populations["fastSpikingsA"], self.gaussWeightWithChanceFactory(
                self.params["LTStoFSWeight"], 1.0), [], [])
        self.populations["lowThresholdsB"].addOutboundConnections(
            self.populations["pyramidalsA"], self.gaussWeightWithChanceFactory(
                self.params["LTStoPyramidalsWeight"], 1.0), [], [])

    # TODO: Separate out connection function factories to another library file
    # somewhere

    def setSerotoninA(self, transmittersA):
        aKeys = ["pyramidalsA", "fastSpikingsA", "lowThresholdsA"]
        for key in aKeys:
            self.populations[key].setDiffuseTransmitters(transmittersA)
            for inboundAxon in self.populations[key].inboundAxons:
                inboundAxon.updateDistalDiffuseTransmitters(transmittersA)
            for outboundAxon in self.populations[key].inboundAxons:
                outboundAxon.updateDistalDiffuseTransmitters(transmittersA)

    def setSerotoninB(self, transmittersB):
        bKeys = ["pyramidalsB", "fastSpikingsB", "lowThresholdsB"]
        for key in bKeys:
            self.populations[key].setDiffuseTransmitters(transmittersB)
            for inboundAxon in self.populations[key].inboundAxons:
                inboundAxon.updateDistalDiffuseTransmitters(transmittersB)
            for outboundAxon in self.populations[key].inboundAxons:
                outboundAxon.updateProximalDiffuseTransmitters(transmittersB)

    def gaussWeightWithChanceFactory(self, inputWeight, chance):
        def weightFunction(source, target):
            if random() < chance:
                return gauss(
                    inputWeight / self.popCount,
                    ((inputWeight / self.popCount) / 10))
            else:
                return None
        return weightFunction

    def selfTargetingGaussWeightFactory(self, inputWeight):
        def weightFunction(source, target):
            if source is target:
                return gauss(
                    inputWeight / self.popCount,
                    ((inputWeight / self.popCount) / 10))
            else:
                return None
        return weightFunction

    def step(self):
        for popName, population in self.populations.items():
            population.stepCells()
        for popName, population in self.populations.items():
            population.stepOutputs()
