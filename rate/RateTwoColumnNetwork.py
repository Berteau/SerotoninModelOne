from random import random, gauss

from rate.RatePopulation import RatePopulation
from rate.RateDiffuseReceptor import RateSomaticReceptorFactory, RateAxonalReceptorFactory

'''
Rate-model version of network/TwoColumnNetwork.py -- the paper's two-region
sensory-loss architecture (regions alpha/A and beta/B, each with pyramidal P,
fast-spiking F, and low-threshold-spiking L populations, plus a sensory input
generator S). Structure mirrors the spiking TwoColumnNetwork and the methods
draft's architecture figure:

  S_A -> P_A (unimodal, axonal 5HT2A)         S_B -> P_B
  S_A -> P_B (cross-modal)                    S_B -> P_A  (cross-modal; the
                                                           remapping synapses)
  P_r -> P_r (recurrent self + sparse all-to-all excitation)
  P_r -> F_r -> P_r  (local E/I loop)
  P_r -> L_r
  L_A -> F_B, L_A -> P_B  (cross-modal inhibition)   L_B -> F_A, L_B -> P_A

Somatic 5HT2A/5HT1A receptors sit on P and L cells; axonal 5HT2A receptors sit
on the S->P input axons (modulating transmission). Weights are drawn per-synapse
from a Gaussian with mean weight/popCount, exactly as in the spiking model.
'''


class RateTwoColumnNetwork:
    def __init__(self, tau, params, name="RateTwoColumn"):
        self.tau = tau
        self.params = params
        self.name = name
        self.popCount = params["popCount"]
        self.populations = {}

        transmittersA = {"5HT2A": params["serotoninLevelA"], "5HT1A": params["serotoninLevelA"]}
        transmittersB = {"5HT2A": params["serotoninLevelB"], "5HT1A": params["serotoninLevelB"]}

        somaP = [RateSomaticReceptorFactory("5HT2A", params["Somatic5HT2AWeight"]),
                 RateSomaticReceptorFactory("5HT1A", params["Somatic5HT1AWeight"])]
        somaL = [RateSomaticReceptorFactory("5HT2A", params["Somatic5HT2AWeightLTS"]),
                 RateSomaticReceptorFactory("5HT1A", params["Somatic5HT1AWeight"])]
        axon5HT2A = [RateAxonalReceptorFactory("5HT2A", params["Axonal5HT2AWeight"])]

        p = self.populations
        # Input generators
        p["InputA"] = RatePopulation(tau, None, self.popCount, [], {}, self, "InputA",
                                     isInput=True, inputRate=params["rateA"])
        p["InputB"] = RatePopulation(tau, None, self.popCount, [], {}, self, "InputB",
                                     isInput=True, inputRate=params["rateB"])
        # Region A
        p["pyramidalsA"] = RatePopulation(tau, "Pyramidal", self.popCount, somaP, transmittersA, self, "pyramidalsA")
        p["fastSpikingsA"] = RatePopulation(tau, "FS", self.popCount, [], transmittersA, self, "fastSpikingsA")
        p["lowThresholdsA"] = RatePopulation(tau, "LTS", self.popCount, somaL, transmittersA, self, "lowThresholdsA")
        # Region B
        p["pyramidalsB"] = RatePopulation(tau, "Pyramidal", self.popCount, somaP, transmittersB, self, "pyramidalsB")
        p["fastSpikingsB"] = RatePopulation(tau, "FS", self.popCount, [], transmittersB, self, "fastSpikingsB")
        p["lowThresholdsB"] = RatePopulation(tau, "LTS", self.popCount, somaL, transmittersB, self, "lowThresholdsB")

        gw = self._gaussWeightWithChance
        sw = self._selfTargetingGaussWeight

        # Unimodal input (axonal 5HT2A on these transmission axons)
        p["InputA"].addOutboundConnections(p["pyramidalsA"], gw(params["inputWeightA"], 1.0), axon5HT2A)
        p["InputB"].addOutboundConnections(p["pyramidalsB"], gw(params["inputWeightB"], 1.0), axon5HT2A)
        # Cross-modal input (S_B -> P_A are the remapping synapses)
        p["InputA"].addOutboundConnections(p["pyramidalsB"], gw(params["inputWeightAB"], params["crossModalABLikelihood"]), axon5HT2A)
        p["InputB"].addOutboundConnections(p["pyramidalsA"], gw(params["inputWeightBA"], params["crossModalBALikelihood"]), axon5HT2A)

        # Pyramidal recurrent excitation (self 1:1 + sparse all-to-all)
        for region in ("A", "B"):
            p["pyramidals" + region].addOutboundConnections(p["pyramidals" + region], sw(params["pyramidalSelfExcitationWeight"]))
            p["pyramidals" + region].addOutboundConnections(p["pyramidals" + region], gw(params["pyramidalToPyramidalWeight"], params["pyramidalToPyramidalLikelihood"]))
            # Local E/I loop
            p["pyramidals" + region].addOutboundConnections(p["fastSpikings" + region], gw(params["PyramidalsToFSWeight"], 1.0))
            p["fastSpikings" + region].addOutboundConnections(p["pyramidals" + region], gw(params["FSToPyramidalsWeight"], 1.0))
            p["pyramidals" + region].addOutboundConnections(p["lowThresholds" + region], gw(params["PyramidalsToLTSWeight"], 1.0))

        # Cross-modal inhibition from LTS
        p["lowThresholdsA"].addOutboundConnections(p["fastSpikingsB"], gw(params["LTStoFSWeight"], 1.0))
        p["lowThresholdsA"].addOutboundConnections(p["pyramidalsB"], gw(params["LTStoPyramidalsWeight"], 1.0))
        p["lowThresholdsB"].addOutboundConnections(p["fastSpikingsA"], gw(params["LTStoFSWeight"], 1.0))
        p["lowThresholdsB"].addOutboundConnections(p["pyramidalsA"], gw(params["LTStoPyramidalsWeight"], 1.0))

    # ---- weight function factories (same math as the spiking model) ----

    def _gaussWeightWithChance(self, inputWeight, chance):
        popCount = self.popCount
        def weightFunction(source, target):
            if random() < chance:
                return gauss(inputWeight / popCount, abs(inputWeight / popCount) / 10)
            return None
        return weightFunction

    def _selfTargetingGaussWeight(self, inputWeight):
        popCount = self.popCount
        def weightFunction(source, target):
            if source is target:
                return gauss(inputWeight / popCount, abs(inputWeight / popCount) / 10)
            return None
        return weightFunction

    # ---- serotonin control ----

    def setSerotoninA(self, level):
        self._setRegionSerotonin("A", level)

    def setSerotoninB(self, level):
        self._setRegionSerotonin("B", level)

    def _setRegionSerotonin(self, region, level):
        transmitters = {"5HT2A": level, "5HT1A": level}
        for key in ("pyramidals" + region, "fastSpikings" + region, "lowThresholds" + region):
            self.populations[key].setDiffuseTransmitters(transmitters)

    # ---- remapping synapses (S_B -> P_A) ----

    def remappingAxons(self):
        return self.populations["InputB"].outboundAxonsTo(self.populations["pyramidalsA"])

    def step(self):
        for population in self.populations.values():
            population.stepCells()
        for population in self.populations.values():
            population.stepOutputs()
