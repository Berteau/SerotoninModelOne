from rate.RateSimulationBase import RateSimulationBase

'''
Three-epoch pharmacological (5HT2A-agonist / psilocybin) simulation on the rate
two-region network, following the methods draft's pharmacological protocol.

  epoch 1  baseline: both inputs on, serotonin at baseline, plasticity off
  epoch 2  5HT2A agonism: ONLY 5HT2A raised (to pharma5HT2ALevel) in BOTH
           regions -- somatic and axonal, 5HT1A left at baseline -- with
           plasticity on for the cross-modal S_B -> P_A synapses at an
           order-of-magnitude weaker rate than the sensory experiment
  epoch 3  return: 5HT2A back to baseline in both regions, plasticity off

Unlike the sensory-loss experiment there is NO sensory loss, so Input alpha
remains present and Input alpha keeps its dominant influence over P_alpha
throughout (the paper's mean pharmacological result). Note: the paper's
headline pharmacological finding is an increase in the *variability* of
cross-modal influence under 5HT2A agonism; that is a stochastic effect the
deterministic rate model does not reproduce, so this simulation captures only
the mean behavior.
'''


class RatePharmaSimulation(RateSimulationBase):
    def _set_5ht2a_both_regions(self, level5ht2a):
        base = self.params["serotoninLevelA"]
        for region in ("A", "B"):
            self.network.setRegionTransmitters(region, {"5HT2A": level5ht2a, "5HT1A": base})

    def epoch1(self):
        self.runEpoch(1)
        self.weightsByEpoch["1_baseline"] = [a.weight for a in self.remapping]

    def epoch2(self):
        self._set_5ht2a_both_regions(self.params["pharma5HT2ALevel"])
        for axon in self.remapping:
            axon.enablePlasticity(self.params["pharma_gamma_p"], self.params["pharma_gamma_d"],
                                  self.params["plasticityThreshold"],
                                  ceilingFactor=self.params["plasticityCeilingFactor"])
        self.runEpoch(2)
        for axon in self.remapping:
            axon.disablePlasticity()
        self.weightsByEpoch["2_5ht2a_agonism"] = [a.weight for a in self.remapping]

    def epoch3(self):
        self._set_5ht2a_both_regions(self.params["serotoninLevelA"])
        self.runEpoch(3)
        self.weightsByEpoch["3_return"] = [a.weight for a in self.remapping]

    def run(self):
        self.warmup()
        self.epoch1()
        self.epoch2()
        self.epoch3()
