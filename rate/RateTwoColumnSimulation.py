from rate.RateSimulationBase import RateSimulationBase

'''
Four-epoch sensory-loss simulation on the rate two-region network, mirroring
network/TwoColumnSimulation.py and the methods draft's Input section:

  epoch 1  baseline: both inputs on, serotonin at baseline
  epoch 2  sensory loss: Input A rate -> 0 (stays 0 for the rest)
  epoch 3  serotonin + plasticity: serotonin in region A raised, and plasticity
           switched on for the cross-modal remapping synapses (S_B -> P_A) with
           the (inflated, per the draft) plasticity constants
  epoch 4  return: serotonin in A back to baseline, plasticity off

The rate model produces biologically sane firing rates (tens-hundreds of Hz),
so the validation target is the qualitative epoch structure and the
persistence of the cross-modal influence shift into epoch 4 -- the paper's
actual claims -- not the inflated absolute rates in the spiking figures (which
were an artifact of the 2003/2007 integration mismatch).
'''


class RateTwoColumnSimulation(RateSimulationBase):
    def epoch1(self):
        self.runEpoch(1)
        self.weightsByEpoch["1_baseline"] = [a.weight for a in self.remapping]

    def epoch2(self):
        self.network.populations["InputA"].setRate(0.0)
        self.runEpoch(2)
        self.weightsByEpoch["2_sensory_loss"] = [a.weight for a in self.remapping]

    def epoch3(self):
        self.network.setSerotoninA(self.params["remapSerotoninLevel"])
        for axon in self.remapping:
            axon.enablePlasticity(self.params["gamma_p"], self.params["gamma_d"],
                                  self.params["plasticityThreshold"],
                                  ceilingFactor=self.params["plasticityCeilingFactor"])
        self.runEpoch(3)
        for axon in self.remapping:
            axon.disablePlasticity()
        self.weightsByEpoch["3_serotonin_plasticity"] = [a.weight for a in self.remapping]

    def epoch4(self):
        self.network.setSerotoninA(self.params["serotoninLevelA"])
        self.runEpoch(4)
        self.weightsByEpoch["4_return"] = [a.weight for a in self.remapping]

    def run(self):
        self.warmup()
        self.epoch1()
        self.epoch2()
        self.epoch3()
        self.epoch4()
