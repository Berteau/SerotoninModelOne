import numpy as np

from rate.RateTwoColumnNetwork import RateTwoColumnNetwork

'''
Shared machinery for the rate two-region experiments (sensory-loss and
pharmacological). Subclasses define the epoch protocol in run().

Common features:
  - a warmup period run at baseline before recording, so recorded epochs start
    from steady state (avoids the startup-overshoot transient in epoch 1);
  - per-step recording of the mean remapping-synapse (S_B -> P_A) weight;
  - epoch boundaries for plotting;
  - weightsByEpoch snapshots keyed by epoch label.
'''


class RateSimulationBase:
    def __init__(self, params):
        self.params = params
        self.tau = params["tau"]
        self.epochDurationMs = params["epochDurationMs"]
        self.warmupMs = params.get("warmupMs", 500.0)
        self.network = RateTwoColumnNetwork(self.tau, params)
        self.remapping = self.network.remappingAxons()   # S_B -> P_A

        self.weightHistory = []
        self.epochBoundaries = []
        self.weightsByEpoch = {}

    def _record_weight(self):
        if self.remapping:
            self.weightHistory.append(float(np.mean([a.weight for a in self.remapping])))

    def warmup(self):
        for _ in np.arange(0.0, self.warmupMs, self.tau):
            self.network.step()
        self._resetRecords()

    def _resetRecords(self):
        # Discard everything recorded during warmup so the recorded window
        # begins at t=0 in steady state.
        self.weightHistory = []
        self.epochBoundaries = []
        for pop in self.network.populations.values():
            pop.rateRecord = []
            pop.ablationInfluence = {}
            for cell in pop.cells:
                cell.rateRecord = []

    def runEpoch(self, epoch):
        start = self.epochDurationMs * (epoch - 1)
        end = self.epochDurationMs * epoch
        for _ in np.arange(start, end, self.tau):
            self.network.step()
            self._record_weight()
        self.epochBoundaries.append(end)

    def run(self):
        raise NotImplementedError
