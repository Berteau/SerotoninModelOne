# from scipy.stats import poisson
from numpy.random import poisson
from model.Neuron import Neuron

class PoissonNeuron(Neuron):
    def __init__(self, tau, meanSpikesPerSecond, name, outputs=None, parentPop=None):
        # Determine Lambda and set up our poisson distribution
        self.expectedMeanSpikeRate = meanSpikesPerSecond
        self.meanSpikeProbabilityPerMs = (self.expectedMeanSpikeRate / 10)
        self.poissonLambda = self.meanSpikeProbabilityPerMs * tau # Lambda mean probability of a spike per timestep (plus variance).
        # self.poissonDistribution = poisson(self.poissonLambda)
        # print(self.poissonDistribution)
        self.name = name
        self.tau = tau
        self.time = 0
        self.parentPopulation = parentPop
        self.type = "PoissonNeuron"

        # Set up outputs
        if outputs is None:
            self.outputs = []
        else:
            self.outputs = outputs

        # Set up records
        self.vv = []
        self.spikeRecord = []

    def setRate(self, meanSpikesPerSecond):
        self.expectedMeanSpikeRate = meanSpikesPerSecond
        self.meanSpikeProbabilityPerMs = (self.expectedMeanSpikeRate / 10)
        self.poissonLambda = self.meanSpikeProbabilityPerMs * self.tau

    def step(self):
        self.time += self.tau
        # Evaluate Poisson source
        if poisson(self.poissonLambda, 1) > 0:
            # if self.poissonDistribution.rvs((1,))[0] > 0:
            self.spikeRecord.append(self.time)
            for axon in self.outputs:
                axon.enqueue()
            self.vv.append(40)
        else:
            self.vv.append(-60)
