from PoissonNeuron import *
from Population import *


class PoissonPopulation(Population):
    def __init__(
            self,
            tau,
            neuronParameters,
            popCount,
            diffuseSomaReceptorFactories,
            diffuseTransmitters,
            parentNetwork,
            name):
        self.name = name
        self.tau = tau
        self.time = 0
        self.cellParameters = neuronParameters
        self.popCount = popCount
        self.meanSpikesPerSecond = neuronParameters["spikeRate"]
        self.diffuseTransmitters = diffuseTransmitters
        self.diffuseSomaReceptorFactories = diffuseSomaReceptorFactories
        self.parentNetwork = parentNetwork
        self.rateRecord = []
        self.influenceRecord = {}

        # Generate Cells
        self.cells = [
            PoissonNeuron(
                self.tau,
                self.meanSpikesPerSecond,
                self.name +
                "." +
                self.cellParameters["type"] +
                "." +
                str(i),
                parentPop=self) for i in range(popCount)]

        # Prepare for connections
        self.outboundAxons = []
        self.outboundAxonsByTargetPop = {}
        self.inboundAxons = []
        self.inboundAxonsByTargetPop = {}

        # Outbound connections will be created later via a call to addOutboundConnections(), even if they are entirely internal to this population.
        # Inbound connections will call registerInboundConnection() as they are created, either via this population or via another.
        # Note: We handle internal connections as just outbound from and
        # inbound to the same population
