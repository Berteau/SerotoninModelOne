from rate.RateNeuron import RateNeuron, RateInputNeuron
from rate.RateAxon import RateAxon

'''
Rate-based analogue of the spiking model's Population, at per-neuron
granularity. Holds a set of RateNeurons (or RateInputNeurons for an input
population) and manages the axons that originate in it.

Interface mirrors the spiking Population closely enough that a network builder
can drive either model with the same connection-construction pattern:
  - addOutboundConnections(targetPop, weightFn, axonalReceptorFactories)
  - setDiffuseTransmitters(dict)  /  setRate(rate)  (input pops)
  - stepCells()  then  stepOutputs()   each timestep

Records kept per step:
  - rateRecord: mean firing rate across the population
  - influenceRecord[targetPop]: summed cross-modal driving factor (phi) from
    this population's excitatory axons onto each target population, the rate
    analogue of the paper's "relative influence" metric.
'''


class RatePopulation:
    def __init__(self, tau, cellType, popCount, diffuseSomaReceptorFactories,
                 diffuseTransmitters, parentNetwork, name,
                 isInput=False, inputRate=0.0):
        self.tau = tau
        self.cellType = cellType
        self.name = name
        self.popCount = popCount
        self.parentNetwork = parentNetwork
        self.isInput = isInput

        self.diffuseSomaReceptorFactories = diffuseSomaReceptorFactories or []
        self.diffuseTransmitters = dict(diffuseTransmitters or {})

        if isInput:
            self.cells = [RateInputNeuron(tau, inputRate, "%s.in.%d" % (name, i), parentPop=self)
                          for i in range(popCount)]
        else:
            self.cells = [RateNeuron(tau, cellType, "%s.%s.%d" % (name, cellType, i), parentPop=self)
                          for i in range(popCount)]
            for cell in self.cells:
                for factory in self.diffuseSomaReceptorFactories:
                    level = self.diffuseTransmitters.get(factory.getTypeString(), 0.0)
                    cell.addDiffuseReceptor(factory.constructReceptor(level))

        self.outboundAxons = []
        self.outboundAxonsByTargetPop = {}
        self.inboundAxons = []
        self.influenceRecord = {}
        self.rateRecord = []

    # ---- construction ----

    def addOutboundConnections(self, targetPopulation, weightFunction, axonalReceptorFactories=None):
        axonalReceptorFactories = axonalReceptorFactories or []
        if targetPopulation not in self.outboundAxonsByTargetPop:
            self.outboundAxonsByTargetPop[targetPopulation] = []
            self.influenceRecord[targetPopulation] = []
        for source in self.cells:
            for target in targetPopulation.cells:
                weight = weightFunction(source, target)
                if weight is None:
                    continue
                axon = RateAxon(self.tau, weight, source, target)
                for factory in axonalReceptorFactories:
                    level = targetPopulation.diffuseTransmitters.get(factory.getTypeString(), 0.0)
                    axon.addAxonalReceptor(factory.constructReceptor(level))
                self.outboundAxons.append(axon)
                self.outboundAxonsByTargetPop[targetPopulation].append(axon)
                targetPopulation.inboundAxons.append(axon)

    # ---- runtime controls ----

    def setRate(self, rate):
        for cell in self.cells:
            cell.setRate(rate)

    def setDiffuseTransmitters(self, diffuseTransmitters):
        self.diffuseTransmitters = dict(diffuseTransmitters)
        # Update somatic receptors on this population's neurons.
        if not self.isInput:
            for cell in self.cells:
                for receptor in cell.diffuseReceptors:
                    receptor.setLevel(self.diffuseTransmitters.get(receptor.getTypeString(), 0.0))
        # Update axonal receptors on inbound axons (distal serotonin sees the
        # target region's transmitter levels).
        for axon in self.inboundAxons:
            for receptor in axon.axonalReceptors:
                receptor.setLevel(self.diffuseTransmitters.get(receptor.getTypeString(), 0.0))

    # ---- per-step ----

    def stepCells(self):
        for cell in self.cells:
            cell.step()
        self.rateRecord.append(sum(c.rate for c in self.cells) / len(self.cells))
        for targetPop, axons in self.outboundAxonsByTargetPop.items():
            self.influenceRecord[targetPop].append(sum(a.tempDriveFactor for a in axons))

    def stepOutputs(self):
        for axon in self.outboundAxons:
            axon.step()

    # ---- plasticity helpers ----

    def outboundAxonsTo(self, targetPopulation):
        return self.outboundAxonsByTargetPop.get(targetPopulation, [])
