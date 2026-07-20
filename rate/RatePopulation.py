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
        self.rateRecord = []

        # Leave-one-out ablation influence: for each source population S that
        # projects to this one, ablationInfluence[S] is a per-step time series
        # of sum_j [ F(I_j) - F(I_j - I_{S->j}) ] over this population's cells j
        # -- the causal effect of removing S's synaptic current on the commanded
        # firing rate. Set trackInfluence=False on populations you don't measure
        # to skip the extra transfer evaluations.
        self.trackInfluence = not isInput
        self.ablationInfluence = {}
        self._inboundSourcePops = None

    # ---- construction ----

    def addOutboundConnections(self, targetPopulation, weightFunction, axonalReceptorFactories=None):
        axonalReceptorFactories = axonalReceptorFactories or []
        if targetPopulation not in self.outboundAxonsByTargetPop:
            self.outboundAxonsByTargetPop[targetPopulation] = []
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

    def _inboundSources(self):
        if self._inboundSourcePops is None:
            self._inboundSourcePops = sorted(
                {a.sourcePopulation for a in self.inboundAxons if a.sourcePopulation is not None},
                key=lambda p: p.name)
        return self._inboundSourcePops

    def stepCells(self):
        for cell in self.cells:
            cell.step()
        self.rateRecord.append(sum(c.rate for c in self.cells) / len(self.cells))

        if self.trackInfluence and not self.isInput:
            # Leave-one-out ablation influence of each source population S on
            # this population: sum over cells j of F(I_j) - F(I_j - I_{S->j}).
            # Difference the commanded rate F(I) (not the membrane-filtered
            # rate) so it reflects the instantaneous causal effect. Iterate a
            # fixed source set so every source's series has equal length.
            for srcPop in self._inboundSources():
                total = 0.0
                for cell in self.cells:
                    iSrc = cell.synapticInputBySource.get(srcPop, 0.0)
                    if iSrc != 0.0:
                        total += cell.transfer(cell.I) - cell.transfer(cell.I - iSrc)
                self.ablationInfluence.setdefault(srcPop, []).append(total)

        if not self.isInput:
            for cell in self.cells:
                cell.clearSynapticAccumulators()

    def stepOutputs(self):
        for axon in self.outboundAxons:
            axon.step()

    # ---- plasticity helpers ----

    def outboundAxonsTo(self, targetPopulation):
        return self.outboundAxonsByTargetPop.get(targetPopulation, [])
