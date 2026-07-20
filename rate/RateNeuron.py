from rate.CellTypes import CELL_TYPES

'''
Rate-based analogues of the spiking model's Neuron / PoissonNeuron.

RateNeuron replaces the Izhikevich point neuron with a single firing-rate
variable r that relaxes toward the cell type's steady-state transfer function:

    tau_m * dr/dt = -r + F(I),     F(I) = k * max(I - theta, 0) ** n

where I is the total input current (synaptic + diffuse serotonin + external),
and (k, theta, n, tau_m) come from CellTypes for the cell's type.

RateInputNeuron replaces the Poisson spike generator: it has no membrane, just
a directly-settable rate (the lambda), which its outbound axons read.

Both expose the small interface RatePopulation / the rate axons rely on:
a `.rate` attribute, `.outputs`, `.addOutput`, `.addInput`, `.parentPopulation`,
`.rateRecord`, and `.step()`.
'''


class RateNeuron:
    def __init__(self, tau, cellType, name, parentPop=None):
        if cellType not in CELL_TYPES:
            raise ValueError("Unknown cell type %r (expected one of %s)"
                             % (cellType, list(CELL_TYPES.keys())))
        self.tau = tau
        self.type = cellType
        self.name = name
        self.parentPopulation = parentPop

        params = CELL_TYPES[cellType]
        self.k = params["k"]
        self.theta = params["theta"]
        self.n = params["n"]
        self.tau_m = params["tau_m"]

        self.rate = 0.0
        self.time = 0.0

        # Inputs into total current I, mirroring the spiking Neuron:
        #   synapticInput  - refreshed every step from inbound rate axons
        #   diffuseCurrent - set when diffuse serotonin changes, persists
        #   externalInput  - optional constant injected current
        self.synapticInput = 0.0
        # Per-source-population synaptic current this step, used by the
        # leave-one-out ablation influence metric (RatePopulation.stepCells).
        self.synapticInputBySource = {}
        self.diffuseCurrent = 0.0
        self.externalInput = 0.0
        self.I = 0.0

        self.inputs = []
        self.outputs = []
        self.diffuseReceptors = []

        self.rateRecord = []

    def transfer(self, I):
        x = I - self.theta
        if x <= 0:
            return 0.0
        return self.k * (x ** self.n)

    def addInput(self, axon):
        self.inputs.append(axon)

    def addOutput(self, axon):
        self.outputs.append(axon)

    def addSynapticTransmission(self, current, sourcePopulation=None):
        self.synapticInput += current
        if sourcePopulation is not None:
            self.synapticInputBySource[sourcePopulation] = \
                self.synapticInputBySource.get(sourcePopulation, 0.0) + current

    def setInjectedCurrent(self, current):
        self.externalInput = float(current)

    def addDiffuseReceptor(self, receptor):
        self.diffuseReceptors.append(receptor)
        receptor.setTarget(self)
        self.recomputeDiffuseCurrent()

    def recomputeDiffuseCurrent(self):
        # Diffuse current = sum over somatic serotonin receptors of weight*level.
        self.diffuseCurrent = sum(r.current for r in self.diffuseReceptors)

    def step(self):
        self.time += self.tau
        self.I = self.externalInput + self.synapticInput + self.diffuseCurrent
        target = self.transfer(self.I)
        self.rate += self.tau * (target - self.rate) / self.tau_m
        if self.rate < 0.0:
            self.rate = 0.0
        self.rateRecord.append(self.rate)
        # NOTE: synapticInput / synapticInputBySource are NOT cleared here.
        # The population clears them (clearSynapticAccumulators) after computing
        # the ablation influence metric, which needs the per-source breakdown
        # that produced this step's self.I. Inbound axons refill them next step.

    def clearSynapticAccumulators(self):
        self.synapticInput = 0.0
        self.synapticInputBySource = {}


class RateInputNeuron:
    def __init__(self, tau, rate, name, parentPop=None):
        self.tau = tau
        self.type = "RateInput"
        self.name = name
        self.parentPopulation = parentPop
        self.rate = float(rate)
        self.time = 0.0
        self.inputs = []
        self.outputs = []
        self.rateRecord = []

    def setRate(self, rate):
        self.rate = float(rate)

    def addInput(self, axon):
        self.inputs.append(axon)

    def addOutput(self, axon):
        self.outputs.append(axon)

    def step(self):
        self.time += self.tau
        self.rateRecord.append(self.rate)
