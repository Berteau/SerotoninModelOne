from Neuron import *
from Axon import *
from DiffuseReceptorFactory import *
from SomaticSerotoninReceptor import *
from AxonalSerotoninReceptor import *
import GABAGSDReceptor
import GlutGSDReceptor
# from SerotoninReceptorAxon import *

# from NetPlotter import *
# from pylab import *
from random import random, gauss

# TODO: Write utilities to manage this at the population level, like a population object that holds the neurons and the connections within population, and all the inter-population connections that originate in it?
# Then I can just create those, pass them a generator/connection pattern, hook them up to each other and inputs, and call their "SetDiffuse5htLevels" and step() methods.
# They could even manage activity summaries and plot generation of
# activity over time.


class Population():
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
        self.time = 0
        self.tau = tau
        self.cellParameters = neuronParameters
        self.popCount = popCount
        self.diffuseTransmitters = diffuseTransmitters
        self.diffuseSomaReceptorFactories = diffuseSomaReceptorFactories
        self.parentNetwork = parentNetwork
        self.rateRecord = []
        self.influenceRecord = {}

        # Generate Cells
        self.cells = [
            Neuron(
                self.tau,
                self.cellParameters,
                self.name +
                "." +
                self.cellParameters["type"] +
                "." +
                str(i)) for i in range(popCount)]
        for cell in self.cells:
            for receptorFactory in self.diffuseSomaReceptorFactories:
                tempReceptorLevel = 0.0
                cell.addDiffuseReceptor(
                    receptorFactory.constructReceptor(tempReceptorLevel))

        # Prepare for connections
        self.outboundAxons = []
        self.outboundAxonsByTargetPop = {}
        self.inboundAxons = []
        self.inboundAxonsByTargetPop = {}

        # Outbound connections will be created later via a call to addOutboundConnections(), even if they are entirely internal to this population.
        # Inbound connections will call registerInboundConnection() as they are created, either via this population or via another.
        # Note: We handle internal connections as just outbound from and
        # inbound to the same population

        # Set Diffuse Transmitter Levels
        self.setDiffuseTransmitters(diffuseTransmitters)

    def getName(self):
        return self.name

    def setInjectedCurrent(self, current):
        if isinstance(current, (int, float)):
            for cell in self.cells:
                cell.setInjectedCurrent(current)
        else:
            raise ValueError(
                "Error: setInjectedCurrent requires an int or float type, representing current in picoamperes")

    def addOutboundConnections(
            self,
            targetPopulation,
            targetingFunction,
            diffuseAxonReceptorFactoriesProximal=None,
            diffuseAxonReceptorFactoriesDistal=None):
        # If the population is not in the dict, add it and map it to a new
        # list.
        if targetPopulation not in self.outboundAxonsByTargetPop.keys():
            self.outboundAxonsByTargetPop[targetPopulation] = []
            for i in range(self.time):
                self.outboundAxonsByTargetPop[targetPopulation].append(NAN)

        if diffuseAxonReceptorFactoriesProximal is None:
            diffuseAxonReceptorFactoriesProximal = []
        if diffuseAxonReceptorFactoriesDistal is None:
            diffuseAxonReceptorFactoriesDistal = []
        for source in self.cells:
            for target in targetPopulation.cells:
                weight = targetingFunction(source, target)
                if weight is not None:
                    tempAxon = None
                    # Create connection
                    tempAxon = Axon(self.tau, weight, source, target)
                    # Add proximal diffuse receptors
                    for diffuseAxonReceptorFactory in diffuseAxonReceptorFactoriesProximal:  # Loop through our receptor factories
                        # Construct a receptor and add it, using the level of
                        # diffuse transmitter associated with that receptor's
                        # type string as obtained from the factory
                        tempAxon.addProximalDiffuseAxonReceptor(
                            diffuseAxonReceptorFactory.constructReceptor(
                                self.getDiffuseTransmitters()[
                                    diffuseAxonReceptorFactory.getTypeString()],
                                target=tempAxon))
                    # Add distal diffuse receptors
                    for diffuseAxonReceptorFactory in diffuseAxonReceptorFactoriesDistal:  # Loop through our receptor factories
                        # Construct a receptor and add it, using the level of
                        # diffuse transmitter associated with that receptor's
                        # type string as obtained from the factory
                        tempAxon.addDistalDiffuseAxonReceptor(
                            diffuseAxonReceptorFactory.constructReceptor(
                                targetPopulation.getDiffuseTransmitters()[
                                    diffuseAxonReceptorFactory.getTypeString()], target=tempAxon))
                    self.outboundAxons.append(tempAxon)
                    self.outboundAxonsByTargetPop[targetPopulation].append(
                        tempAxon)
                    self.influenceRecord[targetPopulation] = []
                    targetPopulation.registerInboundConnection(tempAxon)

    # Do I need inbound connections handled here?  Should a population have
    # some list of what is inbound?  I currently can't think of a reason.
    # Adding a list just in case, since it is minimal overhead.
    def registerInboundConnection(self, axon):
        if isinstance(axon, (Axon)):
            self.inboundAxons.append(axon)
            sourcePop = axon.source.parentPopulation
            if sourcePop not in self.inboundAxonsByTargetPop.keys():
                self.inboundAxonsByTargetPop[sourcePop] = []
            self.inboundAxonsByTargetPop[sourcePop].append(axon)

    def setDiffuseTransmitters(self, diffuseTransmitters):
        if isinstance(diffuseTransmitters, dict):
            self.diffuseTransmitters = diffuseTransmitters
            # Update somatic receptors
            for cell in self.cells:
                cell.updateDiffuseTransmitters(self.diffuseTransmitters)
            # Update proximal receptors on outbound axons
            for outboundAxon in self.outboundAxons:
                outboundAxon.updateProximalDiffuseTransmitters(
                    self.diffuseTransmitters)
            # Update distal receptors in inbound axons
            for inboundAxon in self.inboundAxons:
                inboundAxon.updateDistalDiffuseTransmitters(
                    self.diffuseTransmitters)
        else:
            raise ValueError(
                "Error: setDiffuseTransmitters requires a dictionary type")

    def getDiffuseTransmitters(self):
        return self.diffuseTransmitters

    def getSpikeRatePerSecond(self, window):
        rate = 0
        stepsPerSecond = 1000
        windowSize = window[1] - window[0]
        scaleFactor = stepsPerSecond / windowSize
        for cell in self.cells:
            for spike in cell.spikeRecord:
                if window[0] < spike < window[1]:
                    rate += 1
        return rate * scaleFactor

    def stepCells(self):
        self.time += self.tau
        for cell in self.cells:
            cell.step()
        self.rateRecord.append(
            self.getSpikeRatePerSecond([self.time - 50, self.time]))

        for targetPop in self.outboundAxonsByTargetPop.keys():
            tempInfluence = 0
            for axonOut in self.outboundAxonsByTargetPop[targetPop]:
                if axonOut.weight > 0 and len(
                        axonOut.postSynapticReceptors[0].driveFactor) > 0:
                    tempInfluence += axonOut.postSynapticReceptors[0].driveFactor[-1]
            self.influenceRecord[targetPop].append(tempInfluence)

    def stepOutputs(self):
        for outboundAxon in self.outboundAxons:
            outboundAxon.step()

    def generateStats(self):
        totalSpikes = sum([len(cell.spikeRecord) for cell in self.cells])
        # TODO: Add spikeRatePerSecond as totalSpikes over time, which is
        # gleaned from the Run object which is the parent of the population's
        # parentNetwork object.
