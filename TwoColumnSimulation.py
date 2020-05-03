import sys
from absl import flags
from absl import logging
from scipy import *
from TwoColumnNetwork import *

FLAGS = flags.FLAGS

flags.DEFINE_integer("steps_between_timing_debug", 10,
                     "Number of steps to log progress after")


class TwoColumnSimulation():
    def __init__(self, params):
        self.params = params
        self.tau = self.params["tau"]  # In ms
        self.tspan1 = arange(0, self.params["maxTime"], self.tau)
        self.tspan2 = arange(
            self.params["maxTime"],
            self.params["maxTime"] * 2,
            self.tau)
        self.tspan3 = arange(
            self.params["maxTime"] * 2,
            self.params["maxTime"] * 3,
            self.tau)
        self.tspan4 = arange(
            self.params["maxTime"] * 3,
            self.params["maxTime"] * 4,
            self.tau)
        self.network = TwoColumnNetwork(
            self.tau, self, self.params, "Test Network1")

    def run(self):
        # First Portion, full input
        logging.info("Phase One")
        for t in self.tspan1:
            if t % FLAGS.steps_between_timing_debug == 0:
                logging.debug("Phase 1 step: %d" % t)
            self.network.step()

        self.aEndOfFirstPortionSpikes = [
            len(c.spikeRecord) for c in self.network.populations["pyramidalsA"].cells]

        # Second portion, sensory deprivation
        logging.info("Phase Two")
        for inputCell in self.network.populations["InputA"].cells:
            inputCell.poissonLambda = 0
        for t in self.tspan2:
            if t % FLAGS.steps_between_timing_debug == 0:
                logging.debug("Phase 2 step: %d" % t)
            self.network.step()

        self.aEndOfSecondPortionSpikes = [
            len(c.spikeRecord) for c in self.network.populations["pyramidalsA"].cells]

        # Third portion, Serotonin and Plasticity
        logging.info("Phase Three")
        weightMatrixPriorBA = zeros(
            [self.params["popCount"], self.params["popCount"]])
        for x in range(len(self.network.populations["InputB"].cells)):
            for y in range(len(self.network.populations["pyramidalsA"].cells)):
                for source in self.network.populations["InputB"].cells[x].outputs:
                    if source.target == self.network.populations["pyramidalsA"].cells[y]:
                        weightMatrixPriorBA[x,
                                            y] = source.postSynapticReceptors[0].weight

        figure()
        pcolor(weightMatrixPriorBA)
        colorbar()
        title('Prior Weights from Input B to Pyramidals A')

        # Increase 5HT
        transmittersA = {}
        transmittersA["5HT2A"] = 30
        transmittersA["5HT1A"] = 30
        self.network.setSerotoninA(transmittersA)

        # Turn on plasticity
        for x in range(len(self.network.populations["InputB"].cells)):
            for y in range(len(self.network.populations["pyramidalsA"].cells)):
                for source in self.network.populations["InputB"].cells[x].outputs:
                    # print(source.target.name, sim.network.populations["pyramidalsA"].cells[y].name)
                    if source.target == self.network.populations["pyramidalsA"].cells[y]:
                        source.postSynapticReceptors[0].plasticity = True
                        source.postSynapticReceptors[0].c_p = 180.3

        for t in self.tspan3:
            if t % FLAGS.steps_between_timing_debug == 0:
                logging.debug("Phase 3 step: %d" % t)
            self.network.step()

        self.aEndOfThirdPortionSpikes = [
            len(c.spikeRecord) for c in self.network.populations["pyramidalsA"].cells]

        weightMatrixPostBA = zeros(
            [self.params["popCount"], self.params["popCount"]])
        for x in range(len(self.network.populations["InputB"].cells)):
            for y in range(len(self.network.populations["pyramidalsA"].cells)):
                for source in self.network.populations["InputB"].cells[x].outputs:
                    if source.target == self.network.populations["pyramidalsA"].cells[y]:
                        weightMatrixPostBA[x,
                                           y] = source.postSynapticReceptors[0].weight

        figure()
        pcolor(weightMatrixPostBA)
        colorbar()
        title('Posterior Weights from Input B to Pyramidals A')

        # Fourth portion, remapped functionality.
        logging.info("Phase Four")
        # Turn off plasticity
        for x in range(len(self.network.populations["InputB"].cells)):
            for y in range(len(self.network.populations["pyramidalsA"].cells)):
                for source in self.network.populations["InputB"].cells[x].outputs:
                    # print(source.target.name, sim.network.populations["pyramidalsA"].cells[y].name)
                    if source.target == self.network.populations["pyramidalsA"].cells[y]:
                        source.postSynapticReceptors[0].plasticity = False

        transmittersA = {}
        transmittersA["5HT2A"] = 40
        transmittersA["5HT1A"] = 40
        self.network.setSerotoninA(transmittersA)
        for t in self.tspan4:
            if t % FLAGS.steps_between_timing_debug == 0:
                logging.debug("Phase 4 step: %d" % t)
            self.network.step()

        # for t in self.tspan:
        #     # if t % 10 == 0:
        #     print("Time: ", t)
        #     self.network.step()

        self.aEndOfFourthPortionSpikes = [
            len(c.spikeRecord) for c in self.network.populations["pyramidalsA"].cells]

    def plotColumns(self):
        inputAVoltages = [
            c.vv for c in self.network.populations["InputA"].cells]
        inputBVoltages = [
            c.vv for c in self.network.populations["InputB"].cells]

        aPyramidalVoltages = [
            c.vv for c in self.network.populations["pyramidalsA"].cells]
        aFSVoltages = [
            c.vv for c in self.network.populations["fastSpikingsA"].cells]
        aLTSVoltages = [
            c.vv for c in self.network.populations["lowThresholdsA"].cells]

        bPyramidalVoltages = [
            c.vv for c in self.network.populations["pyramidalsB"].cells]
        bFSVoltages = [
            c.vv for c in self.network.populations["fastSpikingsB"].cells]
        bLTSVoltages = [
            c.vv for c in self.network.populations["lowThresholdsB"].cells]

        aPyramidalSpikes = [len(c.spikeRecord)
                            for c in self.network.populations["pyramidalsA"].cells]
        bPyramidalSpikes = [len(c.spikeRecord)
                            for c in self.network.populations["pyramidalsB"].cells]

        print("A Spikes: ", sum(aPyramidalSpikes))
        print("B Spikes: ", sum(bPyramidalSpikes))

        # Sliding Windowed Spike Rates
        figure()
        subplot(4, 1, 1)
        plot(self.network.populations["InputA"].rateRecord)
        title('A Input')

        subplot(4, 1, 2)
        plot(self.network.populations["pyramidalsA"].rateRecord)
        title('A Pyramidal Cells')

        subplot(4, 1, 3)
        plot(self.network.populations["fastSpikingsA"].rateRecord)
        title('A FS Cells')

        subplot(4, 1, 4)
        plot(self.network.populations["lowThresholdsA"].rateRecord)
        title('A LTS Cells')

        figure()
        subplot(4, 1, 1)
        plot(self.network.populations["InputB"].rateRecord)
        title('B Input')

        subplot(4, 1, 2)
        plot(self.network.populations["pyramidalsB"].rateRecord)
        title('B Pyramidal Cells')

        subplot(4, 1, 3)
        plot(self.network.populations["fastSpikingsB"].rateRecord)
        title('B FS Cells')

        subplot(4, 1, 4)
        plot(self.network.populations["lowThresholdsB"].rateRecord)
        title('B LTS Cells')

        figure()
        # title("Column A")
        subplot(4, 1, 1)
        pcolor(inputAVoltages, vmin=-100, vmax=60)
        colorbar()
        title('A Input')

        subplot(4, 1, 2)
        pcolor(aPyramidalVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["pyramidalsA"].rateRecord, color='White')
        colorbar()
        title('A Pyramidal Cells')

        subplot(4, 1, 3)
        pcolor(aFSVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["fastSpikingsA"].rateRecord, color='White')
        colorbar()
        title('A FS Cells')

        subplot(4, 1, 4)
        pcolor(aLTSVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["lowThresholdsA"].rateRecord, color='White')
        colorbar()
        title('A LTS Cells')

        figure()
        subplot(4, 1, 1)
        pcolor(inputBVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["InputB"].rateRecord, color='White')
        colorbar()
        title('B Input')

        subplot(4, 1, 2)
        pcolor(bPyramidalVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["pyramidalsB"].rateRecord, color='White')
        colorbar()
        title('B Pyramidal Cells')

        subplot(4, 1, 3)
        pcolor(bFSVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["fastSpikingsB"].rateRecord, color='White')
        colorbar()
        title('B FS Cells')

        subplot(4, 1, 4)
        pcolor(bLTSVoltages, vmin=-100, vmax=60)
        # plot(self.network.populations["lowThresholdsB"].rateRecord, color='White')
        colorbar()
        title('B LTS Cells')

        numax = len(self.network.populations["InputB"].outboundAxons)
        timeSpan = len(self.network.populations["InputB"].rateRecord)
        baSpikeFailures = np.zeros((numax, timeSpan))
        axNum = 0
        for ax in self.network.populations["InputB"].outboundAxons:
            baSpikeFailures[axNum, :] = ax.spikeFailures
            axNum += 1

        figure()
        pcolor(baSpikeFailures, cmap="Greys")

        figure()
        plot(sum(baSpikeFailures, axis=0))

        savefig("fig1.png")
