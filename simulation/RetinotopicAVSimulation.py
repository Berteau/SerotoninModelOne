from numpy import arange
from network.RetinotopicAVNetwork import RetinotopicAVNetwork

'''
Phase-based simulation for RetinotopicAVNetwork, following the same
progression as TwoColumnSimulation's phase1..phase4:

  phase1: full input (video + audio both driving normally)
  phase2: loss of visual input - the lesioned columns are muted. Audio keeps
          driving normally throughout every phase.
  phase3: serotonin rises in the affected region and plasticity is switched
          on for the "remapping" connections that carry spared input into the
          deprived columns, allowing them to potentiate.
  phase4: plasticity switches back off and serotonin returns to baseline -
          the deprived columns are now driven by the remapped pathway, and
          that stays stable for further running.

Two lesion modes are supported:
  "scotoma" - a contiguous block of visual columns is silenced. Remapping is
              carried by lateral pyramidal -> pyramidal connections from the
              spared grid-neighbor columns into the lesioned block, mirroring
              classic cortical remapping around a focal visual field lesion.
  "full"    - every visual column is silenced. There are no spared visual
              neighbors, so remapping is instead carried by the cross-modal
              audio -> visual connections at the two grid edges, mirroring
              cross-modal reorganization following complete visual loss.
'''

class RetinotopicAVSimulation():
    def __init__(self, params, videoSource, audioSource, lesionMode="scotoma"):
        if lesionMode not in ("scotoma", "full"):
            raise ValueError("lesionMode must be 'scotoma' or 'full', got %r" % lesionMode)
        self.params = params
        self.tau = params["tau"]
        self.videoSource = videoSource
        self.audioSource = audioSource
        self.lesionMode = lesionMode
        self.frameDurationMs = params["frameDurationMs"]
        self.network = RetinotopicAVNetwork(self.tau, self, params, "RetinotopicAVNetwork_" + lesionMode)

        gridRows = self.network.gridRows
        gridCols = self.network.gridCols
        if lesionMode == "full":
            self.lesionedColumns = [(r, c) for r in range(gridRows) for c in range(gridCols)]
        else:
            scotomaRows = params.get("scotomaRows", range(gridRows // 2 - 1, gridRows // 2 + 1))
            scotomaCols = params.get("scotomaCols", range(gridCols // 2 - 1, gridCols // 2 + 1))
            self.lesionedColumns = [(r, c) for r in scotomaRows for c in scotomaCols]
        self.lesionedSet = set(self.lesionedColumns)

        self.spareNeighborColumns = sorted({
            neighbor
            for (r, c) in self.lesionedColumns
            for neighbor in self.network.gridNeighbors(r, c)
            if neighbor not in self.lesionedSet
        })

        self.remappingAxons = self._collectRemappingAxons()
        self.weightsPrior = None
        self.weightsPost = None

        self._lastFrameIndex = None
        self._lastAudioState = None

    def _collectRemappingAxons(self):
        # The specific connections that carry spared activity into the deprived
        # columns - these are the ones whose weights should grow during phase3
        # if remapping is actually happening.
        axons = []
        if self.lesionMode == "scotoma":
            for (nr, nc) in self.spareNeighborColumns:
                neighborName = self.network.visualColumnNames[(nr, nc)]
                for (lr, lc) in self.network.gridNeighbors(nr, nc):
                    if (lr, lc) in self.lesionedSet:
                        lesionedName = self.network.visualColumnNames[(lr, lc)]
                        axons.extend(self.network.getAxonsBetween("pyramidals" + neighborName, "pyramidals" + lesionedName))
        else:
            for r in range(self.network.gridRows):
                for c in (0, self.network.gridCols - 1):
                    visualName = self.network.visualColumnNames[(r, c)]
                    audioIndex = self.network.leftEdgeBandForRow[r] if c == 0 else self.network.rightEdgeBandForRow[r]
                    audioName = self.network.audioColumnNames[int(audioIndex)]
                    axons.extend(self.network.getAxonsBetween("Input" + audioName, "pyramidals" + visualName))
        return axons

    def _affectedColumnNames(self):
        # Columns whose serotonin level rises during phase3: the lesioned
        # columns themselves, plus (for the scotoma case) the spared
        # neighbors that are being asked to take over.
        coords = list(self.lesionedColumns)
        if self.lesionMode == "scotoma":
            coords = coords + self.spareNeighborColumns
        return [self.network.visualColumnNames[rc] for rc in coords]

    def _driveInputs(self, t, muteVisual):
        # Only push new rates into the input populations when the underlying
        # video frame / beep state actually changes, instead of on every tau
        # step - the network step itself dominates runtime, so this avoids
        # thousands of redundant setRate() calls across a phase.
        frameIndex = int(t // self.frameDurationMs)
        if frameIndex != self._lastFrameIndex:
            self._lastFrameIndex = frameIndex
            self.network.setVisualInputRates(self.videoSource.rateFrameAt(frameIndex))
            if muteVisual:
                self._muteLesionedColumns()

        audioState = self.audioSource.stateAtTime(t)
        if audioState != self._lastAudioState:
            self._lastAudioState = audioState
            self.network.setAudioInputRates(self.audioSource.rateVectorAtTime(t))

    def _muteLesionedColumns(self):
        for (r, c) in self.lesionedColumns:
            self.network.setVisualInputRate(r, c, 0)

    def runPhase(self, phase, muteVisual):
        maxTime = self.params["phaseDurationMs"]
        start = maxTime * (phase - 1)
        end = maxTime * phase
        tspan = arange(start, end, self.tau)
        # Force the first step of every phase to re-apply rates (and muting,
        # if this phase requires it), since the frame/beep index alone can't
        # tell a phase transition apart from an unchanged frame.
        self._lastFrameIndex = None
        self._lastAudioState = None
        for t in tspan:
            self._driveInputs(t, muteVisual)
            self.network.step()

    def phase1(self):
        self.runPhase(1, muteVisual=False)

    def phase2(self):
        self.runPhase(2, muteVisual=True)

    def phase3(self):
        # NOTE: axon.weight is a static snapshot fixed at construction time.
        # The live, plasticity-updated weight lives on the postsynaptic
        # receptor (see GlutGSDReceptor.boutonSpike / postSynapticSpikeFeedback),
        # exactly as TwoColumnSimulation reads source.postSynapticReceptors[0].weight.
        self.weightsPrior = [axon.postSynapticReceptors[0].weight for axon in self.remappingAxons]

        raisedTransmitters = {"5HT2A": self.params["remapSerotoninLevel"], "5HT1A": self.params["remapSerotoninLevel"]}
        self.network.setSerotoninForColumns(self._affectedColumnNames(), raisedTransmitters)

        for axon in self.remappingAxons:
            axon.postSynapticReceptors[0].plasticity = True
            axon.postSynapticReceptors[0].c_p = self.params["remapPlasticityCp"]

        self.runPhase(3, muteVisual=True)

        self.weightsPost = [axon.postSynapticReceptors[0].weight for axon in self.remappingAxons]

    def phase4(self):
        for axon in self.remappingAxons:
            axon.postSynapticReceptors[0].plasticity = False

        baselineTransmitters = {"5HT2A": self.params["serotoninLevelV"], "5HT1A": self.params["serotoninLevelV"]}
        self.network.setSerotoninForColumns(self._affectedColumnNames(), baselineTransmitters)

        self.runPhase(4, muteVisual=True)

    def run(self):
        self.phase1()
        self.phase2()
        self.phase3()
        self.phase4()
