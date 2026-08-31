from rate.RateNeuron import RateNeuron

'''
V1 (primary visual area) pyramidal neuron implementing the "normalized mixture
of top-down and bottom-up" from Sulfaro, Robinson & Carlson (2023), "Modelling
perception as a hierarchical competition differentiates imagined, veridical, and
hallucinated percepts" (Neuroscience of Consciousness 2023(1):niad018).

------------------------------------------------------------------------------
The Sulfaro rule
------------------------------------------------------------------------------
Sulfaro et al. model perception as a stack of layers, each holding a
representation matrix r_l. Every unclamped layer updates each timestep as a
*weighted average* of the layer below (ascending / bottom-up), the layer above
(descending / top-down), and itself, all at the previous timestep (their
Fig. 1b, Methods):

                w_{l-1}.pool(r_{l-1}) + w_l.r_l + w_{l+1}.up(r_{l+1})
     r_l(t)  =  ----------------------------------------------------------
                             w_{l-1} + w_l + w_{l+1}

The defining feature is the denominator: the combination is normalized by the
*sum of the stream weights*, making it a convex combination (a weighted mean),
NOT a plain additive sum. The feedforward-to-feedback weighting ratio
w_{l-1}/w_{l+1} is the single knob that governs perception: > 1 (bottom-up
dominant) yields veridical perception; < 1 (top-down dominant) pushes the layer
toward the hallucinatory / imagined regime (their Fig. 3). Note this is a
fixed-weight convex combination normalized by constant weights -- it is NOT the
activity-dependent divisive normalization of Carandini & Heeger.

------------------------------------------------------------------------------
Why we apply the normalization at the INPUT-CURRENT level (not the rate level)
------------------------------------------------------------------------------
Sulfaro operate on abstract representation matrices with no spike/rate/current
distinction: r is simultaneously "the activity" and "what gets averaged". Our
model is per-neuron and biophysically grounded: each cell has a total input
current I and a firing rate r = F(I), where F is the fitted threshold-power-law
transfer function. There are therefore two places we could insert the
normalized mixture, and they are NOT equivalent because F is nonlinear:

  (a) current level: normalize the input CURRENTS, then pass the mixed current
      through the cell's transfer function once --
          I_V1 = (w_bu.I_bu + w_self.I_self + w_td.I_td) / (w_bu + w_self + w_td)
          r    = F(I_V1)

  (b) rate level: compute each stream's rate separately and average the RATES --
          r_V1 = (w_bu.F(I_bu) + w_self.F(I_self) + w_td.F(I_td)) / (sum w)

We use (a), current level, chosen deliberately for three reasons:

  1. It preserves the cell's transfer nonlinearity as a single, physical
     input-output stage. Option (b) would apply F independently to each stream
     and then average the outputs, which double-counts the threshold/expansive
     nonlinearity and discards the fact that a real neuron sums its synaptic
     currents at the soma before spiking. There is one spike-generating
     mechanism per cell, so there should be one F per cell.

  2. Sulfaro themselves state (Methods) that their weighted average "is
     approximately equivalent to the biologically plausible additive
     combination of neural inputs, followed by a decay of activity proportional
     to the strength of activation." Additive combination *of neural inputs*
     (currents) followed by decay is exactly current-level summation feeding a
     leaky rate variable -- i.e. option (a). So current-level normalization is
     arguably closer to the biological process they invoke to justify the rule
     than the literal matrix-of-rates form.

  3. Serotonin's action falls out of machinery we already validated. Sulfaro's
     own neurochemical bridge (their "Serotonin and acetylcholine..." section;
     Seillier et al. 2017; Michaiel et al. 2019) is that 5HT2A dampens V1's gain
     to bottom-up sensory input -- i.e. it lowers w_{l-1}. Because our bottom-up
     stream is a synaptic current, a serotonergic gain on the bottom-up WEIGHT
     (self.bottomUpGain) reuses the transmission-gain concept from RateAxon
     rather than introducing a new bespoke mechanism.

The cost of (a) is that the leave-one-out ablation influence metric
(RatePopulation.stepCells) assumes I is additive in the per-source currents;
under normalization the marginal effect of a source is not its raw current, so
that metric is disabled on mixture populations (trackInfluence=False) rather
than reported with a wrong additive counterfactual. A normalization-aware
influence metric can be derived later if needed.

------------------------------------------------------------------------------
Stream assignment
------------------------------------------------------------------------------
Each inbound source population is tagged with the stream it belongs to
(bottom-up / top-down / self) via assignStream(). RateAxon already attributes
every injected current to its source population (synapticInputBySource), so we
regroup those per-source currents into the three streams each step. Sources with
no explicit assignment default to bottom-up (ascending sensory drive), matching
Sulfaro's treatment of the lowest layer as the sensory entryway.

Somatic diffuse currents (self.diffuseCurrent, e.g. somatic 5HT1A/5HT2A) and any
external injected current stay ADDITIVE and outside the mixture: they are
soma-level offsets/gains, not one of the three representational streams being
averaged.
'''

STREAM_BOTTOMUP = "bottomup"
STREAM_TOPDOWN = "topdown"
STREAM_SELF = "self"
STREAMS = (STREAM_BOTTOMUP, STREAM_TOPDOWN, STREAM_SELF)


class NormalizedMixtureNeuron(RateNeuron):
    def __init__(self, tau, cellType, name, parentPop=None,
                 w_bottomup=1.0, w_topdown=1.0, w_self=1.0):
        super().__init__(tau, cellType, name, parentPop)
        # Stream weights. Defaults are all 1.0 -> Sulfaro's neutral scheme
        # (w_{l-1} = w_l = w_{l+1} = 1). The feedforward:feedback ratio
        # w_bottomup / w_topdown is the perception knob.
        self.streamWeights = {
            STREAM_BOTTOMUP: float(w_bottomup),
            STREAM_TOPDOWN: float(w_topdown),
            STREAM_SELF: float(w_self),
        }
        # sourcePopulation object -> stream name. Filled by the network builder.
        self.sourceStream = {}
        # Multiplicative gain applied to the bottom-up weight only (serotonergic
        # 5HT2A hook, per Sulfaro's neurochemical bridge). 1.0 = unmodulated.
        self.bottomUpGain = 1.0

        # Last step's per-stream RAW currents (before weighting/normalization),
        # recorded for metrics/plots and the normalization self-check.
        self.lastStreamCurrent = {STREAM_BOTTOMUP: 0.0, STREAM_TOPDOWN: 0.0, STREAM_SELF: 0.0}
        # Last step's RAW current per SOURCE population. The bottom-up stream
        # lumps visual and cross-modal audio together (both are ascending, so
        # both carry the bottom-up weight); this finer breakdown lets an
        # experiment read the audio (cross-modal) drive separately as the
        # remapping readout without changing the mixture's stream weighting.
        self.lastSourceCurrent = {}

    def assignStream(self, sourcePopulation, stream):
        if stream not in STREAMS:
            raise ValueError("Unknown stream %r (expected one of %s)" % (stream, STREAMS))
        self.sourceStream[sourcePopulation] = stream

    def effectiveStreamWeights(self):
        # Bottom-up weight is scaled by the serotonergic gain; the others are
        # used as-is. Returned as (w_bu, w_td, w_self).
        return (self.streamWeights[STREAM_BOTTOMUP] * self.bottomUpGain,
                self.streamWeights[STREAM_TOPDOWN],
                self.streamWeights[STREAM_SELF])

    def _mixtureCurrent(self):
        # Regroup this step's per-source synaptic currents into the three
        # streams, then take the weight-normalized convex combination.
        streamCurrent = {STREAM_BOTTOMUP: 0.0, STREAM_TOPDOWN: 0.0, STREAM_SELF: 0.0}
        for srcPop, current in self.synapticInputBySource.items():
            stream = self.sourceStream.get(srcPop, STREAM_BOTTOMUP)
            streamCurrent[stream] += current
        self.lastStreamCurrent = streamCurrent
        self.lastSourceCurrent = dict(self.synapticInputBySource)
        w_bu, w_td, w_self = self.effectiveStreamWeights()
        wsum = w_bu + w_td + w_self
        if wsum <= 0.0:
            return 0.0
        return (w_bu * streamCurrent[STREAM_BOTTOMUP]
                + w_td * streamCurrent[STREAM_TOPDOWN]
                + w_self * streamCurrent[STREAM_SELF]) / wsum

    def step(self):
        self.time += self.tau
        # Sulfaro normalized mixture at the current level (see module docstring):
        # the synaptic streams are combined by weight-normalized average; somatic
        # diffuse and external currents remain additive offsets outside the mix.
        self.I = self.externalInput + self.diffuseCurrent + self._mixtureCurrent()
        target = self.transfer(self.I)
        self.rate += self.tau * (target - self.rate) / self.tau_m
        if self.rate < 0.0:
            self.rate = 0.0
        if self.recordCellHistory:
            self.rateRecord.append(self.rate)
        # synapticInput / synapticInputBySource are cleared by the population
        # after this step (clearSynapticAccumulators), same contract as RateNeuron.
