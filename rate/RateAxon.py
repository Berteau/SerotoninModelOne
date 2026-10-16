'''
Rate-based analogue of the spiking model's Axon + postsynaptic conductance
(GlutGSDReceptor / GABAGSDReceptor).

Two things differ from the spiking axon, both deliberate:

1. Transmission failure is rolled into the weight.
   In the spiking model each spike is stochastically dropped with probability
   `failureRate` (Axon.enqueue: transmit only if random() > failureRate), and
   axonal 5HT2A serotonin lowers that probability. In a rate model there are no
   discrete spikes to drop, so the *expected* transmitted drive is simply
   scaled by (1 - failureRate). We therefore carry a deterministic effective
   weight

       w_eff = weight * clamp(1 - failureRate, 0, 1)

   and axonal 5HT2A receptors raise w_eff by lowering failureRate (clamped so
   transmission can't exceed 100%, matching the spiking cap of "every spike
   gets through"). This removes the single largest source of run-to-run noise.

2. Synaptic conductance is driven by presynaptic rate, not spikes.
   The spiking synapse gates a saturating Q/g system with a per-timestep spike
   flag K (which the methods draft itself notes is timestep-dependent). The
   rate reduction replaces K with the presynaptic rate and keeps a single
   saturating gating variable per receptor type:

       dg/dt = alpha * (r_pre/1000) * (1 - g) - g / tau_f

   with the paper's decay time constants tau_f (AMPA 10 ms, NMDA 100 ms,
   GABA 30 ms). g is bounded in [0, 1). The rise constants tau_r are collapsed
   into this single decay-dominated form; the slow NMDA conductance (tau_f =
   100 ms) is what feeds the calcium/plasticity term, exactly as in the paper.

A synapse is glutamatergic (AMPA + NMDA conductances) when weight > 0 and
GABAergic (GABA conductance) when weight <= 0, mirroring the spiking model's
choice of receptor class by weight sign. The sign of the injected current comes
from w_eff (negative for GABA), so conductances stay non-negative.
'''

# Decay time constants (ms), from the methods-draft synapse table.
TAU_F = {"AMPA": 10.0, "NMDA": 100.0, "GABA": 30.0}
# Release efficacy per receptor (tunable); sets the presynaptic rate at which
# the conductance half-saturates. Kept uniform by default.
ALPHA = {"AMPA": 1.0, "NMDA": 1.0, "GABA": 1.0}

BASE_FAILURE_RATE = 0.25


class RateAxon:
    def __init__(self, tau, weight, source, target, baseFailureRate=BASE_FAILURE_RATE):
        self.tau = tau
        self.weight = weight
        self.initialWeight = weight
        self.source = source
        self.target = target

        source.addOutput(self)
        target.addInput(self)

        self.glutamatergic = weight > 0

        # Conductance gating variables (non-negative, saturating toward 1).
        self.g_ampa = 0.0
        self.g_nmda = 0.0
        self.g_gaba = 0.0

        # Precomputed per-axon conductance constants (hoisted out of the per-step
        # hot loop -- equivalent to the ALPHA[...]/TAU_F[...] dict lookups, just
        # resolved once). tau is fixed for the life of the axon.
        self._alpha_ampa = ALPHA["AMPA"]
        self._alpha_nmda = ALPHA["NMDA"]
        self._alpha_gaba = ALPHA["GABA"]
        self._inv_tau_ampa = 1.0 / TAU_F["AMPA"]
        self._inv_tau_nmda = 1.0 / TAU_F["NMDA"]
        self._inv_tau_gaba = 1.0 / TAU_F["GABA"]

        # Transmission failure, rolled into the effective weight. Axonal 5HT2A
        # receptors modify self.failureRate; getEffectiveWeight() applies it.
        self.baseFailureRate = baseFailureRate
        self.failureRate = baseFailureRate
        self.axonalReceptors = []

        # Plasticity (Graupner/Brunel-style calcium rule); off unless enabled.
        # See _applyPlasticity for the rate reduction of the paper's rule.
        self.plasticity = False
        self.gamma_p = 0.0
        self.gamma_d = 0.0
        self.plasticityThreshold = 0.0
        # Soft upper bound on the weight (Gütig et al. 2003 / van Rossum et al.
        # 2000 style): potentiation is scaled by (1 - w/w_max) so LTP smoothly
        # vanishes as the weight approaches w_max, making the calcium-driven
        # rule stable instead of diverging (calcium ~ weight is otherwise a
        # positive-feedback loop). A departure from the paper's unbounded rule,
        # adopted because bounded stability is the point of the rate model.
        self.plasticityWeightMax = float("inf")
        self.pruned = False

        # Cache the source population once (used to attribute this axon's
        # injected current to its source for the ablation influence metric).
        self.sourcePopulation = getattr(source, "parentPopulation", None)

        # Cached effective weight w * clamp(1 - failureRate). It changes only
        # when the failure rate changes (serotonin) or the weight changes
        # (plasticity), which are rare relative to per-step reads, so we cache it
        # and refresh at those two points instead of recomputing every step.
        self._effWeight = 0.0
        self._recomputeEffWeight()

    # ---- transmission gain / failure rollup ----

    def getTransmissionGain(self):
        gain = 1.0 - self.failureRate
        if gain < 0.0:
            return 0.0
        if gain > 1.0:
            return 1.0
        return gain

    def _recomputeEffWeight(self):
        self._effWeight = self.weight * self.getTransmissionGain()

    def getEffectiveWeight(self):
        return self._effWeight

    def enablePlasticity(self, gamma_p, gamma_d, threshold, ceilingFactor=None):
        # Turn on the calcium-based plasticity rule with the given constants.
        # A synapse already pruned to zero stays pruned (matches the paper).
        # ceilingFactor sets the soft weight ceiling as a multiple of the
        # synapse's initial weight (w_max = ceilingFactor * initialWeight);
        # None leaves it unbounded (paper-faithful, but can diverge).
        if self.pruned:
            return
        self.plasticity = True
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.plasticityThreshold = threshold
        if ceilingFactor is not None:
            self.plasticityWeightMax = ceilingFactor * self.initialWeight

    def disablePlasticity(self):
        self.plasticity = False

    def addAxonalReceptor(self, receptor):
        self.axonalReceptors.append(receptor)
        receptor.setTarget(self)

    def recomputeFailureRate(self):
        # Axonal 5HT2A serotonin lowers the failure rate (raising transmission),
        # matching the spiking AxonalSerotoninReceptor: base - sum(weight*level).
        self.failureRate = self.baseFailureRate - sum(r.weight * r.level for r in self.axonalReceptors)
        self._recomputeEffWeight()

    # ---- per-step dynamics ----

    def _updateConductances(self):
        rms = self.source.rate / 1000.0     # presynaptic rate -> spikes per ms
        h = self.tau
        if self.glutamatergic:
            ga = self.g_ampa
            self.g_ampa = ga + h * (self._alpha_ampa * rms * (1.0 - ga) - ga * self._inv_tau_ampa)
            gn = self.g_nmda
            self.g_nmda = gn + h * (self._alpha_nmda * rms * (1.0 - gn) - gn * self._inv_tau_nmda)
        else:
            gg = self.g_gaba
            self.g_gaba = gg + h * (self._alpha_gaba * rms * (1.0 - gg) - gg * self._inv_tau_gaba)

    def _conductanceSum(self):
        if self.glutamatergic:
            return self.g_ampa + self.g_nmda
        return self.g_gaba

    def calciumProxy(self):
        # Postsynaptic intracellular calcium approximated as NMDA-mediated
        # conductance scaled by weight (I_ca ~ w * g_NMDA), per the methods draft.
        return self.getEffectiveWeight() * self.g_nmda

    def step(self):
        self._updateConductances()

        if self.plasticity:
            self._applyPlasticity()

        # Inject current into the postsynaptic neuron for this step, attributed
        # to this axon's source population so the target can compute the
        # leave-one-out ablation influence of each source.
        self.target.addSynapticTransmission(
            self.getEffectiveWeight() * self._conductanceSum(), self.sourcePopulation)

    def _applyPlasticity(self):
        # Rate reduction of the methods-draft calcium/STDP rule (Graupner &
        # Brunel 2012 style). The paper updates weight on each presynaptic
        # spike:
        #   LTP: dw = gamma_p * [I_ca - Th]_+ * (eligibility over recent post spikes)
        #   LTD: dw = -gamma_d                       (per presynaptic spike)
        # with I_ca ~ w_eff * g_NMDA. Presynaptic spikes arrive at rate r_pre;
        # the post-spike eligibility trace exp((t_post - t_pre)/tau_p) averages,
        # in rate form, to something proportional to the postsynaptic rate
        # r_post. So the expected weight change per unit time is:
        #
        #   dw/dt = (r_pre/1000) * ( gamma_p * (r_post/1000) * [Ca - Th]_+ - gamma_d )
        #
        # Rates in Hz are converted to per-ms so the constants keep sensible
        # magnitudes. Potentiation therefore requires BOTH pre and post activity
        # (Hebbian coincidence, carried by r_pre and r_post) plus calcium above
        # threshold; depression is unconditional on presynaptic drive. A synapse
        # driven to zero weight is pinned there and pruned, as in the paper.
        r_pre = self.source.rate
        r_post = self.target.rate
        ca_above = self.calciumProxy() - self.plasticityThreshold
        if ca_above < 0.0:
            ca_above = 0.0
        # Soft upper bound: potentiation fades to zero as weight -> w_max.
        headroom = 1.0 - self.weight / self.plasticityWeightMax
        if headroom < 0.0:
            headroom = 0.0
        potentiation = self.gamma_p * (r_post / 1000.0) * ca_above * headroom
        dw = self.tau * (r_pre / 1000.0) * (potentiation - self.gamma_d)
        self.weight += dw
        if self.weight <= 0.0:
            self.weight = 0.0
            self.plasticity = False
            self.pruned = True
        # Weight changed -> refresh the cached effective weight.
        self._recomputeEffWeight()
