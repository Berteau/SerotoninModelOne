def buildGrandPlanParams(gridSize=10, cellsPerColumn=8, categoryCount=2):
    """Parameters for the retinotopic grand-plan architecture.

    Defaults target the user's spec: a 10x10 retinotopic grid (100 visual
    columns), a small population per column (8 pyramidal + 2 FS), 20 tonotopic
    audio bins, and a 2-category secondary classifier (person / horse). The
    grid size is a constructor argument so the self-check can run a small, fast
    network while the target architecture stays the default.

    Weights are on the rate model's validated current scale. The Sulfaro
    perception knob is (wBottomUp, wTopDown): their ratio governs veridical vs
    top-down/hallucinatory V1, and both default to 1 (Sulfaro's neutral scheme).
    """
    params = {}
    params["tau"] = 0.1
    params["epochDurationMs"] = 1000
    params["warmupMs"] = 300

    # Architecture size
    params["gridSize"] = gridSize
    params["cellsPerColumn"] = cellsPerColumn
    params["fsPerColumn"] = 2
    params["audioBins"] = 20
    params["categoryCount"] = categoryCount
    params["topographicRadius"] = 1          # V1 receptive-field radius (columns)

    # Input drive range (Hz), used by StructuredInput to scale [0,1] -> rate.
    params["minRateHz"] = 0.0
    params["maxRateHz"] = 30.0

    # --- Sulfaro normalized-mixture stream weights (the perception knob) ---
    params["wBottomUp"] = 1.0                # w_{l-1}: ascending / sensory
    params["wTopDown"] = 1.0                 # w_{l+1}: descending / feedback
    params["wSelf"] = 1.0                    # w_l: current-layer recurrent

    # Diffuse serotonin (baseline). setSerotonin / set5HT2A change it at runtime.
    params["serotoninLevel"] = 10.0
    params["Somatic5HT2AWeight"] = 5.0
    params["Somatic5HT1AWeight"] = -1.0
    params["Axonal5HT2AWeight"] = 0.005      # base failure 0.25 -> ~20% at 5HT2A=10

    # --- stream currents into V1 (pre-normalization magnitudes) ---
    # These set each stream's raw synaptic current; the mixture then normalizes
    # by the stream WEIGHTS above, so absolute rates stay in a sane range.
    params["visualWeight"] = 500.0           # bottom-up: VisualInput -> V1 (topographic)
    params["audioWeight"] = 300.0            # bottom-up cross-modal: AudioInput -> V1 (remapping)
    params["audioToV1Likelihood"] = 0.3
    params["v1RecurrentWeight"] = 120.0      # self: within-column recurrent excitation
    params["v1ToFsWeight"] = 300.0           # self loop: V1 -> FS drive
    params["fsToV1Weight"] = -300.0          # self loop: FS -> V1 inhibition
    params["categoryToV1Weight"] = 400.0     # top-down: Category -> V1 feedback

    # --- secondary competitive classifier (first-pass, used when useART=False) ---
    params["v1ToCategoryWeight"] = 500.0     # feedforward pooling V1 -> Category
    params["categoryInhibitionWeight"] = -50.0   # winner-take-all lateral inhibition
    params["categoryVigilance"] = 5.0        # reject threshold (Hz); below -> no category wins

    # --- serotonin -> Sulfaro mixture coupling ---
    # When on, raising 5HT2A lowers the V1 bottom-up stream weight (bottomUpGain),
    # tipping the feedforward:feedback ratio toward feedback (Sulfaro/Seillier
    # bridge). Supplements the existing somatic (additive) and axonal
    # (transmission) serotonin effects. See _updateMixtureFromSerotonin.
    params["serotoninShiftsMixture"] = True
    params["bottomUpGainDamp"] = 0.2         # gain = 1 - damp*(L-base)/base
    params["minBottomUpGain"] = 0.1          # floor on bottom-up gain

    # --- "Realistic 5HT" phase-3 mechanism (opt-in) ---
    # Literature-grounded reframing of the deafferentation serotonin response
    # (5HT2A/3A modulation of deprived visual cortex + cross-modal input). When
    # on, phase 3 does NOT raise a single uniform 5HT; instead, in the DEPRIVED
    # region it (i) reduces somatic 5HT in visual cortex, (ii) raises V1 gain
    # (disinhibition -> higher bottom-up stream weight), (iii) raises 5HT on the
    # cross-modal audio->V1 transmission axons, and (iv) lowers the plasticity
    # threshold. All four are consistent with 5HT2A/3A modulation.
    params["realistic5HT"] = False
    params["v1SerotoninDeprived"] = 8.0          # mildly reduced somatic 5HT in deprived V1 (baseline 10)
    params["v1BottomUpGainDeprived"] = 1.5       # raised bottom-up gain in deprived region (5HT2A)
    params["crossModalSerotoninLevel"] = 40.0    # raised 5HT on audio->V1 transmission axons
    params["plasticityThresholdDeprived"] = 1.0  # lowered plasticity threshold (baseline 3.0)
    # Deprivation-induced INTRINSIC EXCITABILITY homeostasis (Desai, Rutherford &
    # Turrigiano 1999; the firing-rate-homeostasis literature). This is the
    # NON-serotonergic driver of the transient hyperactivity: an input-INDEPENDENT
    # tonic depolarization of the deprived region during the reorganization phase.
    # It is required because neither bottom-up gain (amplifies absent input) nor
    # disinhibition (removes absent activity-driven inhibition) can raise firing in
    # a cortex that has lost its feedforward drive; the hyperactivity in turn
    # provides the postsynaptic activity the Hebbian cross-modal remapping needs to
    # bootstrap. Relaxes at return as remapping restores drive (set to 0 in phase 4).
    params["intrinsicExcitabilityDrive"] = 60.0  # tonic depolarizing offset (deprived region, phase 3)

    # --- "Emergent Realistic 5HT" (opt-in): closed-loop, activity-driven ---
    # Instead of imposing the phase-3 responses on a schedule, a homeostatic
    # controller reads the deprived region's activity deficit each update and
    # ramps ALL responses (intrinsic excitability, bottom-up gain, reduced V1
    # 5HT, raised cross-modal 5HT, lowered plasticity threshold) in proportion to
    # a single latent drive h in [0,1]; h relaxes as remapping restores activity.
    # The only imposed event is the sensory loss; the drop -> hyperactivity ->
    # settle trajectory then emerges (and self-limits) from the loop. The
    # response VALUES at h=1 are the realistic5HT constants above; at h=0 they
    # equal baseline. Grounded in firing-rate / intrinsic-excitability homeostasis
    # (Turrigiano; Desai 1999) with the raphe/5HT2A-3A responses tied to the same
    # activity signal (cortico-raphe feedback substrate; Celada 2001).
    params["emergent5HT"] = False
    params["homeostaticTau"] = 800.0          # controller ramp time constant (ms)
    params["homeostaticTauRelax"] = 6000.0     # RELAX time constant (ms); > tau makes the
                                               # homeostatic changes persist after activity
                                               # recovers (intrinsic/scaling changes reverse
                                               # slowly) -> underdamped -> transient overshoot
    params["homeostaticUpdateEvery"] = 50      # steps between controller updates
    params["homeostaticGain"] = 1.4            # deficit->h loop gain (>1 allows overshoot)

    # --- ART secondary area (used when useART=True) ---
    params["useART"] = False                 # opt-in: Fuzzy ART classifier drives top-down
    params["artVigilance"] = 0.75            # ART vigilance rho (reject threshold)
    params["artActivityFloor"] = 0.05        # feature-mean floor below which -> reject
    params["artReferenceRate"] = 6.0         # V1 Hz mapping to feature value 1.0
    params["topDownDriveHz"] = 30.0          # Hz that template value 1.0 projects back as

    # --- plasticity (audio remapping synapses), reusing the validated rule ---
    # Tuned (with audioWeight/ceiling) so the potentiated cross-modal drive is
    # strong enough to functionally restore deprived-column firing after the
    # normalized mixture's weight-sum division, and to persist into the return
    # epoch -- while the no-plasticity control shows no recovery.
    params["gamma_p"] = 200.0
    params["gamma_d"] = 5e-4
    params["plasticityThreshold"] = 3.0
    params["plasticityCeilingFactor"] = 8.0
    params["remapSerotoninLevel"] = 40.0

    return params
