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
