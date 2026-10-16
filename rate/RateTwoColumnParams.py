def buildDefaultParams():
    """Starting parameters for the rate two-region sensory-loss model.

    Weights are on the rate model's current scale (chosen so pyramidal cells sit
    at biologically sane tens-of-Hz rates), NOT the spiking model's inflated
    scale. These are a tuning starting point; see rate/README for the validation
    targets they are being tuned against.
    """
    params = {}
    params["tau"] = 0.1
    params["epochDurationMs"] = 1000     # paper figures use 1000 ms epochs
    params["warmupMs"] = 500             # settle to steady state before epoch 1
    params["popCount"] = 20

    # Input drive (Poisson lambda, Hz). Draft baseline lambda = 30.
    params["rateA"] = 30.0
    params["rateB"] = 30.0

    # Diffuse serotonin levels and the epoch-3 elevation (draft: 10 -> 40 in A).
    params["serotoninLevelA"] = 10.0
    params["serotoninLevelB"] = 10.0
    params["remapSerotoninLevel"] = 40.0

    # Somatic serotonin receptor weights (per unit serotonin level). Code-sign
    # convention: 5HT2A depolarizing (+), 5HT1A hyperpolarizing (-); |2A|>|1A|
    # so raising serotonin in epoch 3 nets depolarizing drive.
    params["Somatic5HT2AWeight"] = 5.0
    params["Somatic5HT1AWeight"] = -1.0
    params["Somatic5HT2AWeightLTS"] = 3.0

    # Axonal 5HT2A on S->P input axons. 0.005 with base failure 0.25 gives the
    # draft's 20% failure at serotonin 10 and 5% at serotonin 40.
    params["Axonal5HT2AWeight"] = 0.005

    # Feedforward input weights (pooled; divided by popCount per synapse).
    params["inputWeightA"] = 500.0
    params["inputWeightB"] = 500.0
    params["inputWeightAB"] = 300.0
    params["crossModalABLikelihood"] = 0.5
    params["inputWeightBA"] = 300.0      # S_B -> P_A: the remapping synapses
    params["crossModalBALikelihood"] = 0.5

    # Recurrent / local circuit
    params["pyramidalSelfExcitationWeight"] = 60.0
    params["pyramidalToPyramidalWeight"] = 60.0
    params["pyramidalToPyramidalLikelihood"] = 0.3
    params["PyramidalsToFSWeight"] = 400.0
    params["FSToPyramidalsWeight"] = -400.0
    params["PyramidalsToLTSWeight"] = 250.0
    params["LTStoFSWeight"] = -150.0
    params["LTStoPyramidalsWeight"] = -150.0

    # Plasticity (Graupner/Brunel calcium rule), inflated per the draft for the
    # short epoch. Tuned so the remapping synapses potentiate meaningfully in
    # epoch 3 without diverging.
    params["gamma_p"] = 50.0
    params["gamma_d"] = 5e-4
    params["plasticityThreshold"] = 3.0
    # Soft weight ceiling: remapping synapses may potentiate up to this multiple
    # of their initial weight, past which LTP smoothly saturates (keeps the
    # calcium-driven rule stable). See RateAxon._applyPlasticity.
    params["plasticityCeilingFactor"] = 4.0

    # Pharmacological (5HT2A-agonist / psilocybin) experiment. No sensory loss;
    # in epoch 2 only 5HT2A is raised (to pharma5HT2ALevel) in BOTH regions,
    # 5HT1A stays at baseline, and plasticity runs an order of magnitude weaker
    # than the sensory experiment (reflecting the far shorter modeled timescale,
    # per the methods draft).
    params["pharma5HT2ALevel"] = 40.0
    params["pharma_gamma_p"] = params["gamma_p"] * 0.1
    params["pharma_gamma_d"] = params["gamma_d"] * 0.1

    return params
