def buildDefaultParams():
    params = {}
    params["popCount"] = 8
    params["gridRows"] = 10
    params["gridCols"] = 10
    params["numAudioColumns"] = 20
    params["tau"] = 0.1
    params["frameDurationMs"] = 5
    params["phaseDurationMs"] = 20

    params["serotoninLevelV"] = 10
    params["serotoninLevelA"] = 10
    params["remapSerotoninLevel"] = 30
    params["remapPlasticityCp"] = 90

    # The pooled connection weights below (pyramidalSelfExcitationWeight etc.)
    # are divided by popCount to get a per-synapse weight (see
    # RetinotopicAVNetwork.gaussWeightWithChanceFactory). SerotoninAVNetwork's
    # values were tuned for popCount=40; reusing them unscaled at a much
    # smaller popCount makes each synapse disproportionately strong and drives
    # the network into pathological, near-saturated firing. popCount/40 alone
    # (0.2 here) overcorrects and leaves the network too quiet to fire once
    # lesioned columns lose their own input, so this uses an empirically
    # tuned factor that keeps activity in a healthy middle ground.
    weightScale = 0.4

    params["baselineRateV"] = 5
    params["baselineRateA"] = 5
    params["inputWeightV"] = 1500 * weightScale
    params["inputWeightA"] = 1500 * weightScale
    params["inputWeightVA"] = 800 * weightScale
    params["inputWeightAV"] = 800 * weightScale
    params["crossModalLikelihood"] = 0.3
    params["lateralVisualWeight"] = 1500 * weightScale
    params["lateralVisualLikelihood"] = 0.3
    params["lateralAudioWeight"] = 1500 * weightScale
    params["lateralAudioLikelihood"] = 0.3

    params["Somatic5HT2AWeight"] = 20
    params["Somatic5HT1AWeight"] = -5
    params["Axonal5HT2AWeight"] = 0.2
    params["Axonal5HT1AWeight"] = -0.2
    params["pyramidalSelfExcitationWeight"] = 15000 * weightScale
    params["PyramidalsToFSWeight"] = 50000 * weightScale
    params["FSToPyramidalsWeight"] = -30000 * weightScale
    params["PyramidalsToLTSWeight"] = 80000 * weightScale
    return params
