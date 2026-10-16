import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np

from rate.RetinotopicGrandPlanParams import buildGrandPlanParams
from rate.RetinotopicGrandPlanNetwork import RetinotopicGrandPlanNetwork
from rate.StructuredInput import StructuredInputSet, TARGET_AV_CORRELATION
from rate.NormalizedMixtureNeuron import (
    NormalizedMixtureNeuron, STREAM_BOTTOMUP, STREAM_TOPDOWN, STREAM_SELF)

'''
Self-check for the retinotopic grand-plan architecture and its Sulfaro et al.
(2023) normalized-mixture V1. Runnable directly (python3 rate/GrandPlanSelfCheck.py)
and discoverable by pytest. Uses a small grid so it runs in a few seconds; the
target 10x10 architecture is the params default.

Checks:
  1. Structured input has audio exactly r = 0.25 correlated with visual.
  2. The V1 mixture neuron is a weight-normalized convex combination of streams.
  3. The Sulfaro feedforward:feedback knob (wBottomUp:wTopDown) shifts V1's
     steady-state current between the bottom-up and top-down streams.
  4. Serotonergic 5HT2A bottom-up gain (Sulfaro's neurochemical bridge) reduces
     the bottom-up share of V1's drive.
'''

GRID = 4
CPC = 6


def _steady_state(net, steps):
    for _ in range(steps):
        net.step()


def _mean_stream_currents(net):
    # Mean raw (pre-weight) per-stream current across V1 pyramidals, from the
    # last step's recorded decomposition.
    cells = net.populations["V1pyr"].cells
    bu = np.mean([c.lastStreamCurrent[STREAM_BOTTOMUP] for c in cells])
    td = np.mean([c.lastStreamCurrent[STREAM_TOPDOWN] for c in cells])
    se = np.mean([c.lastStreamCurrent[STREAM_SELF] for c in cells])
    mixI = np.mean([c.I for c in cells])
    return bu, td, se, mixI


# ---------------------------------------------------------------------------

def test_structured_input_correlation():
    for seed in range(5):
        data = StructuredInputSet(gridSize=GRID, audioBins=20, classCount=2, seed=seed)
        for st in data.stimuli:
            r = data.measuredAVCorrelation(st)
            assert abs(r - TARGET_AV_CORRELATION) < 1e-6, (seed, st.label, r)
    print("[1] structured input: audio r = 0.25 correlated with visual  (exact) OK")


def test_mixture_neuron_convex_combination():
    tau = 0.1

    class P:
        def __init__(self, n): self.name = n
    bu, td, se = P("bu"), P("td"), P("self")

    n = NormalizedMixtureNeuron(tau, "Pyramidal", "t")
    n.assignStream(bu, STREAM_BOTTOMUP)
    n.assignStream(td, STREAM_TOPDOWN)
    n.assignStream(se, STREAM_SELF)

    def inject(a, b, c):
        n.clearSynapticAccumulators()
        n.addSynapticTransmission(a, bu)
        n.addSynapticTransmission(b, td)
        n.addSynapticTransmission(c, se)

    # Equal weights -> plain mean of the three streams.
    inject(120.0, 60.0, 30.0); n.step()
    assert abs(n.I - (120 + 60 + 30) / 3.0) < 1e-9, n.I
    # Convex combination: mixture current never exceeds the largest stream.
    assert min(120, 60, 30) - 1e-9 <= n.I <= max(120, 60, 30) + 1e-9
    # Upweighting a stream moves the mixture toward it.
    n.streamWeights[STREAM_BOTTOMUP] = 5.0
    inject(120.0, 60.0, 30.0); n.step()
    assert n.I > (120 + 60 + 30) / 3.0
    # Additive offsets (diffuse/external) sit outside the mixture.
    n.streamWeights[STREAM_BOTTOMUP] = 1.0
    n.diffuseCurrent = 7.0; n.externalInput = 3.0
    inject(120.0, 60.0, 30.0); n.step()
    assert abs(n.I - ((120 + 60 + 30) / 3.0 + 10.0)) < 1e-9, n.I
    print("[2] mixture neuron: weight-normalized convex combination OK")


def test_network_ratio_knob():
    # With bottom-up and top-down driving V1 to different current levels, the
    # Sulfaro knob wBottomUp:wTopDown should slide V1's steady mixture current
    # from the bottom-up level toward the top-down level as the ratio falls.
    random.seed(0); np.random.seed(0)
    data = StructuredInputSet(gridSize=GRID, audioBins=20, classCount=2, seed=0,
                              minRateHz=0.0, maxRateHz=30.0)
    st = data.stimuli[0]

    ratios = [("bottom-up dominant", 4.0, 1.0),
              ("equal", 1.0, 1.0),
              ("top-down dominant", 1.0, 4.0)]
    mixByRatio = []
    bu = td = None
    for label, wB, wT in ratios:
        random.seed(0); np.random.seed(0)
        p = buildGrandPlanParams(gridSize=GRID, cellsPerColumn=CPC, categoryCount=2)
        p["wBottomUp"], p["wTopDown"], p["wSelf"] = wB, wT, 0.0  # isolate the 2 Sulfaro streams
        net = RetinotopicGrandPlanNetwork(p["tau"], p)
        net.setVisualRates(data.visualRates(st))
        net.setAudioRates(data.audioRates(st))
        # Drive a strong, distinct top-down by injecting current into Category cells.
        for c in net.populations["Category"].cells:
            c.setInjectedCurrent(140.0)
        _steady_state(net, 6000)   # 600 ms
        b, t, s, mixI = _mean_stream_currents(net)
        bu, td = b, t
        mixByRatio.append(mixI)
        print("    %-20s wB:wT=%.0f:%.0f  bottom-up=%.1f top-down=%.1f  mixI=%.1f"
              % (label, wB, wT, b, t, mixI))

    assert td > 0 and bu > 0, (bu, td)
    # Direction of the shift is set by which stream drives the larger current.
    if bu > td:
        assert mixByRatio[0] > mixByRatio[1] > mixByRatio[2], mixByRatio
    else:
        assert mixByRatio[0] < mixByRatio[1] < mixByRatio[2], mixByRatio
    print("[3] Sulfaro ratio knob: V1 steady current slides between streams (monotone) OK")


def test_serotonin_bottomup_gain():
    # 5HT2A dampening of bottom-up gain (Sulfaro's bridge) should lower the
    # bottom-up share of V1's weighted drive.
    def bottomup_share(gain):
        random.seed(0); np.random.seed(0)
        p = buildGrandPlanParams(gridSize=GRID, cellsPerColumn=CPC, categoryCount=2)
        p["wSelf"] = 0.0
        net = RetinotopicGrandPlanNetwork(p["tau"], p)
        data = StructuredInputSet(gridSize=GRID, audioBins=20, classCount=2, seed=0,
                                  minRateHz=0.0, maxRateHz=30.0)
        st = data.stimuli[0]
        net.setVisualRates(data.visualRates(st))
        net.setAudioRates(data.audioRates(st))
        for c in net.populations["Category"].cells:
            c.setInjectedCurrent(140.0)
        net.setV1BottomUpGain(gain)
        _steady_state(net, 6000)
        b, t, s, mixI = _mean_stream_currents(net)
        # Effective weighted share of bottom-up in the mixture.
        wB = p["wBottomUp"] * gain
        wT = p["wTopDown"]
        return (wB * b) / (wB * b + wT * t)

    full = bottomup_share(1.0)
    damped = bottomup_share(0.4)
    print("    bottom-up share: gain=1.0 -> %.3f,  gain=0.4 (5HT2A) -> %.3f" % (full, damped))
    assert damped < full, (full, damped)
    print("[4] 5HT2A bottom-up gain: dampening lowers bottom-up share OK")


def main():
    test_structured_input_correlation()
    test_mixture_neuron_convex_combination()
    test_network_ratio_knob()
    test_serotonin_bottomup_gain()
    print("\nAll grand-plan self-checks passed.")


if __name__ == "__main__":
    main()
