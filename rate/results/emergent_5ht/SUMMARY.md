# Emergent "Realistic 5HT" experiment (option B, ART off)

Closed-loop, activity-driven version of the deafferentation response. The **only
imposed event is the sensory loss**; a homeostatic controller reads the deprived
region's activity deficit each update and ramps every deprivation response
(intrinsic excitability, bottom-up gain, reduced V1 5HT, raised cross-modal 5HT,
lowered plasticity threshold) in proportion to a single latent drive `h ∈ [0,1]`,
which relaxes as remapping restores activity. Option B (`homeostaticGain=2.5`,
`homeostaticTauRelax=6000`) makes the loop underdamped, so a transient
hyperactivity overshoot emerges and then settles — none of it scheduled.

Full 10×10 retinotopic grid, 8 cells/column, 2 classes (person/horse), real
edge-detected image input. Epochs 5000 ms each (baseline + 3 observation
windows); plasticity rate reduced 10× (`gamma_p/10`). ART classifier kept
separate/off (rate trajectory is classifier-independent). Each mode is run with
plasticity and with a no-plasticity control.

## Results (deprived-region V1 firing rate, Hz)

| mode    | set-point A_set | trough (at loss) | peak (overshoot) | final (plast) | final (control) | remap W: base→final |
|---------|-----------------|------------------|------------------|---------------|-----------------|---------------------|
| scotoma | 4.43            | 1.97             | 4.66             | 4.43          | 2.88            | 15 → 102            |
| blind   | 3.53            | 1.82             | 4.64             | 4.63          | 2.58            | 15 → 103            |

`h` peaks at 0.75 (scotoma) / 0.61 (blind) and relaxes to ~0.10 / ~0.07 by the
end — i.e. the controller releases once activity is restored.

## Reading the figures (per mode)

- **homeostatic.png** — the headline: deprived rate (red) and latent drive `h`
  (blue). Flat baseline → drop at loss → `h` emerges from zero → recovery →
  transient overshoot → `h` relaxes / rate settles. Only the loss is imposed.
- **rates.png** — plasticity vs no-plasticity control (and intact columns for the
  scotoma). Control stays depressed; plasticity recovers.
- **streams.png** — input streams into deprived V1: visual (lost at loss), audio
  cross-modal (the remapping readout, climbs), top-down feedback.
- **remap_weight.png** — mean audio→V1 remapping-synapse weight; potentiates
  under plasticity, flat under control.
- **v1_maps.png** — retinotopic V1 activity per epoch, lesion outlined; the
  deprived region darkens at loss and re-lights after remapping.

## Honest notes

- **scotoma** overshoots then settles back onto its baseline set-point.
- **blind** over-recovers and stabilizes *above* baseline (final 4.63 vs
  set-point 3.53): under full blindness the cross-modal audio drive replaces and
  exceeds the lost visual drive across the whole field, so once activity passes
  set-point the controller releases (`h`→0) while the remapping weight it already
  built persists and holds the rate high. This is a genuine model behavior, left
  as-is (not tuned toward a target).
- The baseline set-point differs between modes because it is the mean over the
  *deprived cell set* — the whole field under blindness (3.53) vs only the lesion
  patch under scotoma (4.43) — measured in epoch 1 before any loss is imposed.

## vs. the scheduled "Realistic 5HT" version

Same drop→hyperactivity→settle shape, but there the phase-3 responses are imposed
as a fixed step (intrinsic drive 60, etc.), giving a larger, externally-clamped
hyperactivity. Here the response magnitude is proportional to the *actual*
measured deficit and self-limits via the loop, so the overshoot is gentler
(~+5%) and the return is intrinsic rather than scheduled.
