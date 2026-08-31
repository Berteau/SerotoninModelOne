# Emergent "Realistic 5HT" experiment WITH ART secondary area (option B)

Same emergent, closed-loop option-B run as `../emergent_5ht/`, but with the ART
secondary area active: a Fuzzy ART classifier is pre-trained on the clean-V1
representations of the labeled stimuli, then drives top-down feedback into V1 and
is read out each step. Only the sensory loss is imposed; the homeostatic
controller does the rest. Full 10×10 grid, 5000 ms epochs, `gamma_p/10`, real
person/horse image input. Each mode paired with a no-plasticity control.

Held stimulus true class = **horse** (label 0).

## Rate trajectory (deprived-region V1, Hz)

| mode    | A_set | trough | peak | final (plast) | final (control) | remap W: base→final |
|---------|-------|--------|------|---------------|-----------------|---------------------|
| scotoma | 3.98  | 1.49   | 4.24 | 4.04          | 2.57            | 15 → 100            |
| blind   | 2.90  | 1.28   | 3.75 | 3.71          | 2.10            | 15 → 97             |

Same emergent drop → overshoot → settle as ART-off; the classifier and its
top-down feedback do not disrupt the rate homeostasis.

## ART readout — the hallucination (art_readout.png)

Both modes classify the stimulus correctly as **horse** at baseline. At the loss,
both flip to a confident **"person"** decision — the wrong category — riding on
the remapped cross-modal (audio) drive plus ART top-down feedback. This is the
predicted cross-modal hallucination: a confident percept of a category that the
veridical (now absent) input does not support.

The two modes then diverge:

- **scotoma (partial loss):** the "person" hallucination **latches and persists**
  for the rest of the run, match confidence ~0.86, comfortably above vigilance
  (0.75). The spared/intact surround and the sustained top-down feedback keep
  reinforcing it.
- **blind (full loss):** the hallucination is **transient**. With no intact
  columns to anchor the percept, as the homeostatic drive `h` relaxes and the
  top-down current collapses (return-epoch top-down ≈ 0.95, essentially off), the
  ART match **erodes to the vigilance line (~12 s)** and the readout
  **destabilizes into a person/reject oscillation, collapsing to reject**. Once
  ART rejects, its top-down turns off, which keeps it rejecting — a
  self-consistent feedback shutdown.

## Honest notes

- **No resolution to the correct class.** In neither mode does the readout return
  to "horse". ART has no mechanism to *relearn* the true category while the
  veridical input is gone, so the hallucination either persists (scotoma) or
  decays into reject (blind) — it never corrects. Showing hallucination
  *resolution* is precisely the goal of the shelved v3.0 classifier (see
  `../V3_PLAN.md`): a slow, unsupervised late-relearning readout with a trained
  "blank" class.
- **The blind instability is vigilance-driven.** The person/reject oscillation
  and collapse come from ART's match crossing the vigilance threshold — the same
  threshold fragility noted earlier. A trained "blank" class (v3.0) would replace
  this hard threshold with a learned decision boundary.
- Top-down magnitude here (~30) reflects the ART path (`topDownDriveHz=30`),
  distinct from the non-ART Category feedback in `../emergent_5ht/`.

## Figures (per mode)

- **art_readout.png** — ART decision strip (horse / person / reject) + match vs
  vigilance. The headline hallucination trace.
- **rates.png / streams.png / remap_weight.png / v1_maps.png / homeostatic.png**
  — as in the ART-off run; the emergent rate story is unchanged.
