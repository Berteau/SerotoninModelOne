# Version 3.0 direction (shelved — do not implement yet)

Recorded so the design decision survives; **not** being built now. Current work
continues with the ART secondary area.

## Idea: replace ART with a trained one-hot classifier that has a "blank" class

Motivation: ART's reject is a vigilance threshold and its complement-coded
lower-envelope templates conflate brightness with content (the dimmed-horse
reads-as-person confound, and the odd blind-phase match dynamics). A classifier
with an explicit "nothing there" class turns the hallucination into a clean,
plottable posterior — **P(person) rising while ground truth is blank** — and the
"blank" prior makes the hallucination have to *overcome* a trained "no stimulus"
representation rather than fall out of a threshold artifact (stronger, more
honest test; still not tuned toward the result).

## Decisions taken

- **"Blank" class is trained on ZERO veridical (visual) input** — i.e. "no
  stimulus present," NOT the deprived-cortex activity pattern. (Training blank on
  the deprived state would teach the model to explain away the very thing we test.)
- **Not frozen — slow to re-learn.** The classifier keeps a low learning rate so
  it can acquire *new objects in the final epoch*, which is how the model would
  show the **resolution of hallucinations** (the percept reorganizes as veridical
  structure returns / cross-modal content stabilizes).
- **Consequences of late online learning:**
  - With no new labels available in the final epoch, that late learning likely
    has to be **unsupervised** (self-organized categories), not supervised
    one-hot. So "one-hot classifier" is the readout framing, but the learning
    rule at the end is unsupervised.
  - May require **two audio inputs** (rather than one) to give the cross-modal
    stream enough structure to form/resolve distinct categories.

## Open questions for when this is picked up

- Exact unsupervised rule and how it reconciles with the one-hot readout (e.g. a
  fixed labeled readout head over self-organizing feature clusters).
- Top-down feedback gain: ART's vigilance was an implicit brake on the
  classifier -> V1 -> classifier loop; a confident softmax with strong feedback
  can latch. Set this gain deliberately.
- What the two audio inputs represent and how they map onto V1.

Keep the classifier modular (as now) so this can slot in without disturbing the
V1 / Sulfaro-mixture / emergent-controller core.
