import numpy as np

'''
Fuzzy ART / ARTMAP-lite classifier for the retinotopic grand-plan secondary
cortical area.

Purpose in this project
-----------------------
The secondary area reads the V1 activity pattern and performs a TRIVIAL one-hot
classification (person vs horse) with a REJECT option. The point is not image
recognition; it is to provide a categorical readout that distinguishes:
  * veridical perception : V1 pattern matches a learned category -> classify;
  * blank / lost input    : V1 pattern too weak/unstructured -> reject;
  * hallucination         : V1 confidently classifies despite no veridical
                            bottom-up drive (e.g. top-down- or remapping-driven);
  * recovery              : after cross-modal remapping, V1 re-classifies as the
                            correct category.
It also supplies content-specific top-down feedback: the winning category's
template projected back onto V1 (the descending stream of the Sulfaro mixture).

Why fuzzy ART
-------------
ART is the canonical neural architecture with a built-in reject/novelty test:
the vigilance criterion. Fuzzy ART (Carpenter, Grossberg & Rosen 1991) extends
it to analog inputs in [0,1] via the fuzzy AND (elementwise min) and complement
coding. We use the standard equations:

  complement coding : I = [x, 1 - x]                       (|I| = dim, constant)
  choice (activation): T_j = |I ^ w_j| / (alpha + |w_j|)   (^ = elementwise min)
  match             : rho_j = |I ^ w_j| / |I|
  resonance         : winner J = argmax_j T_j; accept iff rho_J >= vigilance
  fast-commit/slow-recode learning: w_J <- beta (I ^ w_J) + (1 - beta) w_J

Supervised labels (ARTMAP-lite): each committed category carries a class label.
During training, if the resonant category's label disagrees with the target
label, MATCH TRACKING raises the effective vigilance just past the current match
and re-searches, forcing a distinct category for the conflicting class -- the
mechanism ARTMAP uses to guarantee the map is consistent with the labels.

This is the discrete ART algorithm (exact choice/match/vigilance/search), used
as a functional module that the rate network queries each step; it is not itself
a population of rate neurons. That keeps the reject/vigilance semantics -- the
scientific payload -- exact, rather than approximating them with lateral
inhibition. The rate network supplies the input pattern (pooled V1 rates) and
consumes the outputs (winner one-hot, reject flag, match, top-down template).
'''

REJECT = -1


class FuzzyART:
    def __init__(self, dim, vigilance=0.75, alpha=0.01, beta=1.0, activity_floor=0.05):
        self.dim = int(dim)
        self.vigilance = float(vigilance)
        self.alpha = float(alpha)          # choice tie-breaker toward smaller categories
        self.beta = float(beta)            # learning rate (1.0 = fast commit)
        # Activity gate: an input whose mean drive is below this floor is
        # rejected outright, before ART. Complement coding makes fuzzy ART match
        # sparse/blank inputs against the complement half of every template, so
        # vigilance alone does not reliably reject a near-silent input; the gate
        # gives the semantics we need ("deprived / near-silent V1 -> no percept
        # -> reject"), and is non-fragile (monotone in activity). Set to None to
        # disable.
        self.activity_floor = None if activity_floor is None else float(activity_floor)
        self.weights = []                  # committed templates, each length 2*dim
        self.labels = []                   # class label per committed category

    # ---- coding ----

    def _clip01(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.dim:
            raise ValueError("feature length %d != dim %d" % (x.shape[0], self.dim))
        return np.clip(x, 0.0, 1.0)

    def _complement(self, x01):
        return np.concatenate([x01, 1.0 - x01])

    # ---- ART primitives ----

    def _choices(self, I):
        # T_j = |I ^ w_j| / (alpha + |w_j|)
        return np.array([np.sum(np.minimum(I, w)) / (self.alpha + np.sum(w))
                         for w in self.weights])

    def _match(self, I, w):
        # rho_j = |I ^ w_j| / |I|   (|I| = dim under complement coding)
        return np.sum(np.minimum(I, w)) / self.dim

    # ---- inference ----

    def classify(self, x, return_detail=False):
        """Classify a feature vector without learning.

        Returns the class label of the resonant category, or REJECT (-1) if no
        committed category passes vigilance. With return_detail, also returns
        (category_index, match) for the best resonant (or best overall) node.
        """
        x01 = self._clip01(x)
        if self.activity_floor is not None and float(np.mean(x01)) < self.activity_floor:
            # Near-silent input: no percept to classify.
            return (REJECT, (None, 0.0)) if return_detail else REJECT
        I = self._complement(x01)
        if not self.weights:
            return (REJECT, (None, 0.0)) if return_detail else REJECT
        T = self._choices(I)
        order = np.argsort(T)[::-1]
        best_match = 0.0
        for j in order:
            m = self._match(I, self.weights[j])
            best_match = max(best_match, m)
            if m >= self.vigilance:
                lbl = self.labels[j]
                return (lbl, (j, m)) if return_detail else lbl
        return (REJECT, (None, best_match)) if return_detail else REJECT

    def category_template(self, category_index):
        """Decomplemented template x-part (length dim) of a committed category,
        i.e. the pattern that category feeds back top-down. For fuzzy ART the
        x-part of w is the elementwise-min lower envelope of that category's
        exemplars."""
        return self.weights[category_index][:self.dim].copy()

    # ---- learning ----

    def train_one(self, x, label):
        """Present one labeled example; run ART search with match tracking and
        either learn into a resonant same-label category or commit a new one.
        Returns the committed/updated category index."""
        I = self._complement(self._clip01(x))
        if self.weights:
            T = self._choices(I)
            rho = self.vigilance
            order = list(np.argsort(T)[::-1])
            for j in order:
                m = self._match(I, self.weights[j])
                if m >= rho:
                    if self.labels[j] == label:
                        # resonance with correct label -> learn
                        self.weights[j] = (self.beta * np.minimum(I, self.weights[j])
                                           + (1.0 - self.beta) * self.weights[j])
                        return j
                    else:
                        # label conflict -> match tracking: raise vigilance just
                        # above this match so this (and any equally-matching)
                        # wrong-label node is rejected, then keep searching.
                        rho = m + 1e-6
        # no acceptable resonant category -> commit a new one
        self.weights.append(I.copy())
        self.labels.append(label)
        return len(self.weights) - 1

    def fit(self, X, y, epochs=1):
        for _ in range(epochs):
            for x, label in zip(X, y):
                self.train_one(x, label)
        return self

    @property
    def n_categories(self):
        return len(self.weights)
