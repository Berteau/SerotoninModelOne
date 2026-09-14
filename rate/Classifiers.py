import numpy as np

from rate.ARTClassifier import FuzzyART, REJECT

'''
Switchable downstream classifier for the v3.1 experiment. One interface, three
backends selected at run time (classifierType):

  FUZZY_ART     unsupervised fuzzy ART: choice/match/vigilance, slow recode,
                commits new categories via the vigilance test. The scientific
                default -- new (e.g. audio-driven) categories can self-organise.
  FUZZY_ARTMAP  supervised at pretrain (labelled map via match tracking), then
                slow UNSUPERVISED recode during the trial (no labels available
                in a deprivation trial). Comparison backend.
  KNN           online nearest-prototype / k-means-style clusterer: prototypes
                start from pretrain, the nearest drifts toward each input by the
                learning rate, and an input far from every prototype (a distance
                "vigilance") spawns a new one. Comparison backend.

Common interface:
  pretrain(X, y)         initial (labelled) training on clean-V1 features
  observe(x) -> (cluster, label, confidence)   classify WITHOUT learning
  learn(x)               one slow UNSUPERVISED update (drift / spawn)
  n_clusters             current number of categories/prototypes
  cluster_to_class       dict cluster-index -> majority pretrain class (or REJECT)

All three can spawn new clusters and drift, so emergence of new categories is
possible in each; only the mechanism differs. `label` is the pretrain-majority
class of the assigned cluster (REJECT for a novel/too-weak assignment), so the
same hallucination / emergence metrics apply regardless of backend.
'''

FUZZY_ART = "FUZZY_ART"
FUZZY_ARTMAP = "FUZZY_ARTMAP"
KNN = "KNN"
KINDS = (FUZZY_ART, FUZZY_ARTMAP, KNN)


def make_classifier(kind, dim, learningRate, vigilance=0.75,
                    activityFloor=0.05, knnDistanceVigilance=0.30, seed=0):
    if kind == FUZZY_ART:
        return UnsupervisedFuzzyART(dim, beta=learningRate, vigilance=vigilance,
                                    activityFloor=activityFloor)
    if kind == FUZZY_ARTMAP:
        return SupervisedFuzzyARTMAP(dim, beta=learningRate, vigilance=vigilance,
                                     activityFloor=activityFloor)
    if kind == KNN:
        return OnlineKNN(dim, learningRate=learningRate,
                         distanceVigilance=knnDistanceVigilance,
                         activityFloor=activityFloor, seed=seed)
    raise ValueError("Unknown classifierType %r (expected one of %s)" % (kind, KINDS))


def _majority_map(cluster_ids, labels):
    # cluster index -> most common pretrain class label seen in that cluster.
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    for c, y in zip(cluster_ids, labels):
        votes[c][y] += 1
    return {c: cnt.most_common(1)[0][0] for c, cnt in votes.items()}


class _ARTBase:
    """Shared plumbing over a FuzzyART instance (complement coding, choice/match,
    activity gate, cluster->class majority map)."""

    def __init__(self, dim, beta, vigilance, activityFloor):
        self.kind = None
        self.dim = int(dim)
        self.art = FuzzyART(dim=dim, vigilance=vigilance, alpha=0.01,
                            beta=float(beta), activity_floor=activityFloor)
        self.cluster_to_class = {}

    @property
    def n_clusters(self):
        return self.art.n_categories

    def _too_weak(self, x01):
        af = self.art.activity_floor
        return af is not None and float(np.mean(x01)) < af

    def _best_resonant(self, I):
        # (cluster index, match) of the best category passing vigilance, else
        # (None, best_match_overall).
        if not self.art.weights:
            return None, 0.0
        T = self.art._choices(I)
        best_match = 0.0
        for j in np.argsort(T)[::-1]:
            m = self.art._match(I, self.art.weights[j])
            best_match = max(best_match, m)
            if m >= self.art.vigilance:
                return int(j), float(m)
        return None, float(best_match)

    def observe(self, x):
        x01 = self.art._clip01(x)
        if self._too_weak(x01):
            return (REJECT, REJECT, 0.0)
        I = self.art._complement(x01)
        j, m = self._best_resonant(I)
        if j is None:
            return (REJECT, REJECT, m)
        return (j, self.cluster_to_class.get(j, REJECT), m)

    def _recode_or_commit(self, x):
        # Unsupervised: learn into the best resonant category, else commit a new
        # one. Returns the cluster index touched.
        x01 = self.art._clip01(x)
        if self._too_weak(x01):
            return None
        I = self.art._complement(x01)
        j, _ = self._best_resonant(I)
        if j is None:
            self.art.weights.append(I.copy())
            self.art.labels.append(None)
            return self.art.n_categories - 1
        beta = self.art.beta
        self.art.weights[j] = beta * np.minimum(I, self.art.weights[j]) + (1 - beta) * self.art.weights[j]
        return j

    def learn(self, x):
        self._recode_or_commit(x)

    def snapshot(self):
        return {"n_clusters": self.n_clusters,
                "cluster_to_class": dict(self.cluster_to_class)}


class UnsupervisedFuzzyART(_ARTBase):
    def __init__(self, dim, beta, vigilance, activityFloor):
        super().__init__(dim, beta, vigilance, activityFloor)
        self.kind = FUZZY_ART

    def pretrain(self, X, y):
        # Unsupervised clustering of the pretrain features, then label each
        # resulting cluster by the majority true class that landed in it.
        assigned = []
        for x in X:
            assigned.append(self._recode_or_commit(x))
        # a second no-learn pass gives stable final-cluster assignments to map
        final = [self.observe(x)[0] for x in X]
        self.cluster_to_class = _majority_map(
            [c for c in final if c != REJECT],
            [yy for c, yy in zip(final, y) if c != REJECT])
        return self


class SupervisedFuzzyARTMAP(_ARTBase):
    def __init__(self, dim, beta, vigilance, activityFloor):
        super().__init__(dim, beta, vigilance, activityFloor)
        self.kind = FUZZY_ARTMAP

    def pretrain(self, X, y):
        # Supervised ARTMAP training (match tracking on labels).
        for x, label in zip(X, y):
            self.art.train_one(x, label)
        # cluster i already carries its trained label
        self.cluster_to_class = {i: lbl for i, lbl in enumerate(self.art.labels)}
        return self

    def learn(self, x):
        # Trial-time learning has no labels -> slow unsupervised recode of the
        # existing (labelled) categories; a genuinely novel input commits a new,
        # unlabelled cluster (label REJECT until/unless it acquires structure).
        j = self._recode_or_commit(x)
        if j is not None and j not in self.cluster_to_class:
            self.cluster_to_class[j] = REJECT


class OnlineKNN:
    """Online nearest-prototype clusterer (k-means-style with vigilance spawning).

    Not lazy k-NN: prototypes are maintained online so the model can drift and
    grow like the ART backends. Nearest prototype -> cluster; that prototype's
    pretrain-majority class -> label. learn() moves the nearest prototype toward
    the input by learningRate, or spawns a new prototype if the input is farther
    than distanceVigilance (in normalized feature distance) from all of them.
    """

    def __init__(self, dim, learningRate, distanceVigilance, activityFloor, seed=0):
        self.kind = KNN
        self.dim = int(dim)
        self.eta = float(learningRate)
        self.dvig = float(distanceVigilance)
        self.activity_floor = None if activityFloor is None else float(activityFloor)
        self.protos = []                 # list of length-dim vectors in [0,1]
        self.cluster_to_class = {}
        self.rng = np.random.RandomState(seed)

    @property
    def n_clusters(self):
        return len(self.protos)

    def _clip01(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return np.clip(x, 0.0, 1.0)

    def _nearest(self, x01):
        # (index, normalized distance) of nearest prototype; dist in [0,1] via
        # RMS over dims so distanceVigilance is grid-size independent.
        if not self.protos:
            return None, 1.0
        d = [np.sqrt(np.mean((x01 - p) ** 2)) for p in self.protos]
        j = int(np.argmin(d))
        return j, float(d[j])

    def _too_weak(self, x01):
        return self.activity_floor is not None and float(np.mean(x01)) < self.activity_floor

    def pretrain(self, X, y):
        # Seed one prototype per class (class-mean feature); label by that class.
        X = [self._clip01(x) for x in X]
        classes = sorted(set(y))
        for c in classes:
            mean = np.mean([x for x, yy in zip(X, y) if yy == c], axis=0)
            self.protos.append(mean.copy())
            self.cluster_to_class[len(self.protos) - 1] = c
        return self

    def observe(self, x):
        x01 = self._clip01(x)
        if self._too_weak(x01):
            return (REJECT, REJECT, 0.0)
        j, d = self._nearest(x01)
        if j is None:
            return (REJECT, REJECT, 0.0)
        conf = max(0.0, 1.0 - d)
        return (j, self.cluster_to_class.get(j, REJECT), conf)

    def learn(self, x):
        x01 = self._clip01(x)
        if self._too_weak(x01):
            return
        j, d = self._nearest(x01)
        if j is None or d > self.dvig:
            self.protos.append(x01.copy())          # spawn a new prototype
            return
        self.protos[j] = self.protos[j] + self.eta * (x01 - self.protos[j])

    def snapshot(self):
        return {"n_clusters": self.n_clusters,
                "cluster_to_class": dict(self.cluster_to_class)}
