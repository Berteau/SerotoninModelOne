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
KOHONEN_SOM = "KOHONEN_SOM"
KINDS = (KOHONEN_SOM, FUZZY_ART, FUZZY_ARTMAP, KNN)


def make_classifier(kind, dim, learningRate, vigilance=0.75,
                    activityFloor=0.05, knnDistanceVigilance=0.30,
                    pretrainRate=1.0, pretrainEpochs=5, classCount=None,
                    somSide=None, seed=0):
    if kind == KOHONEN_SOM:
        return SelfOrganizingMap(dim, learningRate=learningRate, side=somSide,
                                 classCount=classCount, activityFloor=activityFloor,
                                 pretrainEpochs=pretrainEpochs, seed=seed)
    if kind == FUZZY_ART:
        return UnsupervisedFuzzyART(dim, beta=learningRate, vigilance=vigilance,
                                    activityFloor=activityFloor,
                                    pretrainRate=pretrainRate, pretrainEpochs=pretrainEpochs)
    if kind == FUZZY_ARTMAP:
        return SupervisedFuzzyARTMAP(dim, beta=learningRate, vigilance=vigilance,
                                     activityFloor=activityFloor,
                                     pretrainRate=pretrainRate, pretrainEpochs=pretrainEpochs)
    if kind == KNN:
        return OnlineKNN(dim, learningRate=learningRate,
                         distanceVigilance=knnDistanceVigilance,
                         activityFloor=activityFloor,
                         pretrainEpochs=pretrainEpochs, seed=seed)
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

    def __init__(self, dim, beta, vigilance, activityFloor,
                 pretrainRate=1.0, pretrainEpochs=5):
        self.kind = None
        self.dim = int(dim)
        # beta is the SLOW trial-time learning rate; pretrain runs at the fast
        # pretrainRate over pretrainEpochs so the categories are properly
        # established before the slow continuous drift begins.
        self.art = FuzzyART(dim=dim, vigilance=vigilance, alpha=0.01,
                            beta=float(beta), activity_floor=activityFloor)
        self.learn_beta = float(beta)
        self.pretrain_beta = float(pretrainRate)
        self.pretrain_epochs = int(pretrainEpochs)
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

    def _label_all_clusters(self, X, y):
        # Guarantee EVERY committed cluster has a class label. A cluster can be the
        # best-resonant for a trial input even if no pretrain exemplar's best match
        # was it (distribution shift); without a label it would silently read out
        # as REJECT. Any cluster missing from the majority map is labelled by the
        # class of the pretrain exemplar whose feature best matches its template.
        if not X:
            return
        Is = [self.art._complement(self.art._clip01(x)) for x in X]
        for j in range(self.art.n_categories):
            if j in self.cluster_to_class:
                continue
            ms = [self.art._match(I, self.art.weights[j]) for I in Is]
            self.cluster_to_class[j] = y[int(np.argmax(ms))]

    def snapshot(self):
        return {"n_clusters": self.n_clusters,
                "cluster_to_class": dict(self.cluster_to_class)}


class UnsupervisedFuzzyART(_ARTBase):
    def __init__(self, dim, beta, vigilance, activityFloor,
                 pretrainRate=1.0, pretrainEpochs=5):
        super().__init__(dim, beta, vigilance, activityFloor, pretrainRate, pretrainEpochs)
        self.kind = FUZZY_ART

    def pretrain(self, X, y):
        # Establish the categories at the FAST pretrain rate over several epochs,
        # then drop to the slow trial rate. (Learning at the slow trial rate here
        # would barely form the templates -- the categories must be properly
        # learned from vision before deprivation.)
        self.art.beta = self.pretrain_beta
        for _ in range(max(1, self.pretrain_epochs)):
            for x in X:
                self._recode_or_commit(x)
        self.art.beta = self.learn_beta
        # label each resulting cluster by the majority true class that lands in it
        final = [self.observe(x)[0] for x in X]
        self.cluster_to_class = _majority_map(
            [c for c in final if c != REJECT],
            [yy for c, yy in zip(final, y) if c != REJECT])
        self._label_all_clusters(X, y)
        return self


class SupervisedFuzzyARTMAP(_ARTBase):
    def __init__(self, dim, beta, vigilance, activityFloor,
                 pretrainRate=1.0, pretrainEpochs=5):
        super().__init__(dim, beta, vigilance, activityFloor, pretrainRate, pretrainEpochs)
        self.kind = FUZZY_ARTMAP

    def pretrain(self, X, y):
        # Supervised ARTMAP training (match tracking on labels) at the fast
        # pretrain rate over several epochs, then drop to the slow trial rate.
        self.art.beta = self.pretrain_beta
        for _ in range(max(1, self.pretrain_epochs)):
            for x, label in zip(X, y):
                self.art.train_one(x, label)
        self.art.beta = self.learn_beta
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

    def __init__(self, dim, learningRate, distanceVigilance, activityFloor,
                 pretrainEpochs=5, seed=0):
        self.kind = KNN
        self.dim = int(dim)
        self.eta = float(learningRate)
        self.dvig = float(distanceVigilance)
        self.activity_floor = None if activityFloor is None else float(activityFloor)
        self.pretrain_epochs = int(pretrainEpochs)
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


class SelfOrganizingMap:
    """Kohonen self-organizing map as the unsupervised secondary area.

    A 2-D grid of nodes, each with a weight vector in V1-feature space. Matching
    is EUCLIDEAN (a shared common-mode / DC cancels in x - w, unlike fuzzy ART's
    complement-coded min, which is why the SOM separates the continuous V1 rate
    features fuzzy ART could not). The winner (BMU) and its grid neighbours move
    toward the input; pretraining runs a decaying alpha/sigma schedule to organise
    the map, then the trial uses a small fixed rate for slow drift.

    Readout: BMU index is the "cluster"; its pretrain-majority true class is the
    label (nodes never selected in pretrain are labelled by nearest exemplar, so
    every node is covered). Emergence shows as the map re-tuning / assignments
    migrating (node COUNT is fixed -- set side so there are spare nodes a new
    audio-driven pattern can recruit), tracked by the ARI-vs-class / ARI-vs-clip
    metrics rather than a growing cluster count.
    """

    def __init__(self, dim, learningRate, side=None, classCount=None,
                 activityFloor=0.05, pretrainEpochs=8, seed=0):
        import math
        self.kind = KOHONEN_SOM
        self.dim = int(dim)
        self.eta = float(learningRate)                 # slow trial-time rate
        self.pretrain_epochs = int(pretrainEpochs)
        self.activity_floor = None if activityFloor is None else float(activityFloor)
        k = max(int(classCount or 3), 1)
        self.side = int(side) if side else max(3, int(math.ceil(math.sqrt(k * 4))))
        self.n = self.side * self.side
        self.rng = np.random.RandomState(seed)
        gx, gy = np.meshgrid(np.arange(self.side), np.arange(self.side))
        self.coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(float)  # (n,2)
        self.W = None
        self.cluster_to_class = {}

    @property
    def n_clusters(self):
        return self.n

    def _clip01(self, x):
        return np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0)

    def _too_weak(self, x01):
        return self.activity_floor is not None and float(np.mean(x01)) < self.activity_floor

    def _bmu(self, x):
        d2 = ((self.W - x) ** 2).sum(1)
        return int(np.argmin(d2)), float(d2.min())

    def _update(self, x, alpha, sigma):
        bmu, _ = self._bmu(x)
        dg2 = ((self.coords - self.coords[bmu]) ** 2).sum(1)      # grid distance^2
        h = np.exp(-dg2 / (2.0 * sigma * sigma))                 # neighbourhood
        self.W += (alpha * h)[:, None] * (x - self.W)

    def pretrain(self, X, y):
        X = [self._clip01(x) for x in X]
        Xa = np.asarray(X)
        # init nodes to random training samples (good SOM initialisation)
        idx = self.rng.randint(0, len(Xa), size=self.n)
        self.W = Xa[idx].copy() + self.rng.randn(self.n, self.dim) * 1e-3
        # decaying schedule, enough iterations for a small map to organise
        nIter = max(self.pretrain_epochs * len(Xa), 40 * self.n)
        a0, aF = 0.5, 0.05
        s0, sF = max(self.side / 2.0, 1.0), 0.7
        order = self.rng.permutation(len(Xa))
        for t in range(nIter):
            frac = t / max(nIter - 1, 1)
            alpha = a0 * (aF / a0) ** frac
            sigma = s0 * (sF / s0) ** frac
            x = Xa[order[t % len(Xa)]]
            if t % len(Xa) == 0:
                order = self.rng.permutation(len(Xa))
            self._update(x, alpha, sigma)
        # label nodes by majority true class of the pretrain inputs they win
        bmus = [self._bmu(x)[0] for x in Xa]
        self.cluster_to_class = _majority_map(bmus, list(y))
        for i in range(self.n):                                  # full coverage
            if i not in self.cluster_to_class:
                di = ((Xa - self.W[i]) ** 2).sum(1)
                self.cluster_to_class[i] = y[int(np.argmin(di))]
        return self

    def observe(self, x):
        x01 = self._clip01(x)
        if self._too_weak(x01) or self.W is None:
            return (REJECT, REJECT, 0.0)
        bmu, d2 = self._bmu(x01)
        conf = 1.0 / (1.0 + np.sqrt(d2 / self.dim))
        return (bmu, self.cluster_to_class.get(bmu, REJECT), float(conf))

    def learn(self, x):
        x01 = self._clip01(x)
        if self._too_weak(x01) or self.W is None:
            return
        # slow drift: small fixed rate, tight neighbourhood
        self._update(x01, self.eta, 1.0)

    def snapshot(self):
        return {"n_clusters": self.n, "cluster_to_class": dict(self.cluster_to_class)}
