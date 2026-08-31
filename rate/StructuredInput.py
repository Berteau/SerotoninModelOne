import os

import numpy as np
from scipy.ndimage import sobel

'''
Structured audio-visual input for the retinotopic grand-plan architecture.

Two design points from the grand-plan spec are honored here:

  1. Visual input is a still image run through EDGE DETECTION, reduced to a
     G x G retinotopic grid of values in [0, 1], carrying a class LABEL. This
     module synthesizes simple, class-distinct greyscale patterns and applies a
     Sobel edge filter so the pipeline is self-contained and runnable, but the
     same interface (grid + label) accepts real labeled images (e.g. the
     person/horse folders) by swapping _class_image().

  2. Audio is only r = 0.25 correlated with the visual input. The audio vector
     (A tonotopic bins) is constructed so the SAMPLE Pearson correlation between
     the audio and the pooled visual pattern is exactly r = 0.25, via

         a = r * z_v + sqrt(1 - r^2) * e_perp

     where z_v is the standardized pooled-visual vector and e_perp is Gaussian
     noise Gram-Schmidt-orthogonalized against z_v and restandardized. Because
     z_v and e_perp are sample-orthogonal unit-variance vectors, corr(a, z_v) =
     r for THIS finite sample, not merely in expectation (the plain shared-latent
     formula a = r*z_v + sqrt(1-r^2)*e leaves sampling error of order 1/sqrt(A)
     around r, large for A ~ 20 bins). Min-max rescaling to [0, 1] afterward is a
     positive affine map and preserves Pearson correlation, so the target r holds.

Rates are obtained by scaling the [0, 1] values to a firing-rate range
(minRateHz .. maxRateHz), matching the rate model's input-lambda convention.
'''

TARGET_AV_CORRELATION = 0.25


def _standardize(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0.0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def _minmax01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo == 0.0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _pool_to_length(vec, length):
    # Average-pool a 1-D vector to the requested length (block means). Used to
    # bring the G*G visual vector down to the A-bin audio dimensionality before
    # correlating, the audio analogue of feedforward pooling.
    vec = np.asarray(vec, dtype=float)
    if length >= len(vec):
        # simple repeat-interpolate up (rare; A is usually < G*G)
        idx = np.linspace(0, len(vec) - 1, length).round().astype(int)
        return vec[idx]
    edges = np.linspace(0, len(vec), length + 1).astype(int)
    return np.array([vec[edges[i]:edges[i + 1]].mean() for i in range(length)])


class StructuredAVStimulus:
    def __init__(self, label, visualGrid, audioVector):
        self.label = label                 # class index (int)
        self.visualGrid = visualGrid       # (G, G) float array in [0, 1]
        self.visual = visualGrid.reshape(-1)   # flattened, length G*G
        self.audio = audioVector           # length A, float in [0, 1]


class StructuredInputSet:
    '''
    A tiny labeled audio-visual dataset. classCount distinct visual prototypes
    (edge-detected), each paired with an r=0.25-correlated audio vector.
    '''

    def __init__(self, gridSize=10, audioBins=20, classCount=2,
                 minRateHz=0.0, maxRateHz=30.0, seed=0):
        self.gridSize = gridSize
        self.audioBins = audioBins
        self.classCount = classCount
        self.minRateHz = minRateHz
        self.maxRateHz = maxRateHz
        self.rng = np.random.RandomState(seed)
        self.stimuli = [self._make_stimulus(label, grid)
                        for (label, grid) in self._prototype_grids()]

    # ---- prototype source (override for real images) ----

    def _prototype_grids(self):
        # (label, raw greyscale G x G grid in [0,1]) pairs. The base class
        # synthesizes class-distinct patterns; ImageFolderInputSet overrides this
        # to load real labeled images. Edge detection and the r=0.25 audio
        # construction happen downstream in _make_stimulus, identically for both.
        return [(c, self._class_image(c)) for c in range(self.classCount)]

    # ---- stimulus construction ----

    def _class_image(self, label):
        # Synthesize a class-distinct greyscale image (G x G). Class 0 is
        # dominated by vertical structure, class 1 by horizontal structure,
        # further classes by diagonal bands -- enough for edge detection to
        # produce visibly different maps. Swap this for real image loading to
        # use the person/horse dataset.
        g = self.gridSize
        yy, xx = np.mgrid[0:g, 0:g]
        if label % 3 == 0:
            img = np.sin(2 * np.pi * xx / max(2, g / 3.0))
        elif label % 3 == 1:
            img = np.sin(2 * np.pi * yy / max(2, g / 3.0))
        else:
            img = np.sin(2 * np.pi * (xx + yy) / max(2, g / 3.0))
        img = _minmax01(img)
        return img

    def _edge_detect(self, img):
        gx = sobel(img, axis=0, mode="reflect")
        gy = sobel(img, axis=1, mode="reflect")
        mag = np.hypot(gx, gy)
        return _minmax01(mag).reshape(self.gridSize, self.gridSize)

    def _correlated_audio(self, visualFlat):
        r = TARGET_AV_CORRELATION
        v_pooled = _pool_to_length(visualFlat, self.audioBins)
        z_v = _standardize(v_pooled)
        if not np.any(z_v):
            # degenerate (constant) pooled visual: correlation undefined, return noise
            return _minmax01(self.rng.randn(self.audioBins))
        e = self.rng.randn(self.audioBins)
        # Gram-Schmidt: remove the z_v component so e_perp is sample-orthogonal
        # to z_v (dot product 0), then restandardize. Orthogonality survives the
        # mean-subtraction in _standardize because z_v sums to 0.
        e = e - (e @ z_v) / (z_v @ z_v) * z_v
        e_perp = _standardize(e)
        a_std = r * z_v + np.sqrt(1.0 - r * r) * e_perp
        return _minmax01(a_std)

    def _make_stimulus(self, label, rawImage):
        edges = self._edge_detect(rawImage)
        audio = self._correlated_audio(edges.reshape(-1))
        return StructuredAVStimulus(label, edges, audio)

    # ---- rate conversion ----

    def _to_rate(self, unit01):
        return self.minRateHz + (self.maxRateHz - self.minRateHz) * np.asarray(unit01, dtype=float)

    def visualRates(self, stimulus):
        return self._to_rate(stimulus.visual)

    def audioRates(self, stimulus):
        return self._to_rate(stimulus.audio)

    # ---- verification helper ----

    def measuredAVCorrelation(self, stimulus):
        # Empirical Pearson r between the audio vector and the pooled visual,
        # for the r=0.25 self-check.
        v_pooled = _pool_to_length(stimulus.visual, self.audioBins)
        if v_pooled.std() == 0 or stimulus.audio.std() == 0:
            return float("nan")
        return float(np.corrcoef(v_pooled, stimulus.audio)[0, 1])


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def load_image_grid(path, gridSize):
    """Load an image file as a greyscale G x G grid in [0, 1].

    Uses Pillow (already required by matplotlib). The image is converted to
    luminance and resized to the retinotopic grid; edge detection and the
    r=0.25 audio pairing are applied downstream, identically to synthetic input.
    """
    from PIL import Image
    img = Image.open(path).convert("L").resize((gridSize, gridSize))
    return np.asarray(img, dtype=float) / 255.0


class SoundBank:
    """A fixed bank of `size` distinct sound clips (tonotopic vectors in [0,1]).

    Each clip is a fixed random pattern (seeded). Class c has a canonical clip
    (index c mod size); the presenter plays it with probability matchProb, else a
    uniformly random clip, so clip-ID is loosely correlated with class-ID. The
    bank may be larger than the class count, so some clips carry structure not
    tied to any visual class -- material for novel auditory categories.
    """

    def __init__(self, size, audioBins, seed=0):
        self.size = int(size)
        self.audioBins = int(audioBins)
        rng = np.random.RandomState(seed)
        self.clips = [_minmax01(rng.randn(audioBins)) for _ in range(self.size)]

    def canonical_clip(self, classLabel):
        return classLabel % self.size

    def clip_vector(self, clipId):
        return self.clips[clipId]


class SequencePresenter:
    """Wraps an InputSet (synthetic or ImageFolderInputSet) and adds (a) a fixed
    sound bank whose clip-ID is loosely correlated with image class-ID, and (b) a
    seeded, reproducible, randomly-shuffled presentation sequence.

    A presentation is a (stimulusIndex, clipId) pair. Visual rates come from the
    wrapped set; audio rates come from the chosen clip. Correlation strength is
    set by matchProb (prob the canonical clip for the image's class is played).
    """

    def __init__(self, inputSet, soundBankSize=None, matchProb=0.25, seed=0):
        self.inputSet = inputSet
        self.audioBins = inputSet.audioBins
        self.classNames = getattr(inputSet, "classNames", None)
        self.classCount = inputSet.classCount
        self.matchProb = float(matchProb)
        if soundBankSize is None:
            soundBankSize = self.classCount
        self.bank = SoundBank(soundBankSize, self.audioBins, seed=seed + 101)

    # ---- sequence ----

    def generate_sequence(self, nPresentations, seed=0):
        """List of (stimulusIndex, clipId), length nPresentations. Class exposure
        is balanced (tiled permutations of the stimulus list); the clip for each
        presentation is the canonical clip w.p. matchProb else a uniform-random
        bank clip. Reproducible from seed."""
        rng = np.random.RandomState(seed)
        nStim = len(self.inputSet.stimuli)
        order = []
        while len(order) < nPresentations:
            perm = list(rng.permutation(nStim))
            order.extend(perm)
        order = order[:nPresentations]
        seq = []
        for si in order:
            label = self.inputSet.stimuli[si].label
            if rng.rand() < self.matchProb:
                clip = self.bank.canonical_clip(label)
            else:
                clip = int(rng.randint(self.bank.size))
            seq.append((si, clip))
        return seq

    # ---- rate conversion ----

    def visualRates(self, stimulusIndex):
        st = self.inputSet.stimuli[stimulusIndex]
        return self.inputSet.visualRates(st)

    def audioRates(self, clipId):
        return self.inputSet._to_rate(self.bank.clip_vector(clipId))

    def true_label(self, stimulusIndex):
        return self.inputSet.stimuli[stimulusIndex].label

    # ---- verification ----

    def measuredClipClassAssociation(self, sequence):
        """Cramer's V between clip-ID and class-ID over a sequence, the categorical
        analogue of the old r = 0.25 audio-visual correlation (0 = independent,
        1 = deterministic). Use to calibrate matchProb."""
        labels = [self.inputSet.stimuli[si].label for si, _ in sequence]
        clips = [c for _, c in sequence]
        K = self.classCount
        M = self.bank.size
        table = np.zeros((K, M))
        for lab, cl in zip(labels, clips):
            table[lab, cl] += 1
        n = table.sum()
        if n == 0:
            return float("nan")
        row = table.sum(1, keepdims=True)
        col = table.sum(0, keepdims=True)
        expected = row @ col / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
        denom = n * (min(K, M) - 1)
        return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


class ImageFolderInputSet(StructuredInputSet):
    """StructuredInputSet sourced from a folder of labeled images.

    Expects `root` to contain one subfolder per class (e.g. person/, horse/);
    subfolder names sorted alphabetically become the class labels 0..K-1. Up to
    imagesPerClass images are loaded per class (deterministically, sorted by
    filename). Everything downstream -- Sobel edge detection, the exact r=0.25
    audio pairing, rate conversion -- is inherited unchanged, so the real-image
    stimuli are drop-in compatible with the retinotopic experiment and the
    self-check's correlation guarantee.

    self.stimuli is a flat list ordered by (class, image); self.classNames maps
    label index -> folder name. The sensory-loss experiment uses stimuli[0].
    """

    def __init__(self, root, gridSize=10, audioBins=20, imagesPerClass=1,
                 minRateHz=0.0, maxRateHz=30.0, seed=0):
        self.root = root
        self.imagesPerClass = imagesPerClass
        self.classNames, self._loadedGrids = self._loadGrids(root, gridSize, imagesPerClass)
        super().__init__(gridSize=gridSize, audioBins=audioBins,
                         classCount=len(self.classNames),
                         minRateHz=minRateHz, maxRateHz=maxRateHz, seed=seed)

    def _loadGrids(self, root, gridSize, imagesPerClass):
        classDirs = sorted(d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d)))
        if not classDirs:
            raise ValueError("No class subfolders found under %r" % root)
        classNames = []
        grids = []   # list of (label, grid01)
        for label, name in enumerate(classDirs):
            classNames.append(name)
            d = os.path.join(root, name)
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith(IMAGE_EXTENSIONS))
            if not files:
                raise ValueError("No images in class folder %r" % d)
            for f in files[:imagesPerClass]:
                grids.append((label, load_image_grid(os.path.join(d, f), gridSize)))
        return classNames, grids

    def _prototype_grids(self):
        return self._loadedGrids

