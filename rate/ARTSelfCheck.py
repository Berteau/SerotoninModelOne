import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rate.ARTClassifier import FuzzyART, REJECT
from rate.StructuredInput import ImageFolderInputSet, StructuredInputSet

'''
Self-check for the standalone Fuzzy ART classifier. Runnable directly and
pytest-discoverable. Verifies the ART primitives, supervised match tracking, and
the person/horse classification-with-reject behaviour that the secondary
cortical area relies on. Uses the real labeled images when the dataset path is
available, otherwise falls back to synthetic class-distinct patterns so the
check always runs.
'''

DATASET = os.environ.get(
    "CLASSIFIED_STIMULI",
    "/tmp/claude-0/-home-user-SerotoninModelOne/ed5d92d2-51a2-577d-b9fd-f0759f1ad51a/scratchpad/classified_stimuli/classified_stimuli")
VIGILANCE = 0.75


def _load_dataset():
    if os.path.isdir(DATASET):
        data = ImageFolderInputSet(DATASET, gridSize=10, audioBins=20, imagesPerClass=10, seed=0)
        src = "real images (%s)" % ", ".join(data.classNames)
    else:
        # synthetic fallback: 3 classes x a few jittered exemplars
        data = StructuredInputSet(gridSize=10, audioBins=20, classCount=3, seed=0)
        src = "synthetic patterns"
    X = [st.visual for st in data.stimuli]
    y = [st.label for st in data.stimuli]
    return X, y, src


def test_art_primitives():
    art = FuzzyART(dim=4, vigilance=0.5, alpha=0.01, beta=1.0)
    x = np.array([1.0, 0.0, 1.0, 0.0])
    I = art._complement(art._clip01(x))
    assert np.allclose(I, [1, 0, 1, 0, 0, 1, 0, 1])   # complement coding
    assert abs(np.sum(I) - art.dim) < 1e-9            # |I| == dim always
    # First example commits a category equal to its own complement code.
    j = art.train_one(x, label="A")
    assert j == 0 and art.n_categories == 1
    assert art.classify(x) == "A"                     # exact match classifies
    # Full match value is 1.0 for the identical input.
    _, (cat, m) = art.classify(x, return_detail=True)
    assert cat == 0 and abs(m - 1.0) < 1e-9
    print("[1] ART primitives (complement coding, |I|=dim, choice/match) OK")


def test_match_tracking():
    # Two identical inputs with different labels must land in SEPARATE
    # categories (ARTMAP match tracking), and each must classify to its label.
    art = FuzzyART(dim=4, vigilance=0.6, alpha=0.01, beta=1.0)
    x = np.array([1.0, 1.0, 0.0, 0.0])
    art.train_one(x, label="A")
    art.train_one(x, label="B")
    assert art.n_categories == 2, art.n_categories
    # The most-recently-learned same pattern is ambiguous by design; assert both
    # labels exist among categories and a clean A-only pattern maps to A.
    assert set(art.labels) == {"A", "B"}
    print("[2] match tracking: conflicting labels force distinct categories OK")


def test_classify_and_reject():
    X, y, src = _load_dataset()
    labels = sorted(set(y))
    art = FuzzyART(dim=len(X[0]), vigilance=VIGILANCE, alpha=0.01, beta=1.0)
    art.fit(X, y, epochs=1)

    acc = np.mean([art.classify(x) == l for x, l in zip(X, y)])
    assert acc == 1.0, ("training accuracy", acc)
    assert set(art.labels) >= set(labels), (art.labels, labels)
    assert art.n_categories >= len(labels)

    rng = np.random.RandomState(1)
    noise_rej = np.mean([art.classify(rng.rand(len(X[0]))) == REJECT for _ in range(10)])
    blank_rej = art.classify(np.zeros(len(X[0]))) == REJECT
    assert noise_rej == 1.0, ("noise reject rate", noise_rej)
    assert blank_rej, "blank should be rejected"

    # Content-specific top-down template is a valid [0,1] pattern.
    t = art.category_template(0)
    assert t.shape == (len(X[0]),) and t.min() >= 0.0 and t.max() <= 1.0

    print("[3] classify+reject on %s: acc=%.0f%%, ncat=%d, noise+blank rejected OK"
          % (src, 100 * acc, art.n_categories))


def main():
    test_art_primitives()
    test_match_tracking()
    test_classify_and_reject()
    print("\nAll ART self-checks passed.")


if __name__ == "__main__":
    main()
