"""Tests for the OOD test itself.

A stub extractor replaces the real one, so these tests need no attribution and no
model weights: they check the reference-set bookkeeping and the guards, which is
where the interesting failure modes live.
"""

import pytest
import torch

from stoodx import OODTest


class StubExtractor(torch.nn.Module):
    """Deterministic stand-in for FeatureExtractor: identity features, stub logits."""

    def __init__(self, n_classes: int = 4, n_features: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)[:, : self.n_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)[:, : self.n_classes]


def scattered_reference_set(n: int = 40, n_features: int = 8) -> torch.Tensor:
    """Samples whose predicted class varies, so both modes have something to do."""
    return torch.rand(n, n_features)


def single_class_reference_set(n: int = 40, n_features: int = 8) -> torch.Tensor:
    """Samples whose stub logits always peak at class 0."""
    x = torch.zeros(n, n_features)
    x[:, 1:] = -1.0
    return x


def test_both_intraclass_modes_run():
    test = OODTest(StubExtractor(), k_neighbors=3)
    test.add_features(scattered_reference_set())

    for intraclass in (True, False):
        scores = test.test(torch.zeros(8), intraclass=intraclass)
        assert list(scores.columns) == [
            "d(x,VAL)>d(x_i,VAL)",
            "d(x,VAL)~d(x_i,VAL)",
            "d(x,VAL)<d(x_i,VAL)",
        ]
        assert 1 <= len(scores) <= 3


def test_intraclass_restricts_the_reference_subset():
    """The mask must select the predicted class, not the whole set."""
    test = OODTest(StubExtractor(), k_neighbors=3)
    test.add_features(scattered_reference_set())

    predicted = int(test.forward(torch.zeros(1, 8)).argmax())
    expected = int((test.classes == predicted).sum())
    assert 0 < expected < len(test.classes), "reference set should mix classes"

    scores = test.test(torch.zeros(8), intraclass=True)
    assert len(scores) == min(3, expected)


def test_intraclass_raises_when_the_class_is_absent():
    test = OODTest(StubExtractor(), k_neighbors=3)
    test.add_features(single_class_reference_set())

    unseen = torch.zeros(8)
    unseen[1] = 1.0  # stub logits peak at class 1, absent from the reference set

    with pytest.raises(ValueError, match="no sample of class 1"):
        test.test(unseen, intraclass=True)


def test_test_does_not_mutate_the_reference_set():
    test = OODTest(StubExtractor(), k_neighbors=3)
    test.add_features(scattered_reference_set())
    before = test.feats.clone()

    test.test(torch.zeros(8), n_features=4, intraclass=False)
    test.test(torch.zeros(8), n_features=4, intraclass=False)

    assert torch.equal(test.feats, before), "test() must not zero the stored features"


def test_missing_reference_set_raises():
    test = OODTest(StubExtractor(), k_neighbors=3)
    with pytest.raises(RuntimeError, match="add_features"):
        test.test(torch.zeros(8))


def test_restart_features_clears_the_reference_set():
    test = OODTest(StubExtractor(), k_neighbors=3)
    test.add_features(scattered_reference_set())
    test.restart_features()

    assert test.feats is None and test.classes is None
