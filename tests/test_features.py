"""Tests for the feature extractor, and an end-to-end run of the test itself.

A small convolutional network is used instead of a torchvision model so the suite
runs offline and never downloads weights.
"""

import torch
import torch.nn as nn

from stoodx import FeatureExtractor, OODTest


class TinyNet(nn.Module):
    """A four-class convolutional classifier with a named pooling layer."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def extractor():
    torch.manual_seed(0)
    model = TinyNet().eval()
    return FeatureExtractor(model=model, device=torch.device("cpu"), feature_name="pool")


def test_forward_returns_logits():
    assert extractor().forward(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_feature_activations_keep_the_layer_shape():
    acts = extractor().feature_activations(torch.randn(1, 3, 32, 32))
    assert acts.shape == (1, 16, 1, 1)
    assert torch.isfinite(acts).all()


def test_attribute_returns_a_feature_matrix():
    feats = extractor().attribute(torch.randn(3, 3, 32, 32))
    assert feats.shape == (3, 16), "attribute() must flatten to (B, F) for the test to use it"
    assert torch.isfinite(feats).all()
    assert torch.max(torch.abs(feats)) <= 1.0 + 1e-6


def test_end_to_end_out_of_distribution_scores():
    """Reference set from the model's own inputs; score the same distribution."""
    ext = extractor()
    test = OODTest(ext, k_neighbors=3)

    torch.manual_seed(1)
    test.add_features(torch.randn(24, 3, 32, 32))

    scores = test.test(torch.randn(3, 32, 32), intraclass=False)

    assert len(scores) == 3
    assert list(scores.columns) == [
        "d(x,VAL)>d(x_i,VAL)",
        "d(x,VAL)~d(x_i,VAL)",
        "d(x,VAL)<d(x_i,VAL)",
    ]
