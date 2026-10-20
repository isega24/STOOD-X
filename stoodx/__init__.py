"""STOOD-X: a nonparametric statistical test for explainable out-of-distribution detection.

Reference implementation of the protocol described in

    Sevillano-García, I., Luengo, J., Herrera, F. (2026).
    STOOD-X: Explainable out-of-distribution detection via nonparametric
    statistical testing on large-scale datasets. Pattern Recognition, 177.
    https://doi.org/10.1016/j.patcog.2026.113254

The two public objects are :class:`FeatureExtractor`, which turns a classifier
into a layer-wise feature/relevance extractor, and :class:`OODTest`, which scores
how far a new input sits from the in-distribution validation set and reports the
result of a Bayesian comparison per neighbour.
"""

from .features import FeatureExtractor
from .ood_test import OODTest

__all__ = ["FeatureExtractor", "OODTest"]
__version__ = "0.1.0"
