"""The STOOD-X out-of-distribution test."""

from typing import Callable, Optional

import pandas as pd
import torch
from baycomp import two_on_multiple

from .features import FeatureExtractor

COLUMNS = ("d(x,VAL)>d(x_i,VAL)", "d(x,VAL)~d(x_i,VAL)", "d(x,VAL)<d(x_i,VAL)")


class OODTest:
    """Statistical test that decides whether an input is out of distribution.

    The validator is a set of in-distribution samples, represented by their
    features (``add_features``). For a new input the test compares its distance
    to that set against the distances among the set's own members, using the
    Bayesian comparison of Benavoli et al. (2017), with the region of practical
    equivalence set from the observed distance differences.

    Parameters
    ----------
    model : FeatureExtractor
        Wrapped model used both to predict and to extract features.
    distance : callable, optional
        ``distance(a, b)`` returning the distance between two feature vectors.
        The default is the Euclidean norm over the feature dimension.
    k_neighbors : int, optional
        Number of neighbours used for the comparison. Default is 10.
    """

    def __init__(
        self,
        model: FeatureExtractor,
        distance: Callable = lambda x, y: torch.linalg.norm(x - y, dim=-1),
        k_neighbors: int = 10,
    ):
        self.model = model
        self.distance = distance
        self.k_neighbors = k_neighbors
        self.feats: Optional[torch.Tensor] = None
        self.classes: Optional[torch.Tensor] = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the wrapped model's prediction for ``x``."""
        return self.model(x)

    def add_features(self, x: torch.Tensor) -> None:
        """Add a batch of validation samples to the reference set.

        Parameters
        ----------
        x : torch.Tensor
            Batch of shape ``(B, ...)`` in the model's input format.
        """
        feats = self.model.attribute(x)
        classes = self.forward(x).argmax(1).detach()
        if self.feats is None:
            self.feats = feats
            self.classes = classes
        else:
            self.feats = torch.cat((self.feats, feats), dim=0)
            self.classes = torch.cat((self.classes, classes), dim=0)

    def restart_features(self) -> None:
        """Forget the reference set."""
        self.feats = None
        self.classes = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the features of ``x``."""
        return self.model.attribute(x)

    def test(
        self,
        x: torch.Tensor,
        n_features: Optional[int] = None,
        intraclass: bool = True,
    ) -> pd.DataFrame:
        """Decide whether a single input is out of distribution.

        The protocol is:

        1. Compute the input's features and its predicted class.
        2. Select the reference subset: the validation samples of the same
           predicted class when ``intraclass`` is true, the whole reference set
           otherwise.
        3. If ``n_features`` is given, keep the ``n_features`` features with the
           largest presence across the reference subset and zero the rest, in
           both the reference subset and the input.
        4. Take the ``k_neighbors`` closest reference samples to the input, and
           for each of them the ``k_neighbors`` closest reference samples to
           itself.
        5. For every neighbour, run a Bayesian comparison between the distances
           from the input to the reference set and the distances among the
           reference set's own members, with the region of practical equivalence
           set to the 0.75 quantile of the observed distance differences.

        Parameters
        ----------
        x : torch.Tensor
            A single input, of shape ``(...)`` as the model expects, without a
            batch dimension.
        n_features : int, optional
            Number of features to keep. ``None`` keeps all of them.
        intraclass : bool, optional
            Restrict the reference subset to the input's predicted class.
            Default is true.

        Returns
        -------
        pandas.DataFrame
            One row per neighbour, with the posterior probability that the
            input is farther than, equivalent to, or closer than the reference
            set's own members.
        """
        if self.feats is None:
            raise RuntimeError("no reference set: call add_features() first")

        n_features = n_features if n_features is not None else self.feats.shape[1]
        x_features = self.model.attribute(x.unsqueeze(0)).squeeze(0).detach()
        x_class = self.forward(x.unsqueeze(0)).squeeze(0).argmax().item()

        if intraclass:
            mask = self.classes == x_class
            if not bool(mask.any()):
                raise ValueError(
                    f"the reference set has no sample of class {x_class}; "
                    "add features for that class or pass intraclass=False"
                )
            feat_subset = self.feats[mask].clone()
        else:
            feat_subset = self.feats.clone()

        presence = torch.sum(torch.abs(feat_subset), dim=0)
        least_present_idx = torch.argsort(presence, descending=True)[n_features:]

        feat_subset[:, least_present_idx] = 0
        x_features[least_present_idx] = 0

        x_distances = torch.stack([self.distance(x_features, feature) for feature in feat_subset])
        nearest_idx = torch.argsort(x_distances)[: self.k_neighbors]
        x_distances = x_distances[nearest_idx]

        rows = []
        for atom in nearest_idx:
            # distances from this neighbour to the whole reference subset
            atom_distances = torch.stack(
                [self.distance(feat_subset[atom], other) for other in feat_subset]
            )
            # its own nearest neighbours, excluding itself (distance zero)
            neighbour_idx = torch.argsort(atom_distances)[1 : self.k_neighbors + 1]
            distances_i = atom_distances[neighbour_idx]
            if len(distances_i) < 2:
                continue
            diff_distances = distances_i[:, None] - distances_i[None, :]
            rope = torch.quantile(diff_distances.flatten(), 0.75).item()
            bayes_test = two_on_multiple(
                x_distances, distances_i, rope=rope, nsamples=50000
            )
            rows.append([bayes_test[0], bayes_test[1], bayes_test[2]])
        return pd.DataFrame(rows, columns=list(COLUMNS))
