"""Feature and relevance extraction from an intermediate layer of a classifier."""

from typing import Optional

import torch
from crp.attribution import CondAttribution
from zennit.composites import Composite


class FeatureExtractor(torch.nn.Module):
    """Extract the activations (and their relevance) of one layer of a model.

    A ``FeatureExtractor`` wraps a classifier so that the layer named
    ``feature_name`` is recorded during the forward pass, and its relevances are
    computed with the composite passed in. The relevance--activation product is
    what the OOD test consumes as a feature vector.

    Parameters
    ----------
    model : torch.nn.Module
        The classifier to wrap.
    device : torch.device
        Device the model is moved to and inputs are sent to.
    feature_name : str
        Name of the layer to record. For convolutional classifiers this is
        usually the penultimate pooling layer (``"avgpool"`` for the
        torchvision ResNet/EfficientNet families); for a Vision Transformer it is
        ``"encoder"``, in which case the class token is kept.
    composite : zennit.composites.Composite, optional
        Attribution composite. ``None`` uses the zennit-crp default, which is the
        LRP configuration used in the paper.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_name: str,
        composite: Optional[Composite] = None,
    ):
        super().__init__()
        self.model = model.to(device)
        self.feature_name = feature_name
        self.attribution = CondAttribution(self.model, no_param_grad=True)
        self.composite = composite
        self.device = device

    def _attribution(self, x: torch.Tensor):
        _x = x.to(self.device)
        _x.requires_grad = True
        return self.attribution(
            _x,
            conditions=[{}],
            composite=self.composite,
            record_layer=[self.feature_name],
            init_rel=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the wrapped model's prediction."""
        return self.model(x)

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return activations scaled by their relevance, normalised to unit magnitude.

        The layer's native shape is preserved (for a convolutional layer this is
        ``(B, C, H, W)``), so the result can be visualised as a map.
        """
        attr = self._attribution(x)
        importance = attr.activations[self.feature_name] * attr.relevances[self.feature_name]
        if self.feature_name == "encoder":
            importance = importance[:, 0]
        return importance / torch.max(torch.abs(importance))

    def attribute(self, x: torch.Tensor) -> torch.Tensor:
        """Return the layer's activations as a feature vector of shape ``(B, F)``.

        Spatial dimensions are flattened into the feature axis, which is the
        contract :class:`~stoodx.ood_test.OODTest` relies on. Normalising by the
        maximum absolute activation keeps distances between samples comparable
        across inputs.
        """
        x = x.to(self.device)
        x.requires_grad = True
        attr = self.attribution(
            x,
            conditions=[{}],
            composite=self.composite,
            record_layer=[self.feature_name],
            init_rel=-1,
        )
        importance = attr.activations[self.feature_name]
        if self.feature_name == "encoder":
            importance = importance[:, 0]
        maximum = torch.max(torch.abs(importance))
        maximum = maximum if maximum != 0 else 1
        return (importance / maximum).flatten(1)
