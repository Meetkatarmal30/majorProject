"""
models/resnet.py - Baseline Sentinel-1 ResNet50 Encoder.

Adapts the torchvision ResNet50 architecture for 2-channel Sentinel-1 SAR imagery
(VV and VH polarizations), producing 2048-D latent embeddings before projection.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50S1Encoder(nn.Module):
    """
    Sentinel-1 SAR feature encoder based on ResNet50.

    Modifications from standard ResNet50:
    1. First convolutional layer (conv1) modified from 3 input channels (RGB)
       to 2 input channels (VV, VH). Pretrained weights are preserved by
       mapping channels 0 and 1, and distributing the channel 2 weights evenly.
    2. Final fully connected classification layer (fc) replaced with nn.Identity(),
       outputting the 2048-dimensional average-pooled latent feature representation.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Modify first layer for 2-channel SAR input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # Transfer pretrained weights from 3-channel to 2-channel conv
        if pretrained:
            with torch.no_grad():
                original_weight = original_conv.weight  # shape: (64, 3, 7, 7)
                # Map channels: preserve VV (ch 0), VH (ch 1), distribute ch 2 evenly
                new_weight = original_weight[:, :2, :, :] + original_weight[:, 2:3, :, :] / 2.0
                self.backbone.conv1.weight.copy_(new_weight)

        # Replace final classification head to output 2048-D average-pooled features
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: PyTorch FloatTensor of shape (B, 2, 224, 224).

        Returns:
            torch.Tensor: Feature embeddings of shape (B, 2048).
        """
        return self.backbone(x)
