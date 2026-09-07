# S1 Baseline Training Data & Integration Readiness

This report assesses the availability of dataset splits, pre-computed embeddings, and model components for training the Sentinel-1 (S1) baseline encoder.

---

## 1. Available Training Data
* **S1/SAR Training Patches**: **Available**. 
  * The raw dataset contains approximately 549,488 patches.
  * These can be loaded programmatically using `BigEarthNetS1Dataset(split="train")`.
  * The dataset class automatically performs outlier clipping, DB scaling, and Z-score standardization.

---

## 2. Missing Training Data
* **S2/MS Training Data / Embeddings**: **Missing**.
  * We do **not** have the S2 (multispectral) raw image patches for the training set on disk.
  * We do **not** have pre-computed training embeddings corresponding to the S1 training patches.
  * The file `baseline_val_ms_embeddings.npy` contains **only the 4,500 validation-set embeddings** and cannot be used for training.
  * *Additional Requirement*: We need either S2/MS raw training patches or pre-computed S2 training embeddings (`baseline_train_ms_embeddings.npy`) from the teammate to train the cross-modal model.

---

## 3. Existing S2/MS Components
* **S2/MS Encoder Implementation**: **Missing**.
  * There is no S2/MS encoder model file or checkpoint inside the repository.
  * S2/MS feature extraction cannot be performed dynamically during joint training unless the S2/MS model checkpoint is provided.

---

## 4. Proposed S1 Model Interface
The proposed S1 encoder will expose the following interface:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50S1Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Modify the first layer to accept 2 channels (VV and VH)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Average the original 3-channel weights to initialize the 2-channel layer
        if pretrained:
            with torch.no_grad():
                original_weight = original_conv.weight
                # Map 3 channels to 2 channels by taking mean
                new_weight = original_weight[:, :2] + original_weight[:, 2:3] / 2.0
                self.backbone.conv1.weight.copy_(new_weight)
                
        # Remove the final classification layer to return average pooled features
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 2, 120, 120)
        # Output shape: (B, 2048)
        return self.backbone(x)
```

---

## 5. Proposed Training Input/Output
* **Inputs**:
  * S1 Batch Tensor: shape `(B, 2, 120, 120)`.
  * S2/MS Batch Embeddings (or S2 Batch Tensor): shape `(B, 2048)`.
* **Outputs**:
  * S1 Projected Embeddings: shape `(B, 2048)`.
  * S2 Projected Embeddings: shape `(B, 2048)`.
* **Objective**: Align S1 and S2 representations using InfoNCE / CLIP-style contrastive loss.

---

## 6. Training Integration & Hyperparameters
* **Unspecified Parameters**: 
  * The project **does not specify** a particular loss temperature ($\tau$), optimizer (Adam/SGD), learning rate, batch size, training epochs, or evaluation loop setup. 
  * *Important*: To maintain project integrity, these parameters will remain undefined until training-side requirements are clarified or provided by the teammate.
* **Validation Isolation**: 
  * The 4,500 validation patch IDs listed in `teammate_inputs/baseline_val_patch_ids.txt` will be loaded and explicitly excluded from the training dataset/loader during training to prevent data leakage.

---

## 7. What Must Be Obtained or Implemented Next
1. **Obtain S2 Training Data**: Retrieve either raw S2 patches or pre-computed S2 training embeddings (`baseline_train_ms_embeddings.npy`) from the teammate.
2. **Implement Model Code**: Create the S1 baseline model definition (`models/resnet.py`).
3. **Wait for Training Authorization**: Do not initiate training loops until training inputs are provided.
