import torch
import torch.nn as nn

from dinov3_backbone import DINOv3ViTB16Backbone
from dinov3_conditional_multilabel import LabelCorrelationHead


# Builds the shared DINOv3 backbone for secondary baselines.
def build_dinov3_backbone():
    """Construct the shared pretrained DINOv3 ViT-S/16 backbone."""

    return DINOv3ViTB16Backbone(
        repo_dir="./dinov3",
        weights_path="./model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        model_name="dinov3_vits16",
    )


# Uses a linear head for independent secondary-label prediction.
class DINOv3LinearMultilabelClassifier(nn.Module):
    """DINOv3 backbone followed by a plain linear multilabel classifier."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = build_dinov3_backbone()
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        return self.classifier(self.backbone(x))


DINOv3BCEClassifier = DINOv3LinearMultilabelClassifier
DINOv3ASLClassifier = DINOv3LinearMultilabelClassifier
DINOv3ASLGBClassifier = DINOv3LinearMultilabelClassifier


# Uses the correlation head for conditional BCE secondary prediction.
class DINOv3BCECorrelationClassifier(nn.Module):
    """DINOv3 backbone followed by the conditional label-correlation head."""

    def __init__(self, num_classes, cond_dim):
        super().__init__()
        self.backbone = build_dinov3_backbone()
        self.head = LabelCorrelationHead(
            num_classes=num_classes,
            in_features=self.backbone.embed_dim,
            cond_dim=cond_dim,
            label_embed_dim=384,
            num_heads=8,
            num_layers=3,
            image_tokens=None,
            cond_tokens=8,
            dropout=0.1,
        )

    def forward(self, x, cond):
        features = self.backbone.forward_feature_dict(x)
        return self.head(features["patch_tokens"], cond)
