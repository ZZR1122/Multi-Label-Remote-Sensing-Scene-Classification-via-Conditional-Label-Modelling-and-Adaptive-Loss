import torch
import torch.nn as nn

from dinov3_backbone import DINOv3ViTB16Backbone


# Predicts the single primary scene category from DINOv3 features.
class DINOv3PrimaryClassifier(nn.Module):
    """Map the backbone CLS token to primary-category logits."""

    def __init__(self, num_classes, ):
        super().__init__()
        self.backbone = DINOv3ViTB16Backbone(
        )
        # The backbone already produces a semantic image representation,
        # so a single linear layer is sufficient for the primary-label task.
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x) :
        """Return raw class logits for cross-entropy training and evaluation."""

        features = self.backbone(x)
        return self.classifier(features)
