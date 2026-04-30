from pathlib import Path
import torch
import torch.nn as nn


# https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
# Wraps the local DINOv3 encoder used by project models.
class DINOv3Backbone(nn.Module):
    """Load the local DINOv3 checkpoint and expose CLS and patch-token features."""

    def __init__(self, repo_dir="./dinov3",
                 weights_path="./model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth", model_name="dinov3_vits16"):
        super().__init__()
        repo_dir = Path(repo_dir)
        weights_path = Path(weights_path)

        # Load the encoder from the local repository and force the pretrained checkpoint
        # to come from the explicit .pth path instead of any torch.hub cache directory.
        self.encoder = torch.hub.load(
            str(repo_dir),
            model_name,
            source="local",
            weights=str(weights_path),
        )
        self.embed_dim = self.encoder.embed_dim

    def forward_feature_dict(self, x):
        """Return both the global CLS token and dense patch tokens."""

        outputs = self.encoder.forward_features(x)
        return {
            "cls_token": outputs["x_norm_clstoken"],
            "patch_tokens": outputs["x_norm_patchtokens"],
        }

    def forward_features(self, x):
        """Return the CLS token used as the image-level representation."""

        return self.forward_feature_dict(x)["cls_token"]

    def forward(self, x):
        """Mirror the standard backbone interface and expose CLS features only."""

        return self.forward_features(x)


DINOv3ViTB16Backbone = DINOv3Backbone
