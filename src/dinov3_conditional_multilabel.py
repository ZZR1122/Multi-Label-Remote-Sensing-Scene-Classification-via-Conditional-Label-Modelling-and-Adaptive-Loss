import torch
import torch.nn as nn

from dinov3_backbone import DINOv3ViTB16Backbone


# Refines label tokens with self-attention and image context.
class LabelCorrelationBlock(nn.Module):
    """One correlation block with self-attention, cross-attention, and an FFN."""

    def __init__(self, label_embed_dim, num_heads, dropout):
        super().__init__()
        self.label_attn = nn.MultiheadAttention(
            label_embed_dim,
            num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            label_embed_dim,
            num_heads,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(label_embed_dim, label_embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(label_embed_dim * 4, label_embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(label_embed_dim)
        self.ln2 = nn.LayerNorm(label_embed_dim)
        self.ln3 = nn.LayerNorm(label_embed_dim)

    def forward(self, label_emb, cross_context):
        """Refine label embeddings using label-label and label-context interactions."""

        # Step 1: let label tokens exchange information with each other.
        label_attn_out, _ = self.label_attn(
            label_emb,
            label_emb,
            label_emb,
            need_weights=False,
        )
        label_emb = self.ln1(label_emb + label_attn_out)

        # Step 2: let each label token query the image tokens and condition tokens.
        cross_attn_out, _ = self.cross_attn(
            label_emb,
            cross_context,
            cross_context,
            need_weights=False,
        )
        label_emb = self.ln2(label_emb + cross_attn_out)

        # Step 3: refine each label representation independently.
        ff_out = self.ff(label_emb)
        return self.ln3(label_emb + ff_out)


# Predicts secondary labels from image tokens and primary-label context.
class LabelCorrelationHead(nn.Module):
    """Model dependencies between labels while attending to image and condition tokens."""

    def __init__(
            self,
            num_classes,
            in_features,
            cond_dim,
            label_embed_dim,
            num_heads,
            num_layers,
            image_tokens,
            cond_tokens,
            dropout,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.max_image_tokens = image_tokens
        self.cond_tokens = cond_tokens

        self.label_emb = nn.Parameter(torch.randn(num_classes, label_embed_dim) * 0.01)
        self.patch_token_proj = nn.Linear(in_features, label_embed_dim)
        self.cond_token_proj = nn.Linear(cond_dim, label_embed_dim * cond_tokens)

        self.layers = nn.ModuleList(
            LabelCorrelationBlock(
                label_embed_dim=label_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

        self.logit_attn = nn.MultiheadAttention(label_embed_dim, num_heads, batch_first=True)
        self.logit_mlp = nn.Sequential(
            nn.Linear(label_embed_dim * 2, label_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(label_embed_dim, 1),
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def _build_image_tokens(
            self,
            patch_tokens,
    ):
        """Project backbone patch tokens into the label-attention space."""

        image_tokens = self.patch_token_proj(patch_tokens)
        if (
                self.max_image_tokens is not None
                and 0 < self.max_image_tokens < image_tokens.size(1)
        ):
            # Keep a spatially distributed subset instead of averaging patch tokens.
            idx = torch.linspace(
                0,
                image_tokens.size(1) - 1,
                steps=self.max_image_tokens,
                device=image_tokens.device,
            ).round().long()
            image_tokens = image_tokens.index_select(1, idx)

        return image_tokens

    def forward(
            self,
            patch_tokens,
            cond,
    ):
        """Fuse label tokens with image tokens and primary-label condition tokens."""

        batch_size = patch_tokens.size(0)
        label_emb = self.label_emb.unsqueeze(0).expand(batch_size, -1, -1)

        img_tok = self._build_image_tokens(patch_tokens)
        # The primary-label one-hot vector is expanded into a short learned token sequence.
        cond_tok = self.cond_token_proj(cond).view(batch_size, self.cond_tokens, self.label_embed_dim)
        cross_context = torch.cat([img_tok, cond_tok], dim=1)

        for layer in self.layers:
            label_emb = layer(label_emb, cross_context)

        # A final attention readout extracts label-specific spatial evidence from the image.
        spatial_evidence, _ = self.logit_attn(
            label_emb,
            img_tok,
            img_tok,
            need_weights=False,
        )
        logits = self.logit_mlp(torch.cat([label_emb, spatial_evidence], dim=-1)).squeeze(-1)
        return logits + self.bias


# Combines the DINOv3 backbone with the conditional multilabel head.
class DINOv3ConditionalMultilabelClassifier(nn.Module):
    """Backbone plus correlation head for conditional secondary-label prediction."""

    def __init__(
            self,
            num_classes,
            cond_dim,
            label_embed_dim,
            num_heads,
            num_layers,
            image_tokens,
            cond_tokens,
            dropout,
    ):
        super().__init__()
        self.backbone = DINOv3ViTB16Backbone(

        )
        self.head = LabelCorrelationHead(
            num_classes=num_classes,
            in_features=self.backbone.embed_dim,
            cond_dim=cond_dim,
            label_embed_dim=label_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            image_tokens=image_tokens,
            cond_tokens=cond_tokens,
            dropout=dropout,
        )

    def forward(self, x, cond):
        """Encode the image, keep dense patch tokens, and predict secondary logits."""

        features = self.backbone.forward_feature_dict(x)
        return self.head(features["patch_tokens"], cond)
