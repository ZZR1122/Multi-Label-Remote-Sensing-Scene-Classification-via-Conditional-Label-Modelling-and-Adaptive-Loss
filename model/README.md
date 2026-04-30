# Model Weights

Download or place the DINOv3 ViT-S/16 pretrained checkpoint locally at:

```text
model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Training scripts will create additional checkpoints such as:

```text
model/best_model_primary.pth
model/best_model_conditional_multilabel_asl.pth
model/best_model_conditional_multilabel_aslgb.pth
```

Evaluation scripts expect checkpoints at the default paths shown in their `if __name__ == "__main__":` blocks.
