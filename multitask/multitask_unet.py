# import torch
import torch.nn as nn
# import torch.nn.functional as F
from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    def __init__(self,
                 spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 num_class_labels=2,   # 2 for binary, >2 for multiclass
                 features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        # Classification head: take encoder bottleneck features and pool
        # For MONAI UNet, we use last encoder feature map before upsampling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], num_class_labels)

    def forward(self, x):
        # UNet's forward returns segmentation output, but we want bottleneck too
        # We'll use the UNet's forward features if available, or hack via hooks,
        # but simplest: pass x through encoder, then use decoder
        # Instead, let's pass x through self.unet and also tap bottleneck

        # To get encoder output, MONAI UNet doesn't expose directly.
        # So we use the segmentation output for demo, but in production,
        # subclass UNet or use encoder separately for best results.
        seg_out = self.unet(x)  # [B, out_channels, H, W]
        # Use pooled segmentation output for classification (demo version)
        pooled = self.gap(seg_out)
        pooled = pooled.view(pooled.size(0), -1)
        class_logits = self.classifier(pooled)
        return class_logits, seg_out
