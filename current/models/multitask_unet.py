# multitask_unet.py
import torch.nn as nn
from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    """
    U-Net-based architecture for joint segmentation and classification.
    Returns:
        class_logits: [B, num_classes] - Classification logits.
        seg_out: [B, out_channels, H, W] - Segmentation mask output.
    """
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=1, num_classes=1, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], num_classes)
        self._bottleneck = None

        # Automatically register a hook on the correct (deepest) encoder Conv2d layer
        self._register_bottleneck_hook(features[-1])

    def _register_bottleneck_hook(self, bottleneck_channels):
        # Find the first Conv2d with out_channels == features[-1] and register hook
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == bottleneck_channels:
                # print(f"[MultiTaskUNet] Registering bottleneck forward hook on: {name}")
                module.register_forward_hook(self.save_bottleneck)
                break
        else:
            raise RuntimeError("Could not find a Conv2d layer with the expected bottleneck channels to register the hook.")

    def save_bottleneck(self, module, input, output):
        self._bottleneck = output

    def forward(self, x):
        self._bottleneck = None  # Clear previous value
        seg_out = self.unet(x)   # [B, out_channels, H, W]
        if self._bottleneck is None:
            raise RuntimeError("Bottleneck feature was not set by forward hook!")
        pooled = self.gap(self._bottleneck)
        pooled = pooled.view(pooled.size(0), -1)  # [B, features[-1]]
        class_logits = self.classifier(pooled)     # [B, num_classes]
        # Sanity checks
        assert class_logits.shape[0] == seg_out.shape[0], "Batch size mismatch"
        assert class_logits.ndim == 2, f"Expected [B, num_classes], got {class_logits.shape}"
        assert seg_out.ndim == 4, f"Expected [B, C, H, W], got {seg_out.shape}"
        # print(f"[MultiTaskUNet] class_logits: {class_logits.shape}, seg_out: {seg_out.shape}")
        # return class_logits, seg_out
        return {"class_logits": class_logits, "seg_out": seg_out}
