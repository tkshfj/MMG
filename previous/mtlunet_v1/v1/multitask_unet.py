# multitask_unet.py
import torch.nn as nn
from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    def __init__(
        self, spatial_dims=2, in_channels=1, out_channels=1,
        num_class_labels=1, features=(32, 64, 128, 256, 512)
    ):
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
        self.classifier = nn.Linear(features[-1], num_class_labels)
        self._bottleneck = None

        # === Automatically register a hook on the correct (deepest) encoder Conv2d layer ===
        self._register_bottleneck_hook(features[-1])

    def _register_bottleneck_hook(self, bottleneck_channels):
        # Find the first Conv2d with out_channels == features[-1] and register hook
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == bottleneck_channels:
                print(f"[MultiTaskUNet] Registering bottleneck forward hook on: {name}")
                module.register_forward_hook(self.save_bottleneck)
                break
        else:
            raise RuntimeError("Could not find a Conv2d layer with the expected bottleneck channels to register the hook.")

    def save_bottleneck(self, module, input, output):
        self._bottleneck = output

    def forward(self, x):
        self._bottleneck = None  # Clear previous value
        seg_out = self.unet(x)
        if self._bottleneck is None:
            raise RuntimeError("Bottleneck feature was not set by forward hook!")
        pooled = self.gap(self._bottleneck)
        pooled = pooled.view(pooled.size(0), -1)
        class_logits = self.classifier(pooled)
        return class_logits, seg_out
