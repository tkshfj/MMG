import torch.nn as nn
from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    """
    U-Net-based architecture for joint segmentation and classification.
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of segmentation mask classes.
        num_classes_cls (int): Number of output classes for classification.
        features (list): Number of feature maps at each level.
    Returns:
        seg_mask: Segmentation mask output.
        cls_logits: Classification logits.
    """
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=1, num_class_labels=1, features=(32, 64, 128, 256, 512)):
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

        # Automatically register a hook on the correct (deepest) encoder Conv2d layer
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


# class MultiTaskUNet(nn.Module):
#     def __init__(self, spatial_dims=2, in_channels=1, out_channels=1, num_class_labels=1, features=(32, 64, 128, 256, 512)):
#         super().__init__()
#         self.unet = UNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             channels=features,
#             strides=(2, 2, 2, 2),
#             num_res_units=2,
#         )
#         print("=== UNet children ===")
#         for i, m in enumerate(self.unet.children()):
#             print(f"self.unet[{i}]: {m}")
#         if hasattr(self.unet, 'model'):
#             print("=== self.unet.model children ===")
#             for i, m in enumerate(self.unet.model.children()):
#                 print(f"self.unet.model[{i}]: {m}")
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(features[-1], num_class_labels)
#         self._bottleneck = None

#         # Register a forward hook on the bottleneck (last encoder block)
#         # Find the correct layer: usually self.unet.model[4] for 5-level UNet
#         # Print self.unet.model to check your version!
#         self.unet.model[4].register_forward_hook(self.save_bottleneck)

#     def save_bottleneck(self, module, input, output):
#         self._bottleneck = output

#     def forward(self, x):
#         # Reset bottleneck
#         self._bottleneck = None
#         seg_out = self.unet(x)
#         # Now self._bottleneck contains the output of the encoder's last block
#         if self._bottleneck is None:
#             raise RuntimeError("Bottleneck feature was not set by forward hook!")
#         pooled = self.gap(self._bottleneck)
#         pooled = pooled.view(pooled.size(0), -1)
#         class_logits = self.classifier(pooled)
#         return class_logits, seg_out


# # import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# from monai.networks.nets import UNet


# class MultiTaskUNet(nn.Module):
#     def __init__(self,
#                  spatial_dims=2,
#                  in_channels=1,
#                  out_channels=1,
#                  num_class_labels=2,   # 2 for binary, >2 for multiclass
#                  features=(32, 64, 128, 256, 512)):
#         super().__init__()
#         self.unet = UNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             channels=features,
#             strides=(2, 2, 2, 2),
#             num_res_units=2,
#         )
#         # Classification head: take encoder bottleneck features and pool
#         # For MONAI UNet, we use last encoder feature map before upsampling
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(features[-1], num_class_labels)

#     def forward(self, x):
#         # UNet's forward returns segmentation output, but we want bottleneck too
#         # We'll use the UNet's forward features if available, or hack via hooks,
#         # but simplest: pass x through encoder, then use decoder
#         # Instead, let's pass x through self.unet and also tap bottleneck

#         # To get encoder output, MONAI UNet doesn't expose directly.
#         # So we use the segmentation output for demo, but in production,
#         # subclass UNet or use encoder separately for best results.
#         seg_out = self.unet(x)  # [B, out_channels, H, W]
#         # Use pooled segmentation output for classification (demo version)
#         pooled = self.gap(seg_out)
#         pooled = pooled.view(pooled.size(0), -1)
#         class_logits = self.classifier(pooled)
#         return class_logits, seg_out
