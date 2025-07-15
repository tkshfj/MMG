import torch
import torch.nn as nn
from monai.networks.nets import UNet


class MultiTaskUNet(nn.Module):
    """ U-Net-based architecture for joint segmentation and classification.
    Returns:
        seg_mask: Segmentation mask output (tensor).
        class_logits: Classification logits (tensor).
    """
    def __init__(
        self,
        spatial_dims=2,  # Number of spatial dimensions (2 or 3)
        in_channels=1,  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        out_channels=1,  # Number of output channels for segmentation (e.g., 1 for binary mask, >1 for multiclass)
        num_class_labels=1,  # Number of output classes for classification (e.g., 2 for binary classification)
        features=(32, 64, 128, 256, 512),  # Number of feature maps at each U-Net level
        classifier_dropout=0.0  # Dropout rate before classifier head. Use 0.0 to disable
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        # Dynamically determine U-Net strides based on number of feature levels
        strides = tuple(2 for _ in range(len(features) - 1))
        # MONAI U-Net backbone for segmentation
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            num_res_units=2,
        )
        # Global Average Pooling layer for bottleneck features
        if spatial_dims == 2:
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif spatial_dims == 3:
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError(f"Unsupported spatial_dims: {spatial_dims}. Must be 2 or 3.")
        # Optional dropout before classification head for regularization
        self.classifier_dropout = nn.Dropout(classifier_dropout) if classifier_dropout > 0 else nn.Identity()
        # Linear layer for image-level classification
        self.classifier = nn.Linear(features[-1], num_class_labels)
        # Storage for bottleneck feature, set by forward hook
        self._bottleneck = None
        # Register a forward hook on the deepest encoder ConvNd layer
        self._register_bottleneck_hook(features[-1])

    def _register_bottleneck_hook(self, bottleneck_channels):
        """
        Automatically finds the first Conv2d or Conv3d layer in the U-Net encoder with the expected number of
        channels, and registers a forward hook to save its output as the bottleneck feature.
        """
        conv_type = nn.Conv2d if self.spatial_dims == 2 else nn.Conv3d
        for name, module in self.unet.named_modules():
            if isinstance(module, conv_type) and module.out_channels == bottleneck_channels:
                print(f"[MultiTaskUNet] Registering bottleneck forward hook on: {name}")
                module.register_forward_hook(self.save_bottleneck)
                break
        else:
            # If no such layer is found, raise an informative error
            raise RuntimeError(
                f"Could not find a Conv{self.spatial_dims}d layer with out_channels == {bottleneck_channels} "
                "to register the bottleneck hook. "
                "Check the 'features' argument and U-Net construction."
            )

    def save_bottleneck(self, module, input, output):
        """ Hook function: Saves the bottleneck feature tensor."""
        self._bottleneck = output

    def forward(self, x):
        """
        Forward pass: produces segmentation and classification outputs from input images.
        Args: x (tensor): Input image tensor of shape (B, C, H, W) for 2D, (B, C, D, H, W) for 3D
        Returns:
            seg_out (tensor): Segmentation output mask(s).
            class_logits (tensor): Classification logits.
        """
        self._bottleneck = None  # Clear previous value
        # Run input through U-Net to produce segmentation output
        seg_out = self.unet(x)
        # Ensure the bottleneck feature was captured by the forward hook
        if self._bottleneck is None:
            raise RuntimeError(
                "Bottleneck feature was not set by forward hook! "
                "This usually indicates an architectural mismatch or hook registration failure."
            )
        # Global average pooling of bottleneck feature to get a fixed-size vector per sample
        pooled = self.gap(self._bottleneck)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten
        # Optional dropout, then classification head
        pooled = self.classifier_dropout(pooled)
        class_logits = self.classifier(pooled)
        # Return both outputs: segmentation mask(s), then classification logits
        return seg_out, class_logits


# Example usage for testing/debugging
if __name__ == "__main__":
    # Instantiate for 2D images, binary segmentation, binary classification
    model = MultiTaskUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_labels=2,
        features=(32, 64, 128, 256, 512),
        classifier_dropout=0.2
    )
    img = torch.randn(4, 1, 256, 256)  # Example batch: (batch_size=4, channels=1, height=256, width=256)
    seg, logits = model(img)
    print(seg.shape, logits.shape)  # Expected: torch.Size([4, 1, 256, 256]), torch.Size([4, 2])
