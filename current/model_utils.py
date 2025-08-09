# model_utils.py
from typing import Any, Callable, Dict, List, Tuple, Union
from torch.optim import Adam, SGD, RMSprop, Optimizer
from model_protocol import ModelRegistryProtocol
from metrics_utils import (
    cls_output_transform,
    auc_output_transform,
    seg_output_transform,
    cls_confmat_output_transform,
    seg_confmat_output_transform,
    make_metrics,
)


class BaseModel(ModelRegistryProtocol):
    def get_num_classes(self, config=None):
        # Define a generic way to fetch num_classes for each model
        if hasattr(self, "num_classes"):
            return self.num_classes
        if config is not None and hasattr(config, "num_classes"):
            return config.num_classes
        return 2  # default fallback

    def get_output_transform(self):
        return cls_output_transform

    def get_cls_output_transform(self):
        return cls_output_transform

    def get_cls_confmat_output_transform(self):
        return cls_confmat_output_transform

    def get_auc_output_transform(self):
        return auc_output_transform

    def get_seg_output_transform(self):
        return seg_output_transform

    def get_seg_confmat_output_transform(self):
        return seg_confmat_output_transform

    def get_metrics(self, config=None) -> Dict[str, Any]:
        tasks = self.get_supported_tasks()
        # classification classes
        cls_classes = int(self.get_num_classes(config))
        # segmentation classes (prefer explicit seg_out_channels/out_channels)
        seg_classes = getattr(self, "seg_out_channels", None)
        if seg_classes is None and config is not None and hasattr(config, "out_channels"):
            seg_classes = int(getattr(config, "out_channels"))
        if seg_classes is None:
            seg_classes = cls_classes  # sensible fallback
        return make_metrics(
            tasks=tasks,
            num_classes=cls_classes,  # for classification metrics
            seg_num_classes=seg_classes,  # for segmentation ConfMat
            loss_fn=self.get_loss_fn(),
            cls_ot=self.get_cls_output_transform(),
            auc_ot=self.get_auc_output_transform(),
            seg_cm_ot=self.get_seg_confmat_output_transform(),
            multitask=("multitask" in tasks),
        )


# SimpleCNN
class SimpleCNNModel(BaseModel):
    from models.simple_cnn import SimpleCNN

    def build_model(self, config: Any) -> Any:
        in_channels = getattr(config, "in_channels", 1)
        num_classes = getattr(config, "num_classes", 2)
        return self.SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# DenseNet121
class DenseNet121Model(BaseModel):
    from monai.networks.nets import DenseNet121

    def build_model(self, config: Any) -> Any:
        return self.DenseNet121(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            pretrained=getattr(config, "pretrained", False),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Classification: no segmentation metrics
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }

    def forward(self, x):
        logits = super().forward(x)
        assert logits.ndim == 2, f"DenseNet121: expected [B, num_classes], got {logits.shape}"
        print(f"[DenseNet121] logits shape: {logits.shape}")
        return logits


# UNet
class UNetModel(BaseModel):
    from monai.networks.nets import UNet

    def build_model(self, config: Any) -> Any:
        return self.UNet(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            channels=getattr(config, "features", (32, 64, 128, 256, 512)),
            strides=getattr(config, "strides", (2, 2, 2, 2)),
            num_res_units=getattr(config, "num_res_units", 2),
        )

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }

    def forward(self, x):
        seg_out = super().forward(x)
        assert seg_out.ndim == 4, f"UNet: expected [B, C, H, W], got {seg_out.shape}"
        print(f"[UNet] seg_out shape: {seg_out.shape}")
        return seg_out


# Multitask UNet
class MultitaskUNetModel(BaseModel):
    from models.multitask_unet import MultiTaskUNet

    def build_model(self, config: Any) -> Any:
        # sweep weights (alpha/beta)
        self.cls_loss_weight = float(getattr(config, "cls_loss_weight", 1.0))  # alpha
        self.seg_loss_weight = float(getattr(config, "seg_loss_weight", 1.0))  # beta

        # persist for handlers / loss
        self.in_channels = int(getattr(config, "in_channels", 1))
        self.seg_out_channels = int(getattr(config, "out_channels", 1))   # segmentation branch
        self.num_classes = int(getattr(config, "num_classes", 2))         # classification branch
        self.features = tuple(getattr(config, "features", (32, 64, 128, 256, 512)))

        print(
            ">>> Multitask U-Net DEBUG:",
            {
                "alpha": self.cls_loss_weight,
                "beta": self.seg_loss_weight,
                "in": self.in_channels,
                "seg_out": self.seg_out_channels,
                "num_classes": self.num_classes,
            },
        )

        return self.MultiTaskUNet(
            in_channels=self.in_channels,
            out_channels=self.seg_out_channels,  # seg branch channels
            num_classes=self.num_classes,        # cls branch classes
            features=self.features,
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification", "segmentation", "multitask"]

    def get_loss_fn(self):
        import torch
        import torch.nn as nn
        from monai.data.meta_tensor import MetaTensor

        ce_cls = nn.CrossEntropyLoss()
        bce_seg = nn.BCEWithLogitsLoss()
        ce_seg = nn.CrossEntropyLoss()  # for C>1 segmentation

        # weights set on self (e.g., from sweep)
        alpha = float(getattr(self, "cls_loss_weight", 1.0))
        beta = float(getattr(self, "seg_loss_weight", 1.0))
        strict_cls = bool(getattr(self, "strict_cls_labels", False))

        def _to_tensor(x):
            if isinstance(x, MetaTensor):
                x = x.as_tensor()
            return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

        def _get_nested(d, path):
            cur = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        def _first_present(d, paths):
            for p in paths:
                # v = _get_nested(d, p) if isinstance(p, tuple) else d.get(p, None)
                v = _get_nested(d, p) if isinstance(p, tuple) else (d.get(p, None) if isinstance(d, dict) else None)
                if v is not None:
                    return v
            return None

        # def _dump_once(where, d):
        #     if not getattr(get_loss_fn, f"_dumped_{where}", False):
        #         print(f"[multitask_loss] {where} structure keys: {list(d.keys()) if isinstance(d, dict) else type(d)}")
        #         setattr(get_loss_fn, f"_dumped_{where}", True)

        def _warn_once(tag, msg):
            flag = f"_warned_{tag}"
            if not getattr(self.get_loss_fn, flag, False):
                print(msg)
                setattr(self.get_loss_fn, flag, True)

        def multitask_loss(pred, target):
            # normalize containers
            if isinstance(pred, (tuple, list)) and len(pred) == 2 and isinstance(pred[0], torch.Tensor):
                pred = {"class_logits": pred[0], "seg_out": pred[1]}
            if not isinstance(pred, dict):
                pred = {"class_logits": pred}
            if not isinstance(target, dict):
                target = {"label": target}

            # pull logits/labels (tolerant)
            pl = _first_present(pred, [
                ("class_logits",), ("logits",), ("y_pred",),
                ("pred", "class_logits"), ("pred", "logits"), ("pred", "y_pred"),
                ("cls",), ("output",), ("scores",)
            ])
            yl = _first_present(target, [
                ("label", "label"), ("label", "y"), ("label",),
                ("classification", "label"), ("classification", "y"),
                ("class",), ("cls",), ("y",), ("target",), ("labels",)
            ])

            pm = _first_present(pred, [
                ("seg_out",), ("mask_logits",), ("seg",), ("segmentation",),
                ("pred", "seg_out"), ("pred", "seg"), ("pred", "mask_logits")
            ])
            ym = _first_present(target, [
                ("label", "mask"), ("mask",), ("seg",), ("segmentation",)
            ])

            # classification loss (skip if labels missing)
            cls_loss = None
            if pl is not None and alpha > 0.0:
                if yl is None:
                    if strict_cls:
                        raise KeyError("[multitask_loss] Missing classification label (label/classification/class/y).")
                    _warn_once("missing_cls_label", "[multitask_loss] classification label missing; skipping cls loss for this batch.")
                else:
                    pl = _to_tensor(pl).float()
                    if pl.ndim == 1:
                        pl = pl.unsqueeze(0)
                    yl = _to_tensor(yl).long()
                    if yl.ndim >= 2 and yl.shape[-1] > 1:
                        yl = yl.argmax(dim=-1)
                    if yl.ndim >= 2 and yl.shape[-1] == 1:
                        yl = yl.squeeze(-1)
                    yl = yl.view(-1)
                    cls_loss = ce_cls(pl, yl)

            # segmentation loss (unchanged, tolerant to missing)
            seg_loss = None
            if pm is not None and beta > 0.0:
                pm = _to_tensor(pm).float()
                if ym is not None:
                    ym = _to_tensor(ym)
                    if ym.ndim == 4:
                        ym = ym[:, 0] if ym.shape[1] == 1 else ym.argmax(dim=1)
                    elif ym.ndim == 3 and ym.shape[0] == 1:
                        ym = ym.squeeze(0)
                    if ym.ndim == 2:
                        ym = ym.unsqueeze(0)
                    ym = ym.long()
                    if pm.ndim == 3:
                        pm = pm.unsqueeze(0)
                    seg_loss = bce_seg(pm, ym.float().unsqueeze(1)) if pm.shape[1] == 1 else ce_seg(pm, ym)
                else:
                    _warn_once("missing_seg_label", "[multitask_loss] segmentation label missing; skipping seg loss for this batch.")

            # combine
            total = None
            if cls_loss is not None:
                total = alpha * cls_loss
            if seg_loss is not None:
                total = seg_loss * beta if total is None else total + beta * seg_loss

            if total is None:
                # if both absent, keep trainer alive with a zero tensor on a sane device
                device = (pl.device if isinstance(pl, torch.Tensor)
                          else pm.device if isinstance(pm, torch.Tensor)
                          else torch.device("cpu"))
                return torch.tensor(0.0, device=device)

            return total

        return multitask_loss

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_classes", 2),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# ViT
class ViTModel(BaseModel):
    from monai.networks.nets import ViT

    def _normalize_img_size(
        self,
        input_shape: Union[Tuple[int, ...], List[int]],
        in_channels: int,
        spatial_dims: int,
    ) -> Tuple[int, ...]:
        """Return (H, W) for 2D or (D, H, W) for 3D, stripping a leading channel dim if present."""
        shape = tuple(input_shape)
        if spatial_dims == 2:
            # Accept (H, W) or (C, H, W)
            if len(shape) == 3 and shape[0] == in_channels:
                return shape[1:3]
            if len(shape) == 2:
                return shape
        elif spatial_dims == 3:
            # Accept (D, H, W) or (C, D, H, W)
            if len(shape) == 4 and shape[0] == in_channels:
                return shape[1:4]
            if len(shape) == 3:
                return shape
        raise ValueError(f"input_shape {shape} is not compatible with spatial_dims={spatial_dims} and in_channels={in_channels}")

    def _validate_patching(self, img_size: Tuple[int, ...], patch_size: Union[int, Tuple[int, ...]]):
        if isinstance(patch_size, int):
            ps = (patch_size,) * len(img_size)
        else:
            ps = tuple(patch_size)
        if len(ps) != len(img_size):
            raise ValueError(f"patch_size {ps} must match img_size dims {img_size}")
        bad = [s for s, p in zip(img_size, ps) if s % p != 0]
        if bad:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")

    def build_model(self, config: Any) -> Any:
        spatial_dims = int(getattr(config, "spatial_dims", 2))
        in_channels = int(getattr(config, "in_channels", 1))
        num_classes = int(getattr(config, "num_classes", 2))
        patch_size = getattr(config, "patch_size", 16)

        # Normalize img_size (strip channel dim if provided)
        input_shape = getattr(config, "input_shape", (256, 256))
        img_size = self._normalize_img_size(input_shape, in_channels, spatial_dims)

        # Sanity check patching
        self._validate_patching(img_size, patch_size)

        # Persist for handlers/metrics
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size

        print(">>> ViT DEBUG:", {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "spatial_dims": spatial_dims,
        })

        return self.ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=num_classes,
            hidden_size=int(getattr(config, "hidden_size", 768)),
            mlp_dim=int(getattr(config, "mlp_dim", 3072)),
            num_layers=int(getattr(config, "num_layers", 12)),
            num_heads=int(getattr(config, "num_heads", 12)),
            dropout_rate=float(getattr(config, "dropout_rate", 0.1)),
            # Optional extras if you want to expose them via config:
            # pos_embed=getattr(config, "pos_embed", "conv"),
            # qkv_bias=bool(getattr(config, "qkv_bias", False)),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    @staticmethod
    def extract_logits(y_pred):
        """Extracts logits from ViT output (handles tuple or plain tensor)."""
        if isinstance(y_pred, tuple):
            return y_pred[0]
        return y_pred

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn

        def loss_fn(y_pred, y_true):
            y_pred = self.extract_logits(y_pred)
            # For classification, y_true might be a dict: {"label": ...}
            if isinstance(y_true, dict):
                # Try all possible keys in order of likelihood
                for key in ["label", "classification", "class"]:
                    if key in y_true:
                        y_true = y_true[key]
                        break
                # If not found, raise an error
                else:
                    raise ValueError(f"Expected key 'label' or 'classification' in y_true dict, got keys: {list(y_true.keys())}")
            return nn.CrossEntropyLoss()(y_pred, y_true)
        return loss_fn

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            # classification metrics on
            "add_classification_metrics": True,
            "cls_output_transform": self.get_cls_output_transform(),
            "num_classes": getattr(self, "num_classes", 2),
            "acc_name": "val_acc",
            "auc_name": "val_auc",
            "confmat_name": "val_cls_confmat",
            # segmentation metrics off
            "add_segmentation_metrics": False,
            # "seg_output_transform": None,
            # "dice_name": "val_dice",
            # "iou_name": "val_iou",
        }


# SwinUNETR
class SwinUNETRModel(BaseModel):
    from monai.networks.nets import SwinUNETR

    def build_model(self, config: Any) -> Any:
        return self.SwinUNETR(
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            img_size=getattr(config, "img_size", (256, 256)),
            feature_size=getattr(config, "feature_size", 48),
            use_checkpoint=getattr(config, "use_checkpoint", False),
            norm_layer=getattr(config, "norm_layer", "instance"),
            depths=getattr(config, "depths", (2, 2, 6, 2)),
            num_heads=getattr(config, "num_heads", (3, 6, 12, 24)),
            window_size=getattr(config, "window_size", 7),
            mlp_ratio=getattr(config, "mlp_ratio", 4.0),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Segmentation: segmentation metrics should be enabled
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "out_channels", 1),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }

    def forward(self, x):
        seg_out = super().forward(x)
        assert seg_out.ndim == 4, f"SwinUNETR: expected [B, C, H, W], got {seg_out.shape}"
        print(f"[SwinUNETR] seg_out shape: {seg_out.shape}")
        return seg_out


# Model registry
MODEL_REGISTRY: Dict[str, ModelRegistryProtocol] = {
    "simple_cnn": SimpleCNNModel(),
    "densenet121": DenseNet121Model(),
    "unet": UNetModel(),
    "multitask_unet": MultitaskUNetModel(),
    "vit": ViTModel(),
    "swin_unetr": SwinUNETRModel(),
}


# Optimizer construction
def get_optimizer(
    name: str,
    parameters: Any,
    lr: float,
    weight_decay: float = 1e-4
) -> Optimizer:
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif name == "rmsprop":
        return RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


__all__ = [
    "SimpleCNNModel",
    "DenseNet121Model",
    "UNetModel",
    "MultitaskUNetModel",
    "ViTModel",
    "SwinUNETRModel",
]
