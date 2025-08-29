# image_logger.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def make_image_logger(max_items: int = 8, threshold: float = 0.5, namespace: str = "val"):
    """
    Returns a callable:
        log_fn(payload_like, *, wandb_module=None, epoch_anchor=None)

    - payload_like: either an evaluator output dict (preferred: contains predictions)
                    or a batch dict (contains input/gt only).
                    Expected keys (if available): 'image', 'mask', 'seg_logits' / 'pred'
    - wandb_module: typically the `wandb` module instance returned by your init helper
    - epoch_anchor: int epoch number; will log with {"trainer/epoch": epoch_anchor}

    The function logs once per call with:
        { "trainer/epoch": ep,
          f"{namespace}/images":      [wandb.Image, ...],
          f"{namespace}/gt_masks":    [wandb.Image, ...],   (if GT present)
          f"{namespace}/pred_masks":  [wandb.Image, ...] }  (if preds present)
    """
    def _to_cpu_np(x: torch.Tensor) -> np.ndarray:
        t = x.detach().cpu()
        return t.numpy()

    def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
        # Accept [C,H,W] (C==1 or 3), [H,W], or [B,C,H,W]—caller slices per-item
        if t.ndim == 3:  # [C,H,W] or [H,W]
            if t.shape[0] in (1, 3):  # channel-first
                c, h, w = t.shape
                arr = t[0] if c == 1 else t[:3].permute(1, 2, 0)
            else:
                arr = t  # treat as [H,W,C]? rare for tensors; handled as [H,W,?]
        elif t.ndim == 2:
            arr = t
        else:
            # fallback to first item
            arr = t.view(-1, *t.shape[-2:])[0]
        arr = arr.float()
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = arr * 0.0
        return _to_cpu_np(arr)

    def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
        # Accept [H,W] or [C,H,W] (C==1) or [B,H,W]—caller slices per-item
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        t = (t > 0.5).to(torch.uint8)
        return _to_cpu_np(t)

    def _as_tensor(x):
        # Accept torch.Tensor or array-like
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _extract_triplet(obj: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (images, gt_masks, pred_masks) shaped as:
          images:    [B,1,H,W] or [B,3,H,W] or [B,H,W]
          gt_masks:  [B,H,W] or [B,1,H,W] or None
          pred_masks:[B,H,W] or None
        """
        images = obj.get("image")
        gt = obj.get("mask")

        # predicted logits / masks may live in various keys
        pred = obj.get("seg_logits")
        if pred is None:
            pr = obj.get("pred")
            if isinstance(pr, dict):
                # common nested candidates
                pred = pr.get("seg_out") or pr.get("logits") or pr.get("seg")
            elif isinstance(pr, torch.Tensor):
                pred = pr

        images = _as_tensor(images) if images is not None else None
        gt = _as_tensor(gt) if gt is not None else None
        pred = _as_tensor(pred) if pred is not None else None

        # convert logits -> mask
        if pred is not None:
            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = (torch.sigmoid(pred) > threshold).float().squeeze(1)  # [B,H,W]
            elif pred.ndim == 4 and pred.shape[1] > 1:
                pred = torch.softmax(pred, dim=1).argmax(dim=1).float()      # [B,H,W]
            elif pred.ndim == 3:
                # already a mask [B,H,W]
                pred = (pred > 0.5).float()
            else:
                pred = None  # unsupported shape

        # squeeze to canonical shapes
        if images is not None and images.ndim == 4 and images.shape[1] == 1:
            images = images  # keep [B,1,H,W]
        elif images is not None and images.ndim == 3:
            images = images.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]

        if gt is not None:
            if gt.ndim == 4 and gt.shape[1] == 1:
                gt = gt.squeeze(1)  # [B,1,H,W] -> [B,H,W]
            elif gt.ndim == 3:
                gt = gt  # [B,H,W]
            else:
                gt = None

        if pred is not None and pred.ndim != 3:
            pred = None

        return images, gt, pred

    def _build_lists(images: torch.Tensor,
                     gt: Optional[torch.Tensor],
                     pred: Optional[torch.Tensor],
                     limit: int) -> Tuple[List, List, List]:
        """Return lists of np arrays ready for wandb.Image."""
        B = int(images.shape[0])
        n = min(B, max(1, int(limit)))
        imgs_np: List[np.ndarray] = []
        gts_np: List[np.ndarray] = []
        preds_np: List[np.ndarray] = []

        for i in range(n):
            img_t = images[i]  # [1,H,W]
            np_img = _to_numpy_image(img_t)
            imgs_np.append(np_img)

            if gt is not None:
                gts_np.append(_to_numpy_mask(gt[i]))
            if pred is not None:
                preds_np.append(_to_numpy_mask(pred[i]))

        return imgs_np, gts_np, preds_np

    def _log(payload_like: Dict[str, Any],
             *,
             wandb_module=None,
             epoch_anchor: Optional[int] = None):
        """Main logging callable."""
        if wandb_module is None or getattr(wandb_module, "run", None) is None:
            return
        ep = int(epoch_anchor if epoch_anchor is not None else 0)

        # prefer evaluator output dict; fall back to batch dict
        images, gt, pred = _extract_triplet(payload_like)
        if images is None:
            return

        imgs_np, gts_np, preds_np = _build_lists(images, gt, pred, max_items)

        # Build one payload per epoch (lists so earlier items aren’t overwritten)
        payload = {"trainer/epoch": ep}
        # Store as lists of Images (recommended by W&B for multiple examples at the same step)
        payload[f"{namespace}/images"] = [wandb_module.Image(x, caption=f"{namespace} #{i}")
                                          for i, x in enumerate(imgs_np)]
        if gts_np:
            payload[f"{namespace}/gt_masks"] = [wandb_module.Image(x) for x in gts_np]
        if preds_np:
            payload[f"{namespace}/pred_masks"] = [wandb_module.Image(x) for x in preds_np]

        wandb_module.log(payload)  # <-- no step=

    return _log
