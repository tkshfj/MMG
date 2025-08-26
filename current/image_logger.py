# image_logger.py
import wandb


def make_image_logger(num_images: int = 4, threshold: float = 0.5):
    from ignite.engine import Events
    from monai.data.meta_tensor import MetaTensor
    import numpy as np
    import torch

    buf_imgs, buf_gt, buf_pred = [], [], []

    def _as_tensor(x):
        return x.as_tensor() if isinstance(x, MetaTensor) else x

    def _squeeze01(x):
        return x[:, 0] if (x is not None and x.ndim == 4 and x.shape[1] == 1) else x

    def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
        if x is None:
            return None
        t = x.detach().cpu()
        if t.ndim == 3:
            c, h, w = t.shape
            arr = t[0] if c == 1 else t[:3].permute(1, 2, 0)
        elif t.ndim == 2:
            arr = t
        else:
            arr = t.view(-1, *t.shape[-2:])[0]
        arr = arr.float()
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn) if mx > mn else arr * 0.0
        return arr.numpy()

    def _to_numpy_mask(x: torch.Tensor) -> np.ndarray:
        if x is None:
            return None
        t = x.detach().cpu()
        if t.ndim == 3:
            t = t[0]
        t = (t > 0.5).to(torch.uint8)
        return t.numpy()

    def _collect_from_output(out):
        if not isinstance(out, dict):
            return None

        # 1) find image
        images = out.get("image", None)

        # 2) find mask (support flat and nested)
        mask = out.get("mask", None)
        if mask is None:
            lbl = out.get("label")
            if isinstance(lbl, dict):
                mask = lbl.get("mask")

        # 3) find segmentation logits (support flat and nested)
        seg_logits = out.get("seg_logits", None)
        if seg_logits is None:
            pred = out.get("pred")
            if isinstance(pred, dict):
                seg_logits = pred.get("seg_out") or pred.get("logits") or pred.get("seg")

        if images is None or mask is None or seg_logits is None:
            # Not a segmentation batch or missing fields
            return None

        # Convert logits -> binary/argmax mask (works for C==1 or C>1)
        logits = _as_tensor(seg_logits)
        if logits.ndim == 4 and logits.shape[1] == 1:
            pmask = (torch.sigmoid(logits) > threshold).float()[:, 0]     # [B,H,W]
        elif logits.ndim == 4 and logits.shape[1] > 1:
            pmask = torch.softmax(logits, dim=1).argmax(dim=1).float()    # [B,H,W]
        else:
            # unsupported shape
            return None

        masks = _as_tensor(mask)
        images = _as_tensor(images)

        # squeeze image/mask first channel if [B,1,H,W]
        if images.ndim == 4 and images.shape[1] == 1:
            images = images[:, 0]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        return images, masks, pmask

    def on_iteration(engine):
        nonlocal buf_imgs, buf_gt, buf_pred
        if len(buf_imgs) >= num_images:
            return
        pack = _collect_from_output(engine.state.output)
        if pack is None:
            return
        images, masks, pmask = pack
        b = min(images.shape[0], num_images - len(buf_imgs))
        for i in range(b):
            buf_imgs.append(images[i])
            buf_gt.append(masks[i])
            buf_pred.append(pmask[i])

    def on_epoch_completed(engine):
        nonlocal buf_imgs, buf_gt, buf_pred
        if not buf_imgs:
            return
        step = int(getattr(engine.state, "trainer_iteration", getattr(engine.state, "iteration", 0)))
        ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
        for i in range(len(buf_imgs)):
            np_img = _to_numpy_image(buf_imgs[i])
            np_gt = _to_numpy_mask(buf_gt[i])
            np_pm = _to_numpy_mask(buf_pred[i])
            if np_img is None or np_gt is None or np_pm is None:
                continue
            wandb.log(
                {"image": wandb.Image(np_img, caption="input"),
                 "gt_mask": wandb.Image(np_gt),
                 "pred_mask": wandb.Image(np_pm),
                 "epoch": ep},
                step=step,
            )
        buf_imgs.clear()
        buf_gt.clear()
        buf_pred.clear()

    def attach(evaluator):
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, on_iteration)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, on_epoch_completed)
        return evaluator

    return attach
