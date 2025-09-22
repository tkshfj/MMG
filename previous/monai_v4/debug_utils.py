# debug_utils.py
def debug_batch(batch, model=None):
    """Print one batch structure/shape once for quick sanity checks."""
    if not isinstance(batch, dict):
        print("[DEBUG] batch is not a dict:", type(batch))
        return

    x = batch.get("image")
    if x is not None:
        shape = getattr(x, "shape", None)
        print("[DEBUG] image:", tuple(shape) if shape is not None else "N/A")

    # common classification keys
    for k in ("label", "classification", "class", "target", "y"):
        if k in batch:
            v = batch[k]
            shape = getattr(v, "shape", None)
            print(f"[DEBUG] {k}:", tuple(shape) if shape is not None else "N/A")
            break

    # segmentation (if wrapper advertises it, skip here to stay generic)
    for k in ("mask", "seg", "mask_label", "mask_true"):
        if k in batch:
            v = batch[k]
            shape = getattr(v, "shape", None)
            print(f"[DEBUG] {k}:", tuple(shape) if shape is not None else "N/A")
            break


def run_sanity_checks(wrapper, model, val_loader, num_classes=2, device="cpu"):
    """Optional: call manually when investigating class priors, etc."""
    import torch
    # print classifier layer (if discoverable)
    find_cls = getattr(wrapper, "_find_classifier_linear", None)
    lin = find_cls(getattr(wrapper, "model", model)) if callable(find_cls) else None
    print("[SANITY] classifier layer:", lin)
    if lin is not None and lin.bias is not None:
        print("[SANITY] classifier bias:", lin.bias.detach().cpu().tolist())

    # label distribution
    binc = torch.zeros(num_classes, dtype=torch.long)
    for b in val_loader:
        y = b["label"]
        y = y.as_tensor() if hasattr(y, "as_tensor") else y
        binc += torch.bincount(y.view(-1).long(), minlength=num_classes)
    print("[SANITY] val label counts:", binc.tolist())

    # zero-input forward → prior-ish probs
    model.eval()
    with torch.no_grad():
        xz = torch.zeros(8,  # batch
                         getattr(wrapper, "in_channels", 1),
                         *getattr(wrapper, "input_shape", (256, 256)),
                         device=device)
        out = model(xz)
        logits = wrapper.extract_logits(out) if hasattr(wrapper, "extract_logits") else out
        if isinstance(logits, (list, tuple)):
            logits = next((t for t in logits if torch.is_tensor(t)), logits[0])
        if isinstance(logits, dict):
            for k in ("class_logits", "logits", "cls", "classification", "label"):
                if k in logits:
                    logits = logits[k]
                    break
        if logits.ndim == 3 and logits.size(-1) == num_classes:
            logits = logits[:, 0, :]
        probs = torch.softmax(logits, dim=1).mean(0).detach().cpu().tolist()
        print("[SANITY] bias-only mean probs:", probs)
