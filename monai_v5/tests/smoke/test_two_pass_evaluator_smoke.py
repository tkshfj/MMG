# Smoke test for the two-pass evaluator
# how to run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" pytest -q tests/smoke/test_two_pass_evaluator_smoke.py

import torch
from torch.utils.data import Dataset, DataLoader


def test_two_pass_evaluator_smoke():
    # Local imports to avoid polluting main code paths when tests aren't running
    from evaluator_two_pass import make_two_pass_evaluator

    # Tiny toy dataset
    class ToyDS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            x = torch.randn(1, 32, 32)
            y = torch.randint(0, 2, (1,)).long()
            m = torch.randint(0, 2, (32, 32)).long()
            return {"image": x, "label": y, "mask": m}

    # Tiny model that returns both heads in the shapes our extractors expect
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            B = x.size(0)
            return {
                "cls_out": torch.randn(B, 2),          # logits [B,2]
                "seg_out": torch.randn(B, 1, 32, 32),  # logits [B,1,H,W] (binary seg)
            }

    model = ToyModel()
    loader = DataLoader(ToyDS(), batch_size=2, shuffle=False)

    # Build evaluator with the correct OTs wired inside
    ev = make_two_pass_evaluator(
        calibrator=None,
        task="multitask",
        positive_index=1,
        num_classes=2,
        cls_decision="threshold",
        cls_threshold=0.5,
    )

    # One pass should produce sane metrics without crashing
    t, cls_m, seg_m = ev.validate(epoch=1, model=model, val_loader=loader, base_rate=None)

    # Minimal assertions that verify both raw-logits and decision paths are hit
    assert isinstance(t, float)
    for k in ("auc", "acc", "prec", "recall", "pos_rate", "gt_pos_rate", "cls_confmat"):
        assert k in cls_m
    for k in ("dice", "iou", "seg_confmat"):
        assert k in seg_m
