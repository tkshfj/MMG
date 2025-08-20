# main.py
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # cuBLAS deterministic
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)

import time
from typing import Tuple
from collections.abc import Callable

import torch
import wandb

from monai.utils import set_determinism

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import MODEL_REGISTRY
from engine_utils import build_trainer, build_evaluator, make_prepare_batch, attach_scheduler
from handlers import register_handlers, wandb_log_handler, CHECKPOINT_DIR
from optim_factory import get_optimizer
from resume_utils import restore_training_state


# determinism (CPU side + algorithm choices), safe pre-fork
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_determinism(seed=seed)


def enforce_fp32_policy():
    # Call this AFTER DataLoaders are built (to avoid early CUDA init)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalize_arch(name: str | None) -> str:
    return str(name or "model").lower().replace("/", "_").replace(" ", "_")


def get_task_flags(task: str | None) -> Tuple[bool, bool]:
    t = (task or "multitask").lower()
    return t != "segmentation", t != "classification"


def dict_safe(loss: torch.nn.Module | Callable):
    if not isinstance(loss, torch.nn.Module):
        return loss
    bce = torch.nn.BCEWithLogitsLoss

    def _fn(pred, tgt):
        pred_t = pred.get("cls_out", pred.get("class_logits", pred)) if isinstance(pred, dict) else pred
        tgt_t  = tgt.get("label", tgt.get("y", tgt)) if isinstance(tgt, dict) else tgt
        if isinstance(loss, bce):
            return loss(pred_t, torch.as_tensor(tgt_t).float().view_as(pred_t))
        return loss(pred_t, torch.as_tensor(tgt_t).long().view(-1))
    return _fn


def configure_wandb_step_semantics() -> None:
    try:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")
        wandb.define_metric("val_*",   step_metric="epoch")
    except Exception:
        pass


def run(
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> None:
    # model
    arch = normalize_arch(cfg.get("architecture"))
    try:
        wrapper = MODEL_REGISTRY[arch]()
    except KeyError as e:
        raise ValueError(f"Unknown architecture '{arch}'.") from e

    model = wrapper.build_model(cfg).to(device=device, dtype=torch.float32)

    # optim
    optimizer = get_optimizer(cfg, model)

    # task + metrics
    task = (cfg.get("task") or wrapper.get_supported_tasks()[0]).lower()
    has_cls, has_seg = get_task_flags(task)
    metrics = wrapper.get_metrics(config=cfg, task=task, has_cls=has_cls, has_seg=has_seg)

    # loss (dict-safe)
    loss_fn = dict_safe(wrapper.get_loss_fn(task, cfg))

    #- batch prep-
    prepare_batch = make_prepare_batch(
        task=task,
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
    )

    # engines
    trainer = build_trainer(
        device=device,
        max_epochs=cfg["epochs"],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        prepare_batch=prepare_batch,
    )
    evaluator = build_evaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch,
        metrics=metrics,
        decollate=False,
        postprocessing=None,
        inferer=None,
    )

    #- scheduler-
    scheduler = attach_scheduler(
        cfg=cfg,
        trainer=trainer,
        evaluator=evaluator,
        optimizer=optimizer,
        train_loader=train_loader,
        use_monai_handler=False,
    )

    # optional one-shot debug
    if cfg.get("debug_batch_once", bool(cfg.get("debug", False))):
        from debug_utils import debug_batch
        debug_batch(next(iter(train_loader)), model)

    #- handlers, checkpoints, resume-
    handler_kwargs = {
        **wrapper.get_handler_kwargs(),
        "add_classification_metrics": has_cls,
        "add_segmentation_metrics": has_seg,
    }

    run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)

    last_epoch = restore_training_state(
        dirname=ckpt_dir,
        prefix=arch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        map_location=device.type if device.type != "cuda" else "cuda",
    )
    if last_epoch is not None:
        print(f"[resume] restored training state from epoch {last_epoch}")

    register_handlers(
        cfg=cfg,
        trainer=trainer,
        evaluator=evaluator,
        model=model,
        save_dir=ckpt_dir,
        prefix=arch,
        wandb_log_handler=wandb_log_handler,
        metric_names=list(metrics.keys()),
        **handler_kwargs,
    )

    print("[DEBUG] max_epochs (pre-run) =", trainer.state.max_epochs)
    trainer.run()

    if cfg.get("save_final_state", True):
        os.makedirs("outputs/best_model", exist_ok=True)
        save_path = f"outputs/best_model/{arch}_{wandb.run.id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Training complete. Saved: {save_path}")


if __name__ == "__main__":
    import sys
    import multiprocessing as mp

    method = "fork" if sys.platform.startswith("linux") else "spawn"
    try:
        mp.set_start_method(method, force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context(method)

    with wandb.init(config={}, dir="outputs") as wb_run:
        cfg = load_and_validate_config(dict(wandb.config))
        cfg.setdefault("run_id",   wb_run.id)
        cfg.setdefault("run_name", wb_run.name)
        configure_wandb_step_semantics()

        # Build DataLoaders FIRST (CPU path), pass context & pin_memory as desired
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv=cfg["metadata_csv"],
            input_shape=cfg.get("input_shape", (256, 256)),
            batch_size=cfg.get("batch_size", 16),
            task=cfg.get("task", "multitask"),
            split=cfg.get("split", (0.7, 0.15, 0.15)),
            num_workers=cfg.get("num_workers", 32),
            debug=bool(cfg.get("debug", False)),
            pin_memory=bool(cfg.get("pin_memory", False)),
            multiprocessing_context=ctx,
        )

        if cfg.get("print_dataset_sizes", True):
            print(f"[INFO] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        # Only now touch CUDA/TF32 settings and select device
        enforce_fp32_policy()
        device = auto_device()
        cfg["device"] = device

        # Run training with explicit loaders & device
        run(cfg, train_loader, val_loader, test_loader, device)


# # main.py
# import os
# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # CuBLAS deterministic
# os.environ.setdefault("PYTHONHASHSEED", "42")  # stable hashing for dicts/sets
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import random
# import numpy as np
# seed = 42
# random.seed(seed)
# np.random.seed(seed)

# import sys
# import time
# import torch
# import wandb
# import multiprocessing as mp
# from typing import Tuple
# from collections.abc import Callable
# from monai.utils import set_determinism

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# set_determinism(seed=seed)

# from config_utils import load_and_validate_config
# from data_utils import build_dataloaders
# from model_registry import MODEL_REGISTRY
# from engine_utils import build_trainer, build_evaluator, make_prepare_batch, attach_scheduler
# from handlers import register_handlers, wandb_log_handler, CHECKPOINT_DIR
# from optim_factory import get_optimizer
# from resume_utils import restore_training_state

# # import random
# # import numpy as np
# # seed = 42  # seeding
# # random.seed(seed)
# # np.random.seed(seed)

# # import torch
# # from monai.utils import set_determinism
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# # set_determinism(seed=seed)

# # import time
# # import wandb
# # from typing import Tuple
# # from collections.abc import Callable

# # from config_utils import load_and_validate_config
# # from data_utils import build_dataloaders
# # from model_registry import MODEL_REGISTRY
# # from engine_utils import build_trainer, build_evaluator, make_prepare_batch, attach_scheduler
# # from handlers import register_handlers, wandb_log_handler, CHECKPOINT_DIR
# # from optim_factory import get_optimizer
# # from resume_utils import restore_training_state


# # FP32-only policy: disable TF32 on CUDA and avoid AMP
# def enforce_fp32_policy():
#     if torch.cuda.is_available():
#         torch.backends.cuda.matmul.allow_tf32 = False
#         torch.backends.cudnn.allow_tf32 = False
#     try:
#         torch.set_float32_matmul_precision('highest')
#     except Exception:
#         pass


# def auto_device() -> torch.device:
#     """Select CUDA, then MPS (macOS), else CPU."""
#     return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # if torch.cuda.is_available():
#     #     return torch.device("cuda")
#     # # if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
#     # #     return torch.device("mps")
#     # return torch.device("cpu")


# def normalize_arch(name: str | None) -> str:
#     """Normalize architecture name to a canonical registry key."""
#     return str(name or "model").lower().replace("/", "_").replace(" ", "_")


# def get_task_flags(task: str | None) -> Tuple[bool, bool]:
#     """Return (has_cls, has_seg) from task label."""
#     t = (task or "multitask").lower()
#     return t != "segmentation", t != "classification"


# def dict_safe(loss: torch.nn.Module | Callable):
#     """Wrap a torch loss to accept dict(pred/target) or raw tensors."""
#     if not isinstance(loss, torch.nn.Module):
#         return loss
#     bce = torch.nn.BCEWithLogitsLoss

#     def _fn(pred, tgt):
#         # logits
#         pred_t = pred.get("cls_out", pred.get("class_logits", pred)) if isinstance(pred, dict) else pred
#         # target
#         tgt_t = tgt.get("label", tgt.get("y", tgt)) if isinstance(tgt, dict) else tgt
#         if isinstance(loss, bce):
#             return loss(pred_t, torch.as_tensor(tgt_t).float().view_as(pred_t))
#         return loss(pred_t, torch.as_tensor(tgt_t).long().view(-1))
#     return _fn


# def configure_wandb_step_semantics() -> None:
#     """Unify step semantics: both train/* and val/* step on global_step."""
#     try:
#         wandb.define_metric("epoch")
#         wandb.define_metric("train/*", step_metric="epoch")
#         wandb.define_metric("val/*", step_metric="epoch")
#         wandb.define_metric("val_*", step_metric="epoch")
#     except Exception:
#         pass


# def run(cfg: dict) -> None:
#     # # Multiprocessing context: use explicit fork on Linux
#     # mp_ctx = None
#     # if sys.platform.startswith("linux"):
#     #     mp_ctx = mp.get_context("fork")  # workers will fork safely (CPU-only)

#     # # Build DataLoaders
#     # train_loader, val_loader, test_loader = build_dataloaders(
#     #     metadata_csv=cfg["metadata_csv"],
#     #     input_shape=cfg.get("input_shape", (256, 256)),
#     #     batch_size=cfg.get("batch_size", 16),
#     #     task=cfg.get("task", "multitask"),
#     #     split=cfg.get("split", (0.7, 0.15, 0.15)),
#     #     num_workers=cfg.get("num_workers", 32),
#     #     debug=bool(cfg.get("debug", False)),
#     #     pin_memory=bool(cfg.get("pin_memory", False)),
#     #     multiprocessing_context=mp_ctx,
#     # )

#     # #  Smoke test (fast fail on shapes/dtypes)
#     # if cfg.get("debug", False):
#     #     from data_utils import _sanity_check_batch
#     #     batch = next(iter(val_loader))
#     #     _sanity_check_batch(batch, cfg.get("task", "multitask"))

#     # if cfg.get("print_dataset_sizes", True):
#     #     print(f"[INFO] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

#     # # setup
#     # enforce_fp32_policy()
#     # device = auto_device()
#     # cfg["device"] = device

#     # # data
#     # train_loader, val_loader, test_loader = build_dataloaders(
#     #     metadata_csv=cfg["metadata_csv"],
#     #     input_shape=cfg.get("input_shape", (256, 256)),
#     #     batch_size=cfg.get("batch_size", 16),
#     #     task=cfg.get("task", "multitask"),
#     #     split=cfg.get("split", (0.7, 0.15, 0.15)),
#     #     num_workers=cfg.get("num_workers", 32),
#     #     debug=bool(cfg.get("debug", False)),
#     #     pin_memory=bool(cfg.get("pin_memory", False)),
#     # )
#     # if cfg.get("print_dataset_sizes", True):
#     #     print(
#     #         f"[INFO] Train: {len(train_loader.dataset)} | "
#     #         f"Val: {len(val_loader.dataset)} | "
#     #         f"Test: {len(test_loader.dataset)}"
#     #     )

#     # Model
#     arch = normalize_arch(cfg.get("architecture"))
#     try:
#         wrapper = MODEL_REGISTRY[arch]()  # model wrapper
#     except KeyError as e:
#         raise ValueError(f"Unknown architecture '{arch}'.") from e
#     model = wrapper.build_model(cfg).to(device=device, dtype=torch.float32)

#     # Optimizer
#     optimizer = get_optimizer(cfg, model)

#     # Task + metrics
#     task = (cfg.get("task") or wrapper.get_supported_tasks()[0]).lower()
#     has_cls, has_seg = get_task_flags(task)
#     metrics = wrapper.get_metrics(config=cfg, task=task, has_cls=has_cls, has_seg=has_seg)

#     # Loss (dict-safe)
#     loss_fn = dict_safe(wrapper.get_loss_fn(task, cfg))

#     # Batch prep
#     prepare_batch = make_prepare_batch(
#         task=task,
#         debug=cfg.get("debug"),
#         binary_label_check=True,
#         num_classes=int(cfg.get("num_classes", 2)),
#     )

#     # Engines
#     trainer = build_trainer(
#         device=device,
#         max_epochs=cfg["epochs"],
#         train_data_loader=train_loader,
#         network=model,
#         optimizer=optimizer,
#         loss_function=loss_fn,
#         prepare_batch=prepare_batch,
#     )
#     evaluator = build_evaluator(
#         device=device,
#         val_data_loader=val_loader,
#         network=model,
#         prepare_batch=prepare_batch,
#         metrics=metrics,
#         decollate=False,
#         postprocessing=None,
#         inferer=None,
#     )

#     # Scheduler
#     scheduler = attach_scheduler(
#         cfg=cfg,
#         trainer=trainer,
#         evaluator=evaluator,
#         optimizer=optimizer,
#         train_loader=train_loader,
#         use_monai_handler=False,
#     )

#     # Optional one-shot debug
#     if cfg.get("debug_batch_once", bool(cfg.get("debug", False))):
#         from debug_utils import debug_batch
#         debug_batch(next(iter(train_loader)), model)

#     # Handlers
#     handler_kwargs = {
#         **wrapper.get_handler_kwargs(),
#         "add_classification_metrics": has_cls,
#         "add_segmentation_metrics": has_seg,
#     }

#     # Checkpoints/resume
#     run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
#     ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)

#     last_epoch = restore_training_state(
#         dirname=ckpt_dir,
#         prefix=arch,
#         model=model,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         trainer=trainer,
#         map_location=device.type if device.type != "cuda" else "cuda",
#     )
#     if last_epoch is not None:
#         print(f"[resume] restored training state from epoch {last_epoch}")

#     register_handlers(
#         cfg=cfg,
#         trainer=trainer,
#         evaluator=evaluator,
#         model=model,
#         save_dir=ckpt_dir,
#         prefix=arch,
#         wandb_log_handler=wandb_log_handler,
#         metric_names=list(metrics.keys()),
#         **handler_kwargs,
#     )

#     print("[DEBUG] max_epochs (pre-run) =", trainer.state.max_epochs)
#     trainer.run()

#     # Final export
#     if cfg.get("save_final_state", True):
#         os.makedirs("outputs/best_model", exist_ok=True)
#         save_path = f"outputs/best_model/{arch}_{wandb.run.id}.pth"
#         torch.save(model.state_dict(), save_path)
#         print(f"[INFO] Training complete. Saved: {save_path}")


# def main(config: dict | None = None) -> None:
#     with wandb.init(config=config or {}, dir="outputs") as wb_run:
#         cfg = load_and_validate_config(dict(wandb.config))
#         cfg.setdefault("run_id", wb_run.id)
#         cfg.setdefault("run_name", wb_run.name)
#         configure_wandb_step_semantics()
#         run(cfg)


# if __name__ == "__main__":
#     import multiprocessing as mp

#     # Make Linux act like the plan: explicitly use 'fork' in DataLoader workers.
#     try:
#         mp.set_start_method("fork", force=True)   # optional: set default; safe on Linux
#     except RuntimeError:
#         pass

#     ctx = mp.get_context("fork")                  # the context we will pass to DataLoader

#     with wandb.init(config={}, dir="outputs") as wb_run:
#         cfg = load_and_validate_config(dict(wandb.config))
#         cfg.setdefault("run_id", wb_run.id)
#         cfg.setdefault("run_name", wb_run.name)
#         configure_wandb_step_semantics()

#         # build loaders FIRST (CPU path), pass context & pin_memory as desired
#         train_loader, val_loader, test_loader = build_dataloaders(
#             metadata_csv=cfg["metadata_csv"],
#             input_shape=cfg.get("input_shape", (256, 256)),
#             batch_size=cfg.get("batch_size", 16),
#             task=cfg.get("task", "multitask"),
#             split=cfg.get("split", (0.7, 0.15, 0.15)),
#             num_workers=cfg.get("num_workers", 32),
#             debug=bool(cfg.get("debug", False)),
#             pin_memory=bool(cfg.get("pin_memory", False)),
#             multiprocessing_context=ctx,          # <-- pass it here
#         )

#         # Now safe to touch CUDA settings
#         enforce_fp32_policy()
#         device = auto_device()
#         cfg["device"] = device

#         # (remainder of run(cfg) flow can be inlined or left as-is)
#         run(cfg)

#     # Choose the start method once at process start:
#     # try:
#     #     if sys.platform.startswith("linux"):
#     #         mp.set_start_method("fork", force=True)   # Linux: fork
#     #     else:
#     #         mp.set_start_method("spawn", force=True)  # macOS/Windows: spawn
#     # except RuntimeError:
#     #     pass  # already set by a parent process

#     # main()
