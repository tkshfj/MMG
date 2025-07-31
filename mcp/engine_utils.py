# engine_utils.py
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from typing import Any, Callable, Dict, Optional


def build_trainer(
    device: Any,
    max_epochs: int,
    train_data_loader: Any,
    network: Any,
    optimizer: Any,
    loss_function: Callable,
    prepare_batch: Callable,
    output_transform: Optional[Callable] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> SupervisedTrainer:
    """
    Build a MONAI SupervisedTrainer using MCP-supplied components.
    All arguments are protocol-driven, so this trainer is fully model/task agnostic.
    - If metrics is supplied, will attach at construction (optional; often done externally).
    - If output_transform is not supplied, uses default.
    """
    assert callable(loss_function), "loss_function must be callable"
    assert callable(prepare_batch), "prepare_batch must be callable"
    return SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=prepare_batch,
    )


def build_evaluator(
    device: Any,
    val_data_loader: Any,
    network: Any,
    prepare_batch: Callable,
    metrics: Dict[str, Any],
    output_transform: Optional[Callable] = None,
) -> SupervisedEvaluator:
    """
    Build a MONAI SupervisedEvaluator using MCP-supplied components.
    This function is fully protocol-driven; metrics and output_transform should always
    be obtained from the MCP model class.
    """
    assert callable(prepare_batch), "prepare_batch must be callable"
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        prepare_batch=prepare_batch,
    )
    # Attach metrics after construction
    if metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
    return evaluator
