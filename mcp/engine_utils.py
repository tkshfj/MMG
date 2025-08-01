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
    Build a MONAI SupervisedTrainer using configuration-supplied components.
    All arguments are protocol-driven, so this trainer is fully model and task agnostic.
    - If metrics are supplied, they will be attached at construction (optional; can also be attached externally).
    - If output_transform is not supplied, the default will be used.
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
    Build a MONAI SupervisedEvaluator using configuration-supplied components.
    Metrics and output_transform should always be obtained from the model class.
    """
    assert callable(prepare_batch), "prepare_batch must be callable"
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        prepare_batch=prepare_batch,
    )
    # DEBUG: Attach metrics after construction
    if metrics:
        print("[DEBUG] Attaching metrics to evaluator:")
        for name, metric in metrics.items():
            print(f"  {name} -> {type(metric)}")
            metric.attach(evaluator, name)
        print("------")
    return evaluator
