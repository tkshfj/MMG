# engine_utils.py
from monai.engines import SupervisedTrainer, SupervisedEvaluator


def build_trainer(device, max_epochs, train_data_loader, network, optimizer, loss_function, prepare_batch):
    return SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=prepare_batch
    )


def build_evaluator(device, val_data_loader, network, prepare_batch, key_val_metric):
    return SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        prepare_batch=prepare_batch,
        key_val_metric=key_val_metric
    )
