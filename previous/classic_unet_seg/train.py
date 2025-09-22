# train.py
# To run sweeps:
# tensorboard --logdir logs/fit
# wandb sweep --project classic_unet_segmentation sweep.yaml
# wandb agent <entity/project/sweep-id>
# wandb agent tkshfj-bsc-computer-science-university-of-london/classic_unet_segmentation/sweep_id

# Import necessary libraries
import wandb
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_utils import build_dataset
# from plot_utils import plot_training_curves, plot_example_predictions

INPUT_SHAPE = (256, 256, 1)


# Loss and metric definitions
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# U-Net Model Builder
def build_unet(input_shape, dropout=0.3, l2_reg=1e-4):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(p1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(p2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout)(p3)

    # Bottleneck
    bn = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(p3)
    bn = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(bn)

    # Decoder
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn)
    u3 = layers.concatenate([u3, c3])
    u3 = layers.Dropout(dropout)(u3)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(u3)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c6)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    u2 = layers.Dropout(dropout)(u2)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(u2)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c7)

    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    u1 = layers.Dropout(dropout)(u1)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(u1)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(c8)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c8)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Training and evaluation
def main():
    wandb.init(project="classic_unet_segmentation")
    config = wandb.config
    config = wandb.config

    # input_shape = tuple(config.input_shape) if isinstance(config.input_shape, (list, tuple)) else INPUT_SHAPE
    input_shape = tuple(config.input_shape) if hasattr(config, "input_shape") else INPUT_SHAPE  # (256, 256, 1)
    batch_size = config.batch_size if hasattr(config, "batch_size") else 8
    # learning_rate = config.base_learning_rate * (config.batch_size / 16) * config.lr_multiplier
    learning_rate = config.learning_rate if hasattr(config, "learning_rate") else 1e-4
    epochs = config.epochs if hasattr(config, "epochs") else 40
    dropout = config.dropout if hasattr(config, "dropout") else 0.3
    l2_reg = config.l2_reg if hasattr(config, "l2_reg") else 1e-4

    # Data loading
    train_ds, val_ds, test_ds = build_dataset(
        metadata_csv='../data/processed/cbis_ddsm_metadata_full.csv',
        input_shape=input_shape,
        batch_size=batch_size,
        task='segmentation',
        shuffle=True,
        augment=True,
        split=(0.7, 0.15, 0.15)
    )

    # Model build/compile
    model = build_unet(input_shape=input_shape, dropout=dropout, l2_reg=l2_reg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_metric]
    )
    model.summary(print_fn=wandb.termlog)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("unet_best.keras", save_best_only=True),
        wandb.keras.WandbCallback(
            monitor="val_loss",
            save_model=True,
            log_weights=False,
            log_evaluation=True,
            predictions=16,
            validation_data=val_ds
        ),
    ]

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # # Plot loss/metrics (log to both W&B and local file)
    # plot_training_curves(history, save_path="training_curves.png", log_to_wandb=True)

    # # Evaluate and plot predictions
    # for batch in test_ds.take(1):
    #     imgs, masks = batch
    #     preds = model.predict(imgs)
    #     plot_example_predictions(imgs, masks, preds, max_examples=4, save_path="prediction", log_to_wandb=True)

    # Save final model
    model.save("./models/unet_final.keras")
    wandb.save("./models/unet_final.keras")

    # Evaluate
    results = model.evaluate(test_ds)
    print(f"Test loss: {results[0]:.4f}, Dice: {results[1]:.4f}, IoU: {results[2]:.4f}")
    wandb.log({"test_loss": results[0], "test_dice": results[1], "test_iou": results[2]})

    wandb.finish()


if __name__ == "__main__":
    main()
