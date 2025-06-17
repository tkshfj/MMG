# unet_train.py
# To run sweeps:
# wandb sweep sweep.yaml
# wandb agent <entity/project/sweep-id>

# Import necessary libraries
import wandb
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from data_utils import build_dataset
# from wandb.keras import WandbCallback

# Configuration
INPUT_SHAPE = (256, 256, 1)  # (512, 512, 1)


# Build the U-Net model
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


# Losses and metrics
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


def iou_loss(y_true, y_pred):
    return 1 - iou_metric(y_true, y_pred)


# Optionally: combo loss (BCE + Dice)
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# Train
def main():
    # Initialize W&B
    wandb.init(project="classic_unet_segmentation")
    config = wandb.config
    input_shape = tuple(config.input_shape) if isinstance(config.input_shape, (list, tuple)) else INPUT_SHAPE

    # Build datasets
    train_ds, val_ds, test_ds = build_dataset(
        metadata_csv='../data/processed/cbis_ddsm_metadata_full.csv',
        input_shape=input_shape,
        batch_size=config.batch_size,
        task=config.get("task", "segmentation"),
        shuffle=True,
        augment=True,
        split=(0.7, 0.15, 0.15)
    )
    print(f"Training dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}, Test dataset size: {len(test_ds)}")
    # Compile the model
    model = build_unet(input_shape=input_shape, dropout=config.dropout, l2_reg=config.l2_reg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_metric]
    )
    print(f"Training on {input_shape} images with batch size {config.batch_size} for {config.epochs} epochs.")
    print(f"Using dropout: {config.dropout}, L2 regularization: {config.l2_reg}, learning rate: {config.learning_rate}")
    print("Model summary:")
    model.summary()
    wandb.log({"model_summary": model.summary()})

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("unet_best.h5", save_best_only=True),
        wandb.keras.WandbCallback(
            save_model=True,
            monitor="val_loss",
            log_weights=False,
            log_evaluation=True,
            predictions=16,
            log_best_prefix="best_",
            validation_data=val_ds
        ),
    ]
    print("Callbacks:", callbacks)

    # Train the model
    print("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks
    )

    # Save final model
    model.save("./models/unet_final.h5")
    wandb.save("unet_final.h5")

    # Plot loss/metrics
    plot_training_curves(history)
    evaluate_and_log_predictions(model, test_ds)

    # Clean W&B sweep agent shutdown
    wandb.finish()


# Plot training curves
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Train Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
    plt.legend()
    plt.title('Dice Coefficient')
    plt.tight_layout()
    plt.show()


# Evaluate and log predictions
def evaluate_and_log_predictions(model, test_ds):
    for batch in test_ds.take(1):
        imgs, masks = batch
        preds = model.predict(imgs)
        for i in range(min(4, imgs.shape[0])):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(imgs[i, ..., 0], cmap='gray')
            plt.title('Image')
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i, ..., 0], cmap='gray')
            plt.title('True Mask')
            plt.subplot(1, 3, 3)
            plt.imshow(preds[i, ..., 0] > 0.5, cmap='gray')
            plt.title('Pred Mask')
            plt.show()
            wandb.log({"example_pred": [
                wandb.Image(imgs[i, ..., 0], caption="Image"),
                wandb.Image(masks[i, ..., 0], caption="GT"),
                wandb.Image(preds[i, ..., 0] > 0.5, caption="Pred")
            ]})


if __name__ == "__main__":
    main()
