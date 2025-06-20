# train.py
# To run sweeps:
# tensorboard --logdir logs/fit
# wandb sweep --project classic_unet_segmentation sweep.yaml
# wandb agent <entity/project/sweep-id>
# wandb agent tkshfj-bsc-computer-science-university-of-london/classic_unet_segmentation/sweep_id

# Import necessary libraries
import os
import wandb
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt
from data_utils import build_dataset
from plot_utils import plot_training_curves, plot_example_predictions
import datetime

# Optional: enable pipeline timing for bottleneck diagnosis
DEBUG_TIMING = False

# Enable dynamic GPU memory growth (prevents OOM/fragmentation)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] Enabled GPU memory growth.")
    except RuntimeError as e:
        print(e)

INPUT_SHAPE = (256, 256, 1)


# Build U-Net (same as your current version)
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


# Dice and IOU metric/loss
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


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


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

    # Check data bugs
    # for imgs, masks in train_ds.take(1):
    #     import matplotlib.pyplot as plt
    #     for i in range(min(4, imgs.shape[0])):
    #         plt.figure(figsize=(6,3))
    #         plt.subplot(1,2,1)
    #         plt.imshow(imgs[i,...,0], cmap="gray")
    #         plt.title("Image")
    #         plt.axis('off')
    #         plt.subplot(1,2,2)
    #         plt.imshow(masks[i,...,0], cmap="gray")
    #         plt.title("Mask")
    #         plt.axis('off')
    #         plt.show()
    #         print("Unique values in mask:", tf.unique(tf.reshape(masks[i], [-1]))[0].numpy())

    # (Optional) Pipeline bottleneck diagnosis
    if DEBUG_TIMING:
        import time
        print("[DEBUG] Measuring data pipeline speeds (one epoch)...")
        t0 = time.time()
        for i, batch in enumerate(train_ds):
            if i >= 5:
                break
        print(f"[DEBUG] Time for 5 batches: {time.time() - t0:.2f}s")

    # Build and compile model
    model = build_unet(input_shape=input_shape, dropout=config.dropout, l2_reg=config.l2_reg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coefficient, iou_metric]
    )
    print(f"Training on {input_shape} images with batch size {config.batch_size} for {config.epochs} epochs.")
    print(f"Using dropout: {config.dropout}, L2 regularization: {config.l2_reg}, learning rate: {config.learning_rate}")
    model.summary()

    # Setup TensorBoard logging
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch='10,20',  # Profile the 10th to 20th batches
        write_graph=True
    )

    # Callbacks: WandB logging, clean checkpointing
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("unet_best.keras", save_best_only=True, monitor="val_loss"),
        # Modern WandB logging
        wandb.keras.WandbMetricsLogger(),
        wandb.keras.WandbModelCheckpoint("unet_best_wandb.keras", monitor="val_loss", save_best_only=True),
        tensorboard_cb
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

    # Plot loss/metrics (log to both W&B and local file)
    plot_training_curves(history, save_path="training_curves.png", log_to_wandb=True)

    # Evaluate and plot predictions
    for batch in test_ds.take(1):
        imgs, masks = batch
        preds = model.predict(imgs)
        plot_example_predictions(imgs, masks, preds, max_examples=4, save_path="prediction", log_to_wandb=True)

    # Save final model (in modern Keras format)
    model.save("./models/unet_final.keras")
    wandb.save("./models/unet_final.keras")

    wandb.finish()


if __name__ == "__main__":
    main()
