# train.py
# Set TensorFlow logging level to suppress warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import wandb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import build_tf_dataset

# Global configuration
INPUT_SHAPE = (224, 224, 1)  # (512, 512, 1)


# Build data augmentation pipeline
def build_data_augmentation(rotation=0.1, zoom=0.1, translation=0.1):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(rotation),
        layers.RandomZoom(zoom),
        layers.RandomTranslation(translation, translation)
    ], name="data_augmentation")


# Build a simple CNN model with 2 convolutional layers, dropout and augmentation
def build_model(input_shape, filters=32, kernel_size=3, dropout=0.3, rotation=0.1, zoom=0.1, translation=0.1):
    data_augmentation = build_data_augmentation(rotation, zoom, translation)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(filters * 2, kernel_size, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def main():
    # Initialize wandb
    wandb.init(project="baseline_part_2_cnn_2_conv_layers_dropout")
    config = wandb.config

    # Data loading
    metadata = pd.read_csv("../data/processed/cbis_ddsm_metadata_full.csv")
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)
    train_meta.to_csv("../temporary/train_split.csv", index=False)
    val_meta.to_csv("../temporary/val_split.csv", index=False)

    # Datasets
    train_ds = build_tf_dataset(metadata_csv="../temporary/train_split.csv", batch_size=config.batch_size)
    val_ds = build_tf_dataset(metadata_csv="../temporary/val_split.csv", batch_size=config.batch_size)
    train_ds = train_ds.map(lambda x, y: (x, y["classification"])).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, y["classification"])).prefetch(tf.data.AUTOTUNE)

    # Model
    model = build_model(
        input_shape=INPUT_SHAPE,
        filters=config.filters,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        rotation=config.rotation,
        zoom=config.zoom,
        translation=config.translation
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]# def build_model(input_shape, filters=32, kernel_size=3, dropout=0.3):
#     model = models.Sequential([
#         layers.Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D(2),
#         layers.Conv2D(filters * 2, kernel_size, activation='relu'),
#         layers.MaxPooling2D(2),
#         layers.Flatten(),
#         layers.Dropout(dropout),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     return model
    )

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[
            wandb.keras.WandbMetricsLogger(),
            EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
        ]
        # verbose=2
    )


if __name__ == "__main__":
    main()
