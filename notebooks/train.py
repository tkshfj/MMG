# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import wandb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_utils import build_tf_dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_SHAPE = (224, 224, 1)

# Augmentation
def build_data_augmentation(rotation=0.1, zoom=0.1, translation=0.1):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(rotation),
        layers.RandomZoom(zoom),
        layers.RandomTranslation(translation, translation)
    ], name="data_augmentation")

# Model
def build_model(input_shape, filters=32, kernel_size=3, dropout=0.3, rotation=0.1, zoom=0.1, translation=0.1):
    data_augmentation = build_data_augmentation(rotation, zoom, translation)
    return models.Sequential([
        layers.Input(shape=input_shape),
        # Augmentation
        data_augmentation,
        # Feature extraction
        layers.Conv2D(filters, kernel_size, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(filters * 2, kernel_size, activation='relu'),
        layers.MaxPooling2D(2),
        # Classification head
        layers.Flatten(),
        layers.Dense(filters * 2, activation='relu'),  # filters * 4
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

def get_optimizer(name, learning_rate):
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate)
    elif name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    return cm, plt

def main():
    # Initialize wandb
    # wandb.init(project="baseline_cnn_dropout_augmentation")
    wandb.init()
    wandb.config.update({"optimizer": "Adam"}, allow_val_change=True)
    config = wandb.config

    # Data
    metadata = pd.read_csv("../data/processed/cbis_ddsm_metadata_full.csv")
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)
    train_meta.to_csv("../temporary/train_split.csv", index=False)
    val_meta.to_csv("../temporary/val_split.csv", index=False)

    train_ds = build_tf_dataset("../temporary/train_split.csv", batch_size=config.batch_size)
    val_ds = build_tf_dataset("../temporary/val_split.csv", batch_size=config.batch_size)
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

    optimizer = get_optimizer(config.optimizer, config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[
            wandb.keras.WandbMetricsLogger(),
            EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
        ]
    )

    # Save model
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/model_final.keras"
    model.save(model_path)
    wandb.save(model_path)

    # Final evaluation
    val_images, val_labels = [], []
    for x, y in val_ds:
        val_images.append(x.numpy())
        val_labels.append(y.numpy())

    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0).flatten()

    y_probs = model.predict(val_images)
    y_pred = (y_probs > 0.5).astype(int).flatten()

    val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_ds, verbose=0)
    wandb.log({
        "val_final_loss": val_loss,
        "val_final_accuracy": val_acc,
        "val_final_auc": val_auc,
        "val_final_precision": val_prec,
        "val_final_recall": val_rec
    })

    # Confusion matrix
    cm, fig = plot_confusion_matrix(val_labels, y_pred, labels=["BENIGN", "MALIGNANT"])
    wandb.log({"confusion_matrix": wandb.Image(fig)})

    wandb.finish()

if __name__ == "__main__":
    main()
