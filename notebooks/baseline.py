# baseline.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import wandb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

from data_utils import build_tf_dataset

# --- Configuration ---
INPUT_SHAPE = (224, 224, 1)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_DIR = "./models"
HISTORY_DIR = "../results/history"
PROJECT = "baseline_part_1_cnn_2_conv_layers"
LABELS = ["BENIGN", "MALIGNANT"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# --- Model Definition ---
def build_shallow_cnn(input_shape=INPUT_SHAPE, num_classes=1):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])

# --- Utility Functions ---
def plot_history(history, metrics=('accuracy', 'auc', 'precision', 'recall')):
    plt.figure(figsize=(12, 10))
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        plt.plot(history[metric], label='Train')
        plt.plot(history['val_' + metric], label='Validation')
        plt.title(metric.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return cm

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

# --- Main Workflow ---
def main():
    # 1. Load and split metadata
    metadata = pd.read_csv("../data/processed/cbis_ddsm_metadata_full.csv")
    train_meta, val_meta = train_test_split(
        metadata, test_size=0.2, stratify=metadata['label'], random_state=42
    )
    train_meta.to_csv("../temporary/train_split.csv", index=False)
    val_meta.to_csv("../temporary/val_split.csv", index=False)

    # 2. Build datasets
    train_ds = build_tf_dataset(metadata_csv="../temporary/train_split.csv", batch_size=BATCH_SIZE)
    val_ds = build_tf_dataset(metadata_csv="../temporary/val_split.csv", batch_size=BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (x, y["classification"])).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, y["classification"])).prefetch(tf.data.AUTOTUNE)

    # 3. Build and compile model
    model = build_shallow_cnn(INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # 4. Setup W&B logging
    wandb.init(project=PROJECT, config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "architecture": "shallow_cnn"
    })

    # 5. Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            wandb.keras.WandbMetricsLogger(),
            wandb.keras.WandbModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "best_model_epoch.keras"),
                monitor="val_loss", save_best_only=True
            ),
            EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
        ]
    )

    # 6. Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(HISTORY_DIR, "baseline-part-1-cnn-2-conv-layers.csv"), index=True)
    save_pickle(history.history, os.path.join(HISTORY_DIR, "baseline-part-1-cnn-2-conv-layers.pkl"))

    # 7. Plotting
    plot_loss(history.history)
    plot_history(history.history)

    # 8. Gather validation data for confusion matrix
    val_images, val_labels = [], []
    for imgs, labels in val_ds:
        val_images.append(imgs.numpy())
        val_labels.append(labels.numpy())
    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    y_prob = model.predict(val_images)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = val_labels.flatten()
    cm = plot_confusion_matrix(y_true, y_pred, LABELS)

    # Log confusion matrix image to W&B (optional)
    # wandb.log({"confusion_matrix": wandb.Image(plt)})

    # 9. Save final model
    model.save(os.path.join(MODEL_DIR, "baseline-part-1-cnn-2-conv-layers.keras"))

    # 10. End W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
