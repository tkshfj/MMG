# baseline.py
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from data_utils import build_tf_dataset, INPUT_SHAPE

# Configuration
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_DIR = "./models"
HISTORY_DIR = "../results/history"
PROJECT = "baseline_shallow_cnn"
ENTITY = "tkshfj-bsc-computer-science-university-of-london"
LABELS = ["BENIGN", "MALIGNANT"]
MODEL_FILENAME = "baseline-shallow_cnn.keras"
MODEL_ARTIFACT_NAME = "baseline-shallow_cnn_v1"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

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

# Plotting Utilities
def plot_history(history, metrics=('accuracy', 'auc', 'precision', 'recall')):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        axes[idx].plot(history[metric], label='Train')
        axes[idx].plot(history[f'val_{metric}'], label='Validation')
        axes[idx].set_title(metric.capitalize())
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].legend()
    plt.tight_layout()
    return fig

def plot_loss(history):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig

# Custom Callback for logging epoch count
class LogEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["epoch"] = epoch
        wandb.log({f"epoch/{k}": v for k, v in logs.items()}, step=epoch)

def train():
    wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="baseline-shallow-v1",
        tags=["baseline", "shallow_cnn"],
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "architecture": "shallow_cnn"
        }
    )

    metadata = pd.read_csv("../data/processed/cbis_ddsm_metadata_full.csv")
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, stratify=metadata['label'], random_state=42)
    train_meta.to_csv("../temporary/train_split.csv", index=False)
    val_meta.to_csv("../temporary/val_split.csv", index=False)

    train_ds = build_tf_dataset("../temporary/train_split.csv")
    val_ds = build_tf_dataset("../temporary/val_split.csv")
    train_ds = train_ds.map(lambda x, y: (x, y["classification"])).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, y["classification"])).cache().prefetch(tf.data.AUTOTUNE)

    model = build_shallow_cnn(INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            wandb.keras.WandbMetricsLogger(),
            LogEpochCallback(),
            tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)
        ]
    )

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(HISTORY_DIR, "baseline_shallow_cnn.csv"), index=True)
    with open(os.path.join(HISTORY_DIR, "baseline_shallow_cnn.pkl"), 'wb') as f:
        pickle.dump(history.history, f)

    # Log figures
    wandb.log({
        "metric_plots": wandb.Image(plot_history(history.history)),
        "loss_plot": wandb.Image(plot_loss(history.history))
    })

    # Evaluate and log confusion matrix
    val_images, val_labels = [], []
    for x, y in val_ds:
        val_images.append(x.numpy())
        val_labels.append(y.numpy())
    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels).flatten()

    y_pred = (model.predict(val_images) > 0.5).astype(int).flatten()

    wandb.log({
        "confusion_matrix": wandb.Image(plot_confusion_matrix(val_labels, y_pred, LABELS)),
        "predictions_table": wandb.Table(data=[[int(t), int(p)] for t, p in zip(val_labels, y_pred)],
                                         columns=["true", "predicted"])
    })

    # Log evaluation metrics
    eval_metrics = model.evaluate(val_ds, return_dict=True)
    wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()})

    # Save and log model artifact
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    model.save(model_path)

    artifact = wandb.Artifact(MODEL_ARTIFACT_NAME, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    train()
