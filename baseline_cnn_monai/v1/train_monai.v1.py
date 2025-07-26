# train_monai.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import wandb
import torch
import pandas as pd
import numpy as np
from data_utils import build_dataloaders
from monai.networks.nets import DenseNet121
from torch.optim import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_SHAPE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(name, parameters, lr):
    if name.lower() == "adam":
        return Adam(parameters, lr=lr)
    elif name.lower() == "sgd":
        return SGD(parameters, lr=lr)
    elif name.lower() == "rmsprop":
        return RMSprop(parameters, lr=lr)
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
    wandb.init()
    config = wandb.config

    # Data preparation (single CSV, same as before)
    train_loader, val_loader = build_dataloaders(
        metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
        batch_size=config.batch_size,
        shuffle=True,
        input_shape=INPUT_SHAPE,
        mode="classification",  # can use "segmentation" if needed
        rotation=getattr(config, "rotation", 0.0),
        zoom=getattr(config, "zoom", 0.0),
        # rotation=config.rotation,
        # zoom=config.zoom,
    )

    # Model definition (DenseNet121 for binary classification)
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1).to(DEVICE)
    # optimizer = Adam(model.parameters(), lr=config.learning_rate)
    optimizer_name = getattr(config, "optimizer", "Adam")
    optimizer = get_optimizer(optimizer_name, model.parameters(), config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses, y_true, y_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE).view(-1, 1)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                y_true.extend(labels.cpu().numpy().flatten())
                y_probs.extend(probs.cpu().numpy().flatten())
        avg_val_loss = np.mean(val_losses)
        y_pred = (np.array(y_probs) > 0.5).astype(int)

        val_auc = roc_auc_score(y_true, y_probs)
        val_acc = accuracy_score(y_true, y_pred)
        val_prec = precision_score(y_true, y_pred, zero_division=0)
        val_rec = recall_score(y_true, y_pred, zero_division=0)

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "epoch": epoch,
        })

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), "./models/model_final.pth")
            wandb.save("./models/model_final.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation and confusion matrix
    # model.load_state_dict(torch.load("./models/model_final.pth", map_location=DEVICE))
    state_dict = torch.load("./models/model_final.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE).view(-1, 1)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            y_true.extend(labels.cpu().numpy().flatten())
            y_probs.extend(probs.cpu().numpy().flatten())
    y_pred = (np.array(y_probs) > 0.5).astype(int)
    cm, fig = plot_confusion_matrix(y_true, y_pred, labels=["BENIGN", "MALIGNANT"])
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    wandb.finish()

if __name__ == "__main__":
    main()
