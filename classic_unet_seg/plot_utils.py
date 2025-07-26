import matplotlib.pyplot as plt
import wandb
# import numpy as np


# Plotting utilities for training curves
def plot_training_curves(history, save_path=None, log_to_wandb=False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Val Loss')
    axs[0].legend()
    axs[0].set_title('Loss')
    axs[1].plot(history.history['dice_coefficient'], label='Train Dice')
    axs[1].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axs[1].legend()
    axs[1].set_title('Dice Coefficient')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if log_to_wandb:
        wandb.log({"training_curves": wandb.Image(plt)})
    plt.show()
    plt.close()


# Evaluate and log predictions for a few examples
def plot_example_predictions(images, masks, preds, max_examples=4, save_path=None, log_to_wandb=False):
    for i in range(min(max_examples, images.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(images[i, ..., 0], cmap='gray')
        axs[0].set_title('Image')
        axs[1].imshow(masks[i, ..., 0], cmap='gray')
        axs[1].set_title('True Mask')
        axs[2].imshow(preds[i, ..., 0] > 0.5, cmap='gray')
        axs[2].set_title('Pred Mask')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_example_{i}.png")
        if log_to_wandb:
            wandb.log({f"example_pred_{i}": wandb.Image(plt)})
        plt.show()
        plt.close()
