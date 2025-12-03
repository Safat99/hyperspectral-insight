import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True):
    """
    Plot confusion matrix heatmap.
    Accepts integer labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, 
                annot=True, 
                fmt=".2f" if normalize else "d", 
                cmap="Blues",
                xticklabels=class_names, 
                yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
# Training History
# Accepts history dict
def plot_training_history(history : dict, save_path=None, title="Training History"):
    """
    Plot loss & accuracy curves.
    HPC-safe:
        - Does not call plt.show() as there is nothing to plot unless save_path is None
        - Saves figure when save_path is given
    """
    plt.figure(figsize=(10, 4))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train-loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val-loss")
    plt.title("Loss")
    plt.legend()

    # --- Accuracy---
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    
    # SAVE or SHOW
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        

# -------------------------------------------
# IBRA Visualization Tools
# -------------------------------------------
def plot_ibra_distances(distances, title="IBRA Distances"):
    """
    Plot IBRA distance signal.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(distances, label="|d_left - d_right|")
    plt.xlabel("Band Index")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ibra_candidate_bands(distances, candidates, title="IBRA Band Candidates"):
    """
    Plot IBRA distances and mark candidate band centers.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(distances, label="IBRA Distances")
    plt.scatter(candidates, distances[candidates], color="red", label="Candidates")
    plt.xlabel("Band Index")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

