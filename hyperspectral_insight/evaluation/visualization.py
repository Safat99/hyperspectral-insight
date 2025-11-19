import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True):
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
# Training History
def plot_training_history(history, title="Training History"):
    """
    Plot loss & accuracy curves.
    """
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

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

