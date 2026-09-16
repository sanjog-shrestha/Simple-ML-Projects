"""
Handwritten Digit Classification with MLPClassifier (scikit-learn)
----------------------------------------------------------------------
Classifies handwritten digits from scikit-learn's built-in `digits`
dataset (8x8 grayscale images, 10 classes) using a small multi-layer
perceptron. Covers data loading, preview visualization, train/test
splitting, training, loss curve, accuracy, and a confusion matrix.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    hidden_layer_size: int = 15
    activation: str = "logistic"
    alpha: float = 1e-4
    solver: str = "sgd"
    tol: float = 1e-4
    learning_rate_init: float = 0.1
    random_state: int = 1
    test_size: float = 0.2
    preview_count: int = 16


def load_data():
    """Load the scikit-learn digits dataset."""
    digits = datasets.load_digits()
    print(f"Loaded {len(digits.images)} images, image shape: {digits.images[0].shape}")
    return digits


def preview_samples(digits, cfg: PipelineConfig) -> None:
    """Display a grid of sample digit images with their true labels."""
    fig = plt.figure(figsize=(10, 10))
    for i in range(cfg.preview_count):
        plt.subplot(4, 4, i + 1)
        plt.imshow(digits.images[i], cmap="binary")
        plt.title(digits.target[i])
        plt.axis("off")
    plt.show()


def prepare_datasets(digits, cfg: PipelineConfig):
    """Flatten images to feature vectors and split into train/test sets."""
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )


def build_mlp(cfg: PipelineConfig) -> MLPClassifier:
    """Create the MLP classifier with the configured hyperparameters."""
    return MLPClassifier(
        hidden_layer_sizes=(cfg.hidden_layer_size,),
        activation=cfg.activation,
        alpha=cfg.alpha,
        solver=cfg.solver,
        tol=cfg.tol,
        random_state=cfg.random_state,
        learning_rate_init=cfg.learning_rate_init,
        verbose=True,
    )


def plot_loss_curve(model: MLPClassifier) -> None:
    """Plot the training loss curve recorded during fitting."""
    fig, ax = plt.subplots(1, 1)
    ax.plot(model.loss_curve_, "o-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    plt.show()


def evaluate_model(model: MLPClassifier, X_test, y_test) -> dict:
    """Predict on the test set and compute accuracy."""
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred), "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Render a confusion matrix heatmap for predicted vs. actual digit labels."""
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, train the MLP, evaluate, and visualize results."""
    digits = load_data()
    preview_samples(digits, cfg)

    X_train, X_test, y_train, y_test = prepare_datasets(digits, cfg)

    model = build_mlp(cfg)
    model.fit(X_train, y_train)
    plot_loss_curve(model)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    plot_confusion_matrix(y_test, metrics["y_pred"])


if __name__ == "__main__":
    run_pipeline()