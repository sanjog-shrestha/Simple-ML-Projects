"""
CIFAR-10 Image Classification with a CNN
---------------------------------------------
Trains a convolutional neural network to classify CIFAR-10 images into
10 categories. Covers data loading, preprocessing, sample visualization,
model definition, training, test-set evaluation, and training curves.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    num_classes: int = 10
    epochs: int = 30
    batch_size: int = 64
    validation_split: float = 0.2
    preview_count: int = 16


CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]


def load_data(cfg: PipelineConfig):
    """Load CIFAR-10, normalize pixel values, and one-hot encode labels if needed."""
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # CIFAR-10 labels ship as integer class ids (uint8); only one-hot encode if not already done.
    if y_train.dtype == np.uint8:
        y_train = to_categorical(y_train, num_classes=cfg.num_classes)
        y_test = to_categorical(y_test, num_classes=cfg.num_classes)

    return X_train, y_train, X_test, y_test


def preview_samples(X, y, cfg: PipelineConfig) -> None:
    """Display a grid of sample images with their class labels."""
    plt.figure(figsize=(10, 10))
    for i in range(cfg.preview_count):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        plt.xlabel(CLASS_NAMES[np.argmax(y[i])])
    plt.show()


def build_model(cfg: PipelineConfig) -> models.Sequential:
    """Build a 3-block CNN: (Conv-Conv-Pool-Dropout) x3, then a dense classifier head."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(cfg.num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model: models.Sequential, X_train, y_train, cfg: PipelineConfig):
    """Train the model, holding out a validation split from the training data."""
    return model.fit(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
    )


def evaluate_model(model: models.Sequential, X_test, y_test) -> dict:
    """Evaluate the trained model on the held-out test set."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"loss": loss, "accuracy": accuracy}


def plot_training_curves(history) -> None:
    """Plot training/validation accuracy and loss side by side."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, preview samples, train the CNN, evaluate, and plot training curves."""
    X_train, y_train, X_test, y_test = load_data(cfg)
    preview_samples(X_train, y_train, cfg)

    model = build_model(cfg)
    model.summary()

    history = train_model(model, X_train, y_train, cfg)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nTest Loss: {metrics['loss']:.4f} | Test Accuracy: {metrics['accuracy']:.4f}")

    plot_training_curves(history)


if __name__ == "__main__":
    run_pipeline()