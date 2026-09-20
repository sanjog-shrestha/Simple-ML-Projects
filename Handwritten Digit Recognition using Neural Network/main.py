"""
Digit Classification from CSV Pixel Data (Keras)
------------------------------------------------------
Trains a small dense neural network to classify handwritten digits from
a Kaggle-style CSV dataset (each row: a label + 784 flattened pixel
values). Covers data loading, preprocessing, train/val split, model
training, evaluation, and prediction preview on a separate test file.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    train_csv_path: str = "/content/Train.csv"
    test_csv_path: str = "test.csv"
    image_shape: tuple = (28, 28, 1)
    num_classes: int = 10
    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 10
    batch_size: int = 32
    preview_predictions: int = 5


def load_train_data(cfg: PipelineConfig):
    """Load the training CSV and split into raw features (X) and label (y)."""
    train_data = pd.read_csv(cfg.train_csv_path)
    print("Shape of train_data:", train_data.shape)

    X = train_data.iloc[:, 1:]
    y = train_data.iloc[:, 0]
    print("Shape of X after separating features:", X.shape)
    return X, y


def preprocess_features(X, cfg: PipelineConfig) -> np.ndarray:
    """Clean, normalize, and reshape raw pixel data into image tensors."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.values / 255.0
    X = X.reshape(-1, *cfg.image_shape)
    print("Shape of X after reshaping:", X.shape)
    return X


def preprocess_labels(y, cfg: PipelineConfig) -> np.ndarray:
    """One-hot encode integer labels."""
    y = to_categorical(y, num_classes=cfg.num_classes)
    print("Shape of y after one-hot encoding:", y.shape)
    return y


def build_model(cfg: PipelineConfig) -> Sequential:
    """Build a small dense classifier over flattened pixel input."""
    model = Sequential([
        Input(shape=cfg.image_shape),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(cfg.num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model: Sequential, X_train, y_train, X_val, y_val, cfg: PipelineConfig):
    """Train the model, tracking validation performance each epoch."""
    return model.fit(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_val, y_val),
    )


def evaluate_model(model: Sequential, X_val, y_val) -> dict:
    """Evaluate the trained model on the validation set."""
    loss, accuracy = model.evaluate(X_val, y_val)
    return {"loss": loss, "accuracy": accuracy}


def plot_training_curves(history) -> None:
    """Plot training vs. validation accuracy over epochs."""
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.show()


def load_test_data(cfg: PipelineConfig) -> np.ndarray:
    """Load and preprocess the separate test CSV (no labels)."""
    test_data = pd.read_csv(cfg.test_csv_path)
    X_test = test_data.values / 255.0
    X_test = X_test.reshape(-1, *cfg.image_shape)
    return X_test


def preview_predictions(model: Sequential, X_test, cfg: PipelineConfig) -> None:
    """Predict on the test set and display a few sample predictions."""
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(cfg.preview_predictions):
        plt.imshow(X_test[i].reshape(cfg.image_shape[:2]), cmap="gray")
        plt.title(f"Predicted Label: {predicted_labels[i]}")
        plt.axis("off")
        plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, train the model, evaluate, and preview test predictions."""
    X, y = load_train_data(cfg)
    X = preprocess_features(X, cfg)
    y = preprocess_labels(y, cfg)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model = build_model(cfg)
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val, cfg)

    metrics = evaluate_model(model, X_val, y_val)
    print(f"Validation Accuracy: {metrics['accuracy'] * 100:.2f}%")

    plot_training_curves(history)

    X_test = load_test_data(cfg)
    preview_predictions(model, X_test, cfg)


if __name__ == "__main__":
    run_pipeline()