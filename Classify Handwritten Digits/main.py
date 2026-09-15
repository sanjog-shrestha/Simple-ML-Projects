"""
MNIST Digit Classification — Softmax vs. MLP
----------------------------------------------
Trains and compares two models on the MNIST handwritten digit dataset:
a single-layer softmax classifier (logistic regression baseline) and a
multi-layer perceptron (MLP). Covers data loading, training, evaluation,
sample visualization, and model save/reload.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    softmax_epochs: int = 10
    mlp_epochs: int = 3
    batch_size: int = 100
    model_save_path: str = "mnist_mlp.keras"


def load_mnist_data():
    """Load MNIST once and return normalized train/test splits (both flat and 2D forms)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_norm = tf.keras.utils.normalize(x_train, axis=1)
    x_test_norm = tf.keras.utils.normalize(x_test, axis=1)

    x_train_flat = x_train_norm.reshape(x_train_norm.shape[0], -1).astype("float32")
    x_test_flat = x_test_norm.reshape(x_test_norm.shape[0], -1).astype("float32")

    return x_train_norm, x_test_norm, x_train_flat, x_test_flat, y_train, y_test


def build_softmax_model(input_dim: int) -> tf.keras.Model:
    """Single dense layer with softmax activation — a logistic regression baseline."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp_model() -> tf.keras.Model:
    """Two hidden dense layers on the flattened 28x28 image."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def show_digit(image: np.ndarray, true_label: int, predicted_label: int = None) -> None:
    """Display a single digit image with its true (and optionally predicted) label."""
    title = f"label: {true_label}"
    if predicted_label is not None:
        title += f"  |  prediction: {predicted_label}"
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(title)
    plt.axis("off")
    plt.show()


def evaluate_model(model: tf.keras.Model, X_test, y_test) -> dict:
    """Evaluate a trained model on the test set."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"loss": loss, "accuracy": accuracy}


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, train both models, compare results, and demonstrate save/reload."""
    x_train_2d, x_test_2d, x_train_flat, x_test_flat, y_train, y_test = load_mnist_data()

    show_digit(x_train_2d[0], y_train[0])

    # --- Model 1: Softmax baseline ---
    print("\n=== Training: Softmax Baseline ===")
    softmax_model = build_softmax_model(input_dim=x_train_flat.shape[1])
    softmax_model.fit(x_train_flat, y_train, batch_size=cfg.batch_size, epochs=cfg.softmax_epochs, verbose=1)
    softmax_metrics = evaluate_model(softmax_model, x_test_flat, y_test)
    print(f"Softmax  -> loss: {softmax_metrics['loss']:.4f}, accuracy: {softmax_metrics['accuracy']:.4f}")

    # --- Model 2: MLP ---
    print("\n=== Training: MLP (2 hidden layers) ===")
    mlp_model = build_mlp_model()
    mlp_model.fit(x_train_2d, y_train, epochs=cfg.mlp_epochs, verbose=1)
    mlp_metrics = evaluate_model(mlp_model, x_test_2d, y_test)
    print(f"MLP      -> loss: {mlp_metrics['loss']:.4f}, accuracy: {mlp_metrics['accuracy']:.4f}")

    # --- Compare on one sample ---
    sample_idx = 2
    mlp_pred = np.argmax(mlp_model.predict(x_test_2d[sample_idx:sample_idx + 1], verbose=0))
    show_digit(x_test_2d[sample_idx], y_test[sample_idx], predicted_label=mlp_pred)

    # --- Save and reload the MLP model ---
    mlp_model.save(cfg.model_save_path)
    reloaded_model = tf.keras.models.load_model(cfg.model_save_path)
    reloaded_pred = np.argmax(reloaded_model.predict(x_test_2d[sample_idx:sample_idx + 1], verbose=0))
    print(f"\nReloaded model prediction: {reloaded_pred}  |  true label: {y_test[sample_idx]}")


if __name__ == "__main__":
    run_pipeline()