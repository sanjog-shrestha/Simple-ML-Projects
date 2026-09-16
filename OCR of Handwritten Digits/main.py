"""
Handwritten Digit Classification with k-Nearest Neighbors (OpenCV)
--------------------------------------------------------------------
Classifies handwritten digits using OpenCV's k-NN implementation on a
grid-of-digits sprite sheet (the classic 50x100 cell "digits.png" layout:
10 digit classes, arranged in row-blocks, split into train/test halves
by column). Covers data loading, training, prediction, accuracy, and
visualization.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import cv2


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    image_path: str = "digits1.png"
    num_rows: int = 50
    num_cols: int = 100
    num_classes: int = 10
    k_neighbors: int = 3


def load_digit_grid(cfg: PipelineConfig) -> np.ndarray:
    """Load the sprite sheet and convert it to grayscale."""
    image = cv2.imread(cfg.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{cfg.image_path}'")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def split_into_cells(gray_image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Split the grid image into individual digit cells: shape (rows, cols, cell_h, cell_w)."""
    rows = np.vsplit(gray_image, cfg.num_rows)
    cells = [np.hsplit(row, cfg.num_cols) for row in rows]
    return np.array(cells)


def prepare_datasets(cells: np.ndarray, cfg: PipelineConfig):
  half = cfg.num_cols // 2
  cell_h, cell_w = cells.shape[2], cells.shape[3]
  flat_dim = cell_h * cell_w

  train_data = np.ascontiguousarray(cells[:, :half].reshape(-1, flat_dim).astype(np.float32))
  test_data = np.ascontiguousarray(cells[:, half:].reshape(-1, flat_dim).astype(np.float32))

  samples_per_class = (cfg.num_rows // cfg.num_classes) * half
  classes = np.arange(cfg.num_classes)
  train_labels = np.repeat(classes, samples_per_class)[:, np.newaxis].astype(np.float32)
  test_labels = np.repeat(classes, samples_per_class)[:, np.newaxis].astype(np.float32)

  return train_data, test_data, train_labels, test_labels


def create_knn_classifier():
    """Create a cv2 KNearest instance, handling API differences across OpenCV versions."""
    try:
        return cv2.ml.KNearest.create()
    except AttributeError:
        try:
            return cv2.ml.KNearest_create()
        except AttributeError as exc:
            raise RuntimeError(
                "KNearest functionality not found in cv2.ml. Check your OpenCV installation."
            ) from exc


def train_and_predict(train_data, train_labels, test_data, cfg: PipelineConfig):
    """Train the k-NN model and predict labels for the test set."""
    print(f"train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
    knn = create_knn_classifier()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    _, predictions, _, _ = knn.findNearest(test_data, k=cfg.k_neighbors)
    return predictions


def compute_accuracy(predictions: np.ndarray, test_labels: np.ndarray) -> float:
    """Percentage of test samples correctly classified."""
    correct = np.count_nonzero(predictions == test_labels)
    return (correct * 100.0) / predictions.size


def show_sample_cell(cells: np.ndarray, row: int, col: int) -> None:
    """Display a single digit cell for a visual sanity check."""
    plt.imshow(cells[row, col], cmap="gray")
    plt.title(f"Sample cell — row {row}, col {col}")
    plt.axis("off")
    plt.show()


def plot_confusion_matrix(test_labels: np.ndarray, predictions: np.ndarray, cfg: PipelineConfig) -> None:
    """Render a confusion matrix heatmap for predicted vs. actual digit labels."""
    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=int)
    for actual, pred in zip(test_labels.flatten(), predictions.flatten()):
        cm[int(actual), int(pred)] += 1

    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(range(cfg.num_classes))
    plt.yticks(range(cfg.num_classes))
    for i in range(cfg.num_classes):
        for j in range(cfg.num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, train the k-NN classifier, evaluate, and visualize results."""
    gray_image = load_digit_grid(cfg)
    cells = split_into_cells(gray_image, cfg)

    show_sample_cell(cells, row=0, col=0)

    train_data, test_data, train_labels, test_labels = prepare_datasets(cells, cfg)

    predictions = train_and_predict(train_data, train_labels, test_data, cfg)

    accuracy = compute_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(test_labels, predictions, cfg)


if __name__ == "__main__":
    run_pipeline()