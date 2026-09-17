"""
MNIST Digit Classification with Logistic Regression (PyTorch)
-----------------------------------------------------------------
Trains a single-layer logistic regression model on MNIST using PyTorch.
Covers data loading, model definition, training with loss tracking,
evaluation, and a loss curve visualization.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    data_root: str = "./data"
    input_size: int = 784
    num_classes: int = 10
    num_epochs: int = 5
    batch_size: int = 100
    learning_rate: float = 0.001
    log_every: int = 100


class LogisticRegression(nn.Module):
    """A single linear layer; softmax is applied internally by the loss function."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


def load_data(cfg: PipelineConfig):
    """Download (if needed) and load the MNIST train/test datasets and loaders."""
    train_dataset = dsets.MNIST(
        root=cfg.data_root, train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root=cfg.data_root, train=False, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model: nn.Module, train_loader, cfg: PipelineConfig) -> list:
    """Train the model with SGD and cross-entropy loss; returns per-step loss history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_history = []
    steps_per_epoch = len(train_loader)

    for epoch in range(cfg.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, cfg.input_size)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (i + 1) % cfg.log_every == 0:
                print(
                    f"Epoch: [{epoch + 1}/{cfg.num_epochs}], "
                    f"Step: [{i + 1}/{steps_per_epoch}], Loss: {loss.item():.4f}"
                )

    return loss_history


def evaluate_model(model: nn.Module, test_loader, cfg: PipelineConfig) -> float:
    """Evaluate accuracy on the test set."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, cfg.input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def plot_loss_curve(loss_history: list) -> None:
    """Plot training loss recorded at every step."""
    plt.plot(loss_history)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, train the model, evaluate, and visualize the loss curve."""
    train_loader, test_loader = load_data(cfg)

    model = LogisticRegression(cfg.input_size, cfg.num_classes)

    loss_history = train_model(model, train_loader, cfg)
    plot_loss_curve(loss_history)

    accuracy = evaluate_model(model, test_loader, cfg)
    print(f"Accuracy of the model on the 10000 test images: {accuracy:.2f}%")


if __name__ == "__main__":
    run_pipeline()