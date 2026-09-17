"""
Cartoon Effect Image Filter (OpenCV)
---------------------------------------
Applies a cartoon-style effect to an image: edge detection via adaptive
thresholding combined with a bilaterally-smoothed color image. Covers
image loading, edge extraction, color smoothing, compositing, display,
and saving.
"""

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    image_path: str = "Ulquiorra.jpg"
    output_path: str = "cartoon_output.jpg"
    median_blur_ksize: int = 5
    edge_block_size: int = 9
    edge_c: int = 9
    bilateral_d: int = 9
    bilateral_sigma_color: int = 250
    bilateral_sigma_space: int = 250


def load_image(cfg: PipelineConfig):
    """Load the source image, raising a clear error if it can't be read."""
    img = cv2.imread(cfg.image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{cfg.image_path}'")
    return img


def extract_edge_mask(img, cfg: PipelineConfig):
    """Convert to grayscale, denoise, and threshold to get a binary edge mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, cfg.median_blur_ksize)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, cfg.edge_block_size, cfg.edge_c,
    )
    return edges


def smooth_color(img, cfg: PipelineConfig):
    """Flatten color regions while preserving edges, using a bilateral filter."""
    return cv2.bilateralFilter(
        img, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
    )


def cartoonize(img, cfg: PipelineConfig):
    """Combine the smoothed color image with the edge mask into a cartoon effect."""
    edges = extract_edge_mask(img, cfg)
    smoothed = smooth_color(img, cfg)
    return cv2.bitwise_and(smoothed, smoothed, mask=edges)


def show_image(img, title: str = "Result") -> None:
    """Display a BGR OpenCV image using matplotlib (portable across environments)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load the image, apply the cartoon effect, display it, and save the result."""
    img = load_image(cfg)
    cartoon = cartoonize(img, cfg)

    show_image(cartoon, title="Cartoon Effect")

    cv2.imwrite(cfg.output_path, cartoon)
    print(f"Saved result to {cfg.output_path}")


if __name__ == "__main__":
    run_pipeline()