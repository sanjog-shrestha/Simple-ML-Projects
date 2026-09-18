"""
OCR Text Extraction with pytesseract
-----------------------------------------
Extracts text from an image using Tesseract OCR, and visualizes detected
word-level bounding boxes. Covers image loading, text extraction,
word-box filtering, and visualization.
"""

from dataclasses import dataclass

import cv2
import pytesseract
from matplotlib import pyplot as plt


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    image_path: str = "Original-Image.png"
    min_confidence: float = 0.0
    box_color: tuple = (0, 255, 0)
    box_thickness: int = 2


def load_image(cfg: PipelineConfig):
    """Load the source image as RGB, raising a clear error if it can't be read."""
    image = cv2.imread(cfg.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{cfg.image_path}'")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(image, title: str = "Image") -> None:
    """Display an RGB image with matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def extract_text(image_rgb) -> str:
    """Run OCR and return the extracted text as a single string."""
    return pytesseract.image_to_string(image_rgb)


def extract_word_boxes(image_rgb, cfg: PipelineConfig) -> list:
    """Run OCR with layout data and return only word-level boxes with real text."""
    data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)

    boxes = []
    for i in range(len(data["level"])):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])
        if not text or confidence < cfg.min_confidence:
            continue
        boxes.append({
            "text": text,
            "confidence": confidence,
            "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
        })
    return boxes


def draw_word_boxes(image_rgb, boxes: list, cfg: PipelineConfig):
    """Draw a rectangle around each detected word box."""
    annotated = image_rgb.copy()
    for box in boxes:
        x, y, w, h = box["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), cfg.box_color, cfg.box_thickness)
    return annotated


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load the image, run OCR, print extracted text, and visualize word boxes."""
    image_rgb = load_image(cfg)
    show_image(image_rgb, title="Original Image")

    text = extract_text(image_rgb)
    print("Extracted Text:\n")
    print(text)

    boxes = extract_word_boxes(image_rgb, cfg)
    print(f"\nDetected {len(boxes)} word-level boxes")

    annotated = draw_word_boxes(image_rgb, boxes, cfg)
    show_image(annotated, title="Image with Bounding Boxes")


if __name__ == "__main__":
    run_pipeline()