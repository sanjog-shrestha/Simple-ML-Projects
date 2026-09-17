# Coin Counting with Contour Detection (OpenCV)

A classical computer vision pipeline that counts coins in an image using edge detection and contour finding — no machine learning involved, just image processing.

## What it does

1. Loads the source image, raising a clear error if the file can't be found.
2. Converts to grayscale and applies a Gaussian blur to reduce noise before edge detection.
3. Runs Canny edge detection to find edges, then dilates them slightly to close small gaps (so a coin's outline forms one continuous closed shape rather than a broken one).
4. Finds external contours (outer boundaries) in the dilated edge map, filtering out tiny contours that are too small to plausibly be a coin (noise/artifacts).
5. Draws the detected contours onto the original image and displays it, alongside the final count.

## Concepts covered

- **Gaussian blur before edge detection** — edge detectors like Canny are sensitive to noise; blurring first smooths out small pixel-level variations so only genuine object edges get picked up, not texture noise.
- **Canny edge detection** — a two-threshold edge detector (`canny_low`/`canny_high`) that finds strong edges and connects them to nearby weaker ones, producing a clean binary edge map.
- **Dilation to close gaps** — edges from Canny are sometimes broken into small disconnected segments; dilating them slightly bridges small gaps so contour detection sees one continuous coin outline instead of several fragments.
- **Contour detection (`RETR_EXTERNAL`)** — extracts only the outermost boundary of each connected shape, ideal for counting whole objects like coins rather than every internal edge or texture line.
- **Contour area filtering** — real-world edge detection often picks up small noise blobs as "contours"; filtering by minimum area is a simple way to discard those before counting.

## Project structure

```
main.py      # full pipeline, organized into functions (config, image loading,
              # preprocessing, edge detection, contour finding, visualization)
README.md    # this file
```

The code is organized into small, named functions driven by a `PipelineConfig` dataclass — `load_image`, `preprocess`, `detect_edges`, `find_coin_contours`, `draw_contours`, `show_image` — tied together by `run_pipeline()`.

## Requirements

```
opencv-python
matplotlib
```

Install with:

```bash
pip install opencv-python matplotlib
```

## How to run

Place your source image (e.g. `coins.png`) in the same directory as `main.py`, then:

```bash
python main.py
```

## Sample output

**Original**

![Original image](coins.png)

**Detected coins**

![Detected coins](image.png)


## Notes on this implementation

- **Fixed a real bug**: the original called `cv2.dilate(canny, (1, 1), iterations=0)` — with `iterations=0`, dilation does nothing at all, so that step was silently a no-op. This version uses `iterations=1` with a proper `MORPH_ELLIPSE` structuring element so dilation actually runs and helps close small gaps in the edges.
- Added contour area filtering (`min_contour_area`), since raw Canny + external-contour output often includes tiny noise blobs that would otherwise inflate the coin count. The original counted every contour returned, with no filtering.
- The original only printed the coin count — it drew contours onto an image but never displayed it. This version shows the annotated result so you can visually verify the detection matches the printed count.
- All the "magic numbers" (blur kernel size, Canny thresholds, dilation settings, minimum contour area) now live in one `PipelineConfig` dataclass instead of being inlined, making them easy to tune per image.

## Things I'd like to try next

- Try `cv2.HoughCircles` as an alternative/complementary method, since coins are circular — it might be more robust than generic contour detection for this specific shape.
- Add adaptive thresholding as a preprocessing option for images with uneven lighting.
- Estimate coin denominations by contour size/radius, not just count total coins.

---
*This is a personal learning project, not a production coin-counting system.*
