# Live Webcam Face Detection in Google Colab (dlib + JS bridge)

A real-time face detector that streams live webcam video into a Google Colab notebook and draws bounding boxes around detected faces, using a JavaScript/Python bridge and dlib's built-in face detector.

**This only runs inside Google Colab** — it depends on `google.colab.output` and injects JavaScript to access the browser's camera, so it won't work as a plain local script.

## What it does

1. Injects JavaScript (`video_stream()`) that creates a live `<video>` element in the notebook output, requests webcam access, and continuously captures frames onto a canvas.
2. In a Python loop, repeatedly requests the latest frame (`video_frame`) from the JS side as a base64-encoded JPEG.
3. Decodes each frame into an OpenCV image (`js_to_image`).
4. Runs dlib's HOG-based frontal face detector on the grayscale frame.
5. Draws bounding boxes and labels ("face num1", "face num2", ...) onto a transparent RGBA overlay — not onto the video frame itself, so only the boxes/labels get redrawn each frame instead of re-rendering the whole video.
6. Sends the overlay back to the browser as a base64 PNG, which JS layers on top of the live video.
7. Repeats until the user clicks the video (or the "click here to stop" text) to shut it down.

## Concepts covered

- **JS/Python bridge in Colab** — `eval_js` lets Python call into injected JavaScript and get a return value back, which is how this notebook accesses the browser's webcam (something Python alone can't do in a hosted notebook environment).
- **HOG-based face detection (dlib)** — `dlib.get_frontal_face_detector()` uses a Histogram of Oriented Gradients + linear SVM classifier; it's lightweight and doesn't require downloading a separate model file, unlike dlib's landmark predictor.
- **Transparent overlay compositing** — instead of redrawing the whole video frame with boxes burned in, an RGBA image with the alpha channel set only where boxes/text exist is layered on top of the live `<video>` element in the browser. This keeps the video smooth since only the (cheap) overlay gets updated each loop, not the (expensive) full video frame.
- **Base64 image round-tripping** — frames go browser → Python as base64 JPEG, and boxes go Python → browser as base64 PNG, since that's the only way to move binary image data through the JS/Python bridge.

## Project structure

```
main.py      # full pipeline: JS webcam bridge, config, frame conversion,
              # face detection, overlay drawing, main loop
README.md    # this file
```

The JavaScript webcam bridge (`video_stream`, `video_frame`) is kept intact since it's inherent boilerplate for Colab's camera access. The Python side is organized into small functions — `js_to_image`, `detect_faces`, `draw_face_overlay`, `overlay_to_data_url` — driven by a `PipelineConfig` dataclass, tied together by `run_pipeline()`.

## Requirements

```
opencv-python
numpy
dlib
```

Plus `IPython` and `google.colab`, which are already available in the Colab environment — no separate install needed there.

Install the rest with:

```bash
pip install opencv-python numpy dlib
```

(`dlib` can be slow to build from source on some systems — a prebuilt wheel is recommended if available for your platform.)

## How to run

1. Open this file in a Google Colab notebook (paste the contents into a cell, or `%run main.py`).
2. Run the cell and grant camera permission when the browser prompts.
3. Click the video (or the red instruction text) to stop the stream.


## Things I'd like to try next

- Swap dlib's HOG detector for a CNN-based one (`dlib.cnn_face_detection_model_v1`) for better accuracy on angled/partially occluded faces, at the cost of speed.
- Add face landmark detection (dlib's 68-point predictor) to draw facial features, not just bounding boxes.
- Track a rolling FPS counter using the `create`/`show`/`capture` timing data already returned by the JS bridge, to see where time is actually being spent per frame.

## References

- [GeeksforGeeks — Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/) — used as a general reference/inspiration while working on this project.


---
*This is a personal learning project, not a production surveillance or biometric system.*
