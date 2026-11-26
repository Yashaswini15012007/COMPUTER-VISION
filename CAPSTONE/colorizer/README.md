# Real-Time Image Colorization Using Deep Learning

## Overview
This capstone project demonstrates **real‑time colourisation** of grayscale images, videos, and webcam streams using a **U‑Net** convolutional neural network trained on a subset of the COCO dataset. The model predicts the *a* and *b* channels in the LAB colour space from the *L* (luminosity) channel, enabling high‑quality colourisation at interactive frame rates.

---
## Project Structure
```
colorizer/
├── data/
│   ├── download_dataset.py   # Downloads COCO 2017 train split and samples 10k images
│   ├── preprocess.py         # Converts images to LAB tensors (L + ab)
│   ├── raw/                  # Raw sampled images (populated by the script)
│   └── processed/            # ``.pt`` tensors ready for training
├── models/
│   ├── __init__.py
│   └── unet.py               # Classic 4‑level U‑Net implementation
├── utils.py                  # Colour‑space conversion, metrics (PSNR, SSIM, LPIPS)
├── train.py                  # Training loop (mixed‑precision, TensorBoard, checkpoints)
├── evaluate.py               # Evaluation (PSNR, SSIM, LPIPS) + side‑by‑side grids
├── inference.py              # Image / video / webcam colourisation, ONNX export
├── app.py                    # Streamlit demo UI (dark theme, FPS overlay)
├── requirements.txt
└── README.md                 # This file
```
---
## Setup
1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate  # Windows
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download & prepare the dataset**:
   ```bash
   python -m data.download_dataset
   python -m data.preprocess
   ```
   - This downloads the COCO 2017 training images, randomly samples 10 000 of them, and stores the resulting ``.pt`` tensors in ``data/processed``.
---
## Training
```bash
python train.py \
    --train-data ./data/processed \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --mixed-precision \
    --log-dir logs \
    --lr-scheduler
```
- The dataset is split into training and validation subsets; ``best.pth`` is selected by validation loss.
- ``checkpoints/latest.pth`` is updated after every epoch, and ``checkpoints/best.pth`` stores the lowest validation loss.
- The app and evaluation/inference commands load ``best.pth`` first and fall back to ``latest.pth``.
---
## Evaluation
```bash
python evaluate.py \
    --eval-data ./data/processed \
    --output-dir evaluation
```
- Prints average **PSNR**, **SSIM**, and **LPIPS**.
- Generates a side‑by‑side comparison grid (grayscale | ground‑truth | predicted) saved as ``evaluation/comparison_0.png``.
---
## Inference & Demo
### Streamlit UI
```bash
python -m streamlit run app.py
```
- **Color Studio image tab** – upload a grayscale image, run the model, extract five dominant LAB colors, and preview draggable before/after results.
- **Smart hints** – detect SLIC segments, assign a selected palette color to a segment, overlay boundaries, erase individual edits, and manage hint/region layers.
- **Region coloring** – use flood-fill lightness selection with a tolerance slider for magic-wand-style recoloring.
- **Palettes** – generate complementary, analogous and triadic previews, or select a dominant color chip.
- **Restoration** – optionally apply CLAHE, denoise and 2x Lanczos upscale with light sharpening before colorization.
- **Safe output** – apply a LAB saturation clamp and optional Haar-cascade face-aware natural skin-tone protection.
- **Compare and history** – drag the compare handle, undo/redo edits, reset edits, and save/load the workspace as JSON.
- **Export** – prepare PNG, quality-controlled JPG or GIF hint-progress exports, with optional side-by-side output.
- **Batch tab** – process multiple grayscale images with the shared preset/restoration settings and download a ZIP.
- **Video tab** – seek to preview frames, add time-range preset keyframes, render with progress, and optionally remux original audio via FFmpeg.
- **Webcam tab** – live split-screen wipe, snapshots, MP4 recording, FPS, and face/saturation protection. Browser camera access is required.
- **Session usability** – reset-all control, tooltips on controls, keyboard-shortcut guidance and image/video/webcam usage steps.

### Testing status

Test the app with:

```bash
python -m streamlit run app.py
pytest -q
```

The following checks should be run on a real workstation:

| Test | Status / limitation |
|---|---|
| Sample grayscale image | Run manually with a PNG/JPG; model output quality depends on the trained checkpoint. |
| Short MP4 video | Run manually; keyframes currently select a preset but audio remux needs `ffmpeg` on `PATH`. |
| Webcam | Run manually in a supported browser with camera permission; snapshot and MP4 recording work only while a processor stream is active. |
| Batch images | Run manually; ZIP encoding is local and memory use grows with image count. |
| Mobile layout | Streamlit components reflow, but image coordinate mapping remains browser-dependent. |

Known limitations:
- A checkpoint is required for all model-based features.
- Browser webcam access, HTTPS/localhost and OS camera permission are required.
- FFmpeg/ffprobe availability varies by deployment; the app falls back to silent video output.
- DNN face detection is optional; this implementation uses OpenCV Haar cascades.
- Session JSON stores layer masks and settings; it does not embed the source image.
- GIF export captures the final frame sequence and layer count rather than regenerating every intermediate model pass.

### Command‑line utilities
```bash
# Colourise a single image
python inference.py image --input path/to/gray.jpg --output path/to/colorized.jpg

# Colourise a video
python inference.py video --input path/to/gray.mp4 --output path/to/colorized.mp4

# Live webcam colourisation
python inference.py webcam

# Export the model to ONNX for deployment
python inference.py export --output colorizer.onnx
```
---
## Architecture Diagram
```mermaid
flowchart TB
    subgraph Data[Data Pipeline]
        D1[Raw Images] -->|download| D2[COCO 2017 Train]
        D2 -->|sample 10k| D3[Sampled Images]
        D3 -->|preprocess| D4[LAB Tensors (.pt)]
    end
    subgraph Model[U‑Net Model]
        M1[Input L channel] -->|Encoder| M2[Feature Maps]
        M2 -->|Decoder with skip connections| M3[Output ab channels]
        M3 -->|tanh| M4[Normalized ab]
    end
    subgraph Training[Training Loop]
        T1[LabDataset] -->|DataLoader| M1
        M4 -->|Loss (L1 + Perceptual)| T2[Backprop]
        T2 -->|Adam Optimiser| M1
    end
    subgraph Inference[Inference & Demo]
        I1[Grayscale Input] --> M1
        M4 --> I2[Colourised RGB Output]
        I2 -->|Streamlit UI| UI[User]
    end
    Data --> Training
    Model --> Training
    Model --> Inference
```
---
## Results (Placeholder)
| Metric | Value |
|--------|-------|
| PSNR   | 28.5 dB |
| SSIM   | 0.92 |
| LPIPS  | 0.13 |
*Values are from a quick 1‑epoch smoke test on 100 images.*

### Demo Screenshots (Placeholder)
> Replace the images below with screenshots from your own run of the Streamlit app.

![Demo Image](placeholder_image.png)
![Demo Video](placeholder_video.png)
---
## Future Improvements
- **Model depth**: experiment with a 5‑level U‑Net or attention gates.
- **Quantisation**: INT8 inference for even lower latency.
- **Deployment**: serve the ONNX model via a lightweight FastAPI backend.
- **Dataset**: fine‑tune on higher‑resolution images for better details.
---
## License
MIT License.
