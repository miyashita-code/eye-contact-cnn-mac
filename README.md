# Eye Contact CLI

Terminal-first fork of the original `eye-contact-cnn` demo. The GUI overlay has been removed so the project focuses on lightweight, scriptable inference that prints eye-contact scores straight to stdout.

## Features
- ResNet-based eye-contact classifier (`model_static`) with pretrained weights in `data/model_weights.pkl`.
- Flexible face detection workflow: dlib CNN detector when available, or OpenCV Haar cascades as a fallback. Precomputed detections can be supplied via `--face`.
- Pure CLI experience: per-frame scores stream to the terminal; optional `--quiet` switch suppresses noisy logs.
- macOS-friendly: automatically uses CUDA, Metal (MPS), or CPU depending on what PyTorch reports.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision opencv-python numpy Pillow pandas
python demo.py --video /path/to/clip.avi
```

> **Note:** dlib is optional but required for the CNN face detector. If you prefer OpenCV Haar cascades or precomputed detections, you can skip installing it.

## Usage

- Live webcam (requires dlib or Haar cascades):
  ```bash
  python demo.py
  ```
- Process a video file with automatic face detection:
  ```bash
  python demo.py --video /path/to/clip.avi
  ```
- Process a video file with precomputed face boxes (CSV formatted as `[frame,left,top,right,bottom]`):
  ```bash
  python demo.py --video /path/to/clip.avi --face detections.csv
  ```

The script prints a line per processed frame such as:

```
[path/to/clip.avi] frame 128: scores=0.912, 0.337
```

Use `--quiet` to disable per-frame logging when you only need the completion message.

### CLI Flags

- `--video`: Input video path. Default is `None`, which opens the first webcam.
- `--face`: CSV with precomputed face detections.
- `--model_weight`: Path to the model checkpoint (default `data/model_weights.pkl`).
- `--jitter`: Integer specifying how many additional jittered crops to average per face.
- `--max_frames`: Limit how many frames are processed (useful for smoke tests).
- `--quiet`: Suppress per-frame logs.

### Face Detection Notes

- **dlib CNN**: Requires `data/mmod_human_face_detector.dat` (bundled). Install `dlib` and the detector will be used automatically.
- **OpenCV Haar**: No external dependencies beyond OpenCV itself. Keep the OpenCV data folder accessible (macOS installs include it).
- **Precomputed**: Provide a CSV via `--face`. Bounding boxes are automatically expanded to stabilize scores.

## Environment Tips

- Keep the virtual environment directory out of Git history by adding `.venv/` to `.gitignore` if you plan to commit.
- To format the codebase, install Black (`python -m pip install black`) and run `python -m black demo.py model.py`.
- The pretrained weights are ~90 MB. Budget ~400 MB of RAM for CPU inference runs on macOS.

## Forking / Sharing

1. Create a new GitHub repository or fork under your account.
2. Add it as a remote: `git remote add myfork git@github.com:<you>/eye-contact-cli.git`
3. Push your branch: `git push myfork HEAD:main`
4. Document any dependency or environment differences in your fork’s README and PR descriptions as needed.

Please cite the original authors if you use this project in research or production work.
