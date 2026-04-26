# Local Machine Setup

This guide is the fastest way to run BridgeLink ASL on a fresh Windows machine
for the class demo, local testing, or VLM comparison work.

## 1. Install system prerequisites

### Python

Use **Python 3.11**. MediaPipe has been much more reliable there than on newer
Python versions.

Check:

```powershell
py -3.11 --version
```

If needed, install with:

```powershell
winget install Python.Python.3.11
```

### FFmpeg

Required for Gradio upload / record clip mode.

Check:

```powershell
ffmpeg -version
```

If needed, install with:

```powershell
winget install --id Gyan.FFmpeg -e
```

Restart the terminal after installing FFmpeg.

## 2. Clone the repo and enter it

```powershell
git clone https://github.com/ofra123/BridgeLink-ASL.git
cd BridgeLink-ASL
```

## 3. Create the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4. Install project dependencies

### For the CNN demo app and tests

```powershell
pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
```

### For the local Qwen VLM comparison

```powershell
python -m pip install -e ".[vlm]"
python -m pip install torchvision
```

## 5. Put the trained model files in `models/`

The local app expects these files in the repo `models/` directory:

```text
models/cnn_landmark_best.pt
models/cnn_landmark_wlasl25_best.pt
models/sign_transformer_best.pt
models/labels.json
```

Recommended usage:

- `cnn_landmark_wlasl25_best.pt`: live demo / Hugging Face Space
- `cnn_landmark_best.pt`: main WLASL-100 report model
- `sign_transformer_best.pt`: optional attention-based extension

## 6. Run the live demo locally

Use the smaller WLASL-25 model for the most stable demo:

```powershell
$env:HF_MODEL_FILENAME="cnn_landmark_wlasl25_best.pt"
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The app supports:

- **Live Webcam**
- **Upload / Record Clip**

## 7. Run the local VLM comparison

The hybrid evaluation set is already committed here:

```text
data/vlm_eval_wlasl25_cnn/wlasl25_cnn_hybrid_eval.jsonl
```

Run the local Qwen reranker:

```powershell
$env:BRIDGELINK_VLM_PROVIDER="local"
$env:BRIDGELINK_VLM_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

run_wrapper --mode compare --manifest data\vlm_eval_wlasl25_cnn\wlasl25_cnn_hybrid_eval.jsonl --output outputs\vlm_compare_local_fixed.jsonl
```

Current known result on the committed 36-clip eval set:

```text
CNN top-1: 25.0%
CNN top-5: 58.3%
Qwen rerank: 25.0%
```

## 8. Regenerate presentation visuals

```powershell
python scripts\generate_presentation_visuals.py
```

Outputs go to:

```text
presentation/visuals/
```

## 9. Run the test suite

```powershell
python -m pytest -q
```

At the current tested state, this should pass with:

```text
33 passed, 2 warnings
```

## 10. Common troubleshooting

### `mediapipe has no attribute solutions`

You are probably using the wrong Python version or the wrong environment.
Use Python 3.11 and reinstall into `.venv`.

### Webcam works but the caption buffer stays at `0/32`

Check that:

- camera permission is allowed in the browser
- the app is running at `http://127.0.0.1:7860`
- no other app is locking the webcam

### Upload / record clip throws `ffmpeg not found`

Install FFmpeg and restart the terminal:

```powershell
winget install --id Gyan.FFmpeg -e
```

### VLM run says `Torchvision library was not found`

Install torchvision into `.venv`:

```powershell
python -m pip install torchvision
```

### VLM run is very slow

That is expected on CPU. The 7B model may offload to CPU/disk and take a long
time. Keep the VLM for local comparison, not the live Space demo.

## 11. Recommended local demo flow

1. Activate `.venv`
2. Set `HF_MODEL_FILENAME=cnn_landmark_wlasl25_best.pt`
3. Run `python app.py`
4. Test `computer`, `bed`, `help`, `drink`, `yes`
5. Keep a backup recorded demo clip ready
