# Demo Checklist

Use this checklist on the presentation machine before class.

## 1. Environment

- Python 3.11 installed
- FFmpeg installed and available from the terminal
- `.venv` activated
- dependencies installed from `requirements.txt`
- `cnn_landmark_wlasl25_best.pt` present in `models/`
- `labels.json` present in `models/`

## 2. Launch The Local App

```powershell
$env:HF_MODEL_FILENAME="cnn_landmark_wlasl25_best.pt"
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## 3. Live Demo Checks

- browser camera permission granted
- webcam image appears immediately
- tracking overlay shows pose and hands
- buffer reaches `32/32`
- caption updates after a stable sign
- room lighting is consistent
- upper body stays centered in frame

## 4. Recommended Signs

Start with the most stable demo signs:

- `computer`
- `bed`
- `help`
- `drink`
- `yes`

Reset the caption between attempts and hold each sign for 3-5 seconds.

## 5. Upload / Record Clip Checks

- recorded clip preview appears
- clip inference returns top-5 predictions
- no FFmpeg errors appear in the terminal

## 6. Presentation Visuals

Make sure these files are ready in `presentation/visuals/`:

- `bridgelink_pipeline.svg`
- `project_scope_board.svg`
- `cnn_vs_vlm_comparison.svg`

## 7. VLM Comparison Talking Point

Use this exact honest summary if asked:

- the CNN is the main trained model
- the local Qwen VLM is a zero-shot reranker over the CNN top-5
- on the 36-clip eval set:
  - CNN top-1 = 25.0%
  - CNN top-5 = 58.3%
  - Qwen rerank = 25.0%

## 8. Backup Plan

If the live webcam becomes unstable:

1. switch to the upload / record clip tab
2. use a prerecorded successful clip
3. show the tracking overlay screenshot and results slide
4. explain that the local demo model is WLASL-25 for stability

If the app fails entirely:

1. show the generated presentation visuals
2. show the trained metrics and comparison numbers
3. explain the pipeline using the methodology diagram
