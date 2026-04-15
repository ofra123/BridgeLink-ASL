# Model Weights

Place the trained model files here after running the Colab training notebook.

## Required files

| File | Source | Description |
|---|---|---|
| `sign_transformer_best.pt` | Google Drive → BridgeLink-ASL → models/ | Trained Transformer weights + label map + config (~5 MB) |
| `labels.json` | Google Drive → BridgeLink-ASL → models/ | Label index → gloss name mapping |

## How to get them

1. Open Google Drive → `BridgeLink-ASL/models/`
2. Download `sign_transformer_best.pt` and `labels.json`
3. Place both files in this `models/` directory

The app loads `sign_transformer_best.pt` at startup. If the file is missing,
the app will start but show a "Model: NOT LOADED" banner and predictions
won't work.

## Alternative: Hugging Face Hub

Instead of committing the `.pt` file to git, you can upload it to a HF model
repo and set `HF_MODEL_REPO=your-username/repo-name` as an environment
variable. The app will auto-download from the Hub at startup.
