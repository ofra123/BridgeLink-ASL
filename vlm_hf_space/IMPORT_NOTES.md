# VLM Space Import Notes

This folder is an imported snapshot of the teammate Hugging Face Space repo:

`https://huggingface.co/spaces/ofraij123/ASL-Video-To-Sentence-Translation`

It was copied into the main `BridgeLink-ASL` repository on 2026-04-29 so the
final GitLab submission contains the full VLM-side project materials instead of
splitting the work across separate repos.

What is preserved here:

- Qwen2.5-VL QLoRA training scripts
- How2Sign / ASL Citizen data-prep scripts
- experiment configs
- archived baseline vs fine-tuned metrics
- sample prediction outputs
- Docker/requirements files from the original Space

Important note about the prepared JSONL manifests:

- `data/how2sign_train.jsonl` and `data/how2sign_val.jsonl` contain the paths
  used in the teammate training environment.
- Those paths are valid as experiment records, but they may need to be
  regenerated with the local scripts if training is rerun on another machine.
