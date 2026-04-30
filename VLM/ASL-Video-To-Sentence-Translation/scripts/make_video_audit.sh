#!/usr/bin/env bash
set -euo pipefail

JSONL="${1:-data/how2sign_val.jsonl}"
OUT_DIR="${2:-outputs/video_audit}"
N="${3:-5}"

mkdir -p "${OUT_DIR}"

python - "$JSONL" "$OUT_DIR" "$N" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
out = Path(sys.argv[2])
n = int(sys.argv[3])

out.mkdir(parents=True, exist_ok=True)

if not jsonl.exists():
    raise FileNotFoundError(f"Missing JSONL: {jsonl}")

examples = []
with jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        if len(examples) >= n:
            break
        line = line.strip()
        if line:
            examples.append(json.loads(line))

for i, ex in enumerate(examples):
    video = ex["video_path"]
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(ex.get("id", i)))
    img = out / f"val_{i}_{safe_id}.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-vf", "fps=1,scale=320:-1,tile=4x4",
        "-frames:v", "1",
        str(img),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"FAILED: {video}")
        print(result.stderr[-1000:])
        continue

    print(f"{img} <- {video}")
PY

echo "Wrote audit images to ${OUT_DIR}"
ls -lh "${OUT_DIR}"