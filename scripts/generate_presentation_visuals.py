"""Generate slide-ready SVG visuals for the BridgeLink ASL presentation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "vlm_eval_wlasl25_cnn" / "wlasl25_cnn_hybrid_eval.jsonl"
OUTPUT_CANDIDATES = (
    ROOT / "outputs" / "vlm_compare_local_fixed.jsonl",
    ROOT / "outputs" / "vlm_compare_local_retry.jsonl",
    ROOT / "outputs" / "vlm_compare_local.jsonl",
)
OUTPUT_DIR = ROOT / "presentation" / "visuals"


DISPLAY_FONT = "Fraunces, Georgia, Times New Roman, serif"
BODY_FONT = "Aptos, Segoe UI, Helvetica, Arial, sans-serif"
MONO_FONT = "IBM Plex Mono, Cascadia Code, Consolas, monospace"


@dataclass(frozen=True)
class ComparisonMetrics:
    samples: int
    classes: int
    cnn_top1: float
    cnn_top5: float
    vlm_accuracy: float | None
    vlm_failures: int | None
    output_path: str | None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    write_summary_json(metrics)
    write_pipeline_svg(metrics)
    write_scope_svg(metrics)
    write_comparison_svg(metrics)
    write_readme(metrics)
    print(f"Generated presentation visuals in: {OUTPUT_DIR}")


def load_metrics() -> ComparisonMetrics:
    manifest_rows = load_jsonl(MANIFEST_PATH)
    true_labels = [normalize(row.get("true_label")) for row in manifest_rows]
    top1_labels = [normalize(row.get("cnn_top1") or row.get("model_top1")) for row in manifest_rows]
    top5_hits = []
    for row, actual in zip(manifest_rows, true_labels):
        labels = [
            normalize(item.get("label") if isinstance(item, dict) else item)
            for item in (row.get("cnn_top5") or row.get("model_top5") or [])
        ]
        top5_hits.append(actual in labels)

    cnn_top1 = round(sum(a == b for a, b in zip(true_labels, top1_labels)) / len(true_labels), 4)
    cnn_top5 = round(sum(top5_hits) / len(top5_hits), 4)
    classes = len({label for label in true_labels if label})

    vlm_accuracy: float | None = None
    vlm_failures: int | None = None
    chosen_output: Path | None = None
    for candidate in OUTPUT_CANDIDATES:
        if candidate.exists():
            chosen_output = candidate
            break

    if chosen_output is not None:
        rows = load_jsonl(chosen_output)
        if rows:
            vlm_correct = 0
            vlm_failures = 0
            for row in rows:
                expected = normalize(row.get("expected_text"))
                prediction = extract_vlm_label(row)
                if prediction is not None and normalize(prediction) == expected:
                    vlm_correct += 1
                if row.get("failure_notes"):
                    vlm_failures += 1
            vlm_accuracy = round(vlm_correct / len(rows), 4)

    return ComparisonMetrics(
        samples=len(manifest_rows),
        classes=classes,
        cnn_top1=cnn_top1,
        cnn_top5=cnn_top5,
        vlm_accuracy=vlm_accuracy,
        vlm_failures=vlm_failures,
        output_path=str(chosen_output) if chosen_output else None,
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def extract_vlm_label(row: dict[str, object]) -> str | None:
    prediction = row.get("vlm_prediction")
    if isinstance(prediction, dict):
        gloss = prediction.get("gloss")
        if isinstance(gloss, list) and gloss:
            return str(gloss[0])
        sentence = prediction.get("sentence")
        if sentence:
            return str(sentence)
    if prediction:
        return str(prediction)
    return None


def write_summary_json(metrics: ComparisonMetrics) -> None:
    payload = {
        "samples": metrics.samples,
        "classes": metrics.classes,
        "cnn_top1": metrics.cnn_top1,
        "cnn_top5": metrics.cnn_top5,
        "vlm_accuracy": metrics.vlm_accuracy,
        "vlm_failures": metrics.vlm_failures,
        "output_path": metrics.output_path,
    }
    (OUTPUT_DIR / "metrics-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_pipeline_svg(metrics: ComparisonMetrics) -> None:
    cards = [
        (90, 250, 250, 180, "Input", "Live webcam\nor 2-5 second clip", "#FF8A5B"),
        (390, 180, 300, 250, "Per-frame perception", "MediaPipe Holistic\n33 pose + 21 left hand + 21 right hand\n225-D landmark vector per frame", "#F2C94C"),
        (740, 160, 320, 270, "Temporal modeling", "Rolling 32-frame buffer\nTemporal landmark CNN\nWLASL-25 for demo\nWLASL-100 for report experiment", "#2DCEB1"),
        (1110, 180, 320, 250, "Output + comparison", "Top-1 sign caption\nTop-5 candidates\nOptional Qwen2.5-VL reranker\nSpeech + Hugging Face Space UI", "#6AA6FF"),
    ]

    svg = [
        svg_header(1600, 900),
        dark_background(),
        glow(180, 120, 260, "#FF8A5B", 0.22),
        glow(1320, 170, 300, "#6AA6FF", 0.18),
        title_block(
            "BridgeLink ASL Pipeline",
            "The deployed demo uses MediaPipe landmarks plus a trained temporal CNN. "
            "The VLM stays off the live path and acts only as a comparison reranker.",
        ),
        stat_pill(110, 105, "Evaluation set", f"{metrics.samples} clips / {metrics.classes} classes"),
        stat_pill(470, 105, "Live demo model", "WLASL-25 landmark CNN"),
        stat_pill(855, 105, "Comparison model", "Local Qwen2.5-VL reranker"),
    ]

    for x, y, w, h, heading, body, color in cards:
        svg.append(card(x, y, w, h, heading, body, color))

    svg.extend(
        [
            arrow(340, 340, 390, 305, "#F7F1E8"),
            arrow(690, 305, 740, 295, "#F7F1E8"),
            arrow(1060, 295, 1110, 305, "#F7F1E8"),
            note_callout(
                1060,
                560,
                430,
                170,
                "Presentation talking point",
                "Train the CNN ourselves. Use the local VLM only to rerank the CNN's top-5 labels. "
                "That keeps the live demo fast and makes the comparison academically honest.",
                "#111827",
                "#F7F1E8",
            ),
        ]
    )
    svg.append(svg_footer())
    (OUTPUT_DIR / "bridgelink_pipeline.svg").write_text("".join(svg), encoding="utf-8")


def write_scope_svg(metrics: ComparisonMetrics) -> None:
    svg = [
        svg_header(1600, 900),
        light_background(),
        editorial_grid(),
        f'<text x="110" y="118" font-family="{DISPLAY_FONT}" font-size="52" fill="#11203C" font-weight="700">What We Trained vs What We Reused</text>',
        f'<text x="110" y="164" font-family="{BODY_FONT}" font-size="22" fill="#4B587C">A slide-ready scope board for explaining the final project story clearly.</text>',
        panel(105, 220, 650, 560, "Trained by our team", "#FFFBF5", "#D97B37"),
        panel(845, 220, 650, 560, "Used without training", "#F7FAFF", "#457BFF"),
    ]

    trained_items = [
        ("WLASL-100 landmark CNN", "Primary report-scale experiment. Trained on MediaPipe landmark sequences."),
        ("WLASL-25 landmark CNN", "Smaller vocabulary for the live webcam demo and HF Space stability."),
        ("Landmark Transformer", "Attention-based extension. Kept as extra-credit / modern-method evidence."),
    ]
    reused_items = [
        ("MediaPipe Holistic", "Frozen feature extractor that turns each frame into 225 landmark coordinates."),
        ("Qwen2.5-VL local model", "Pretrained VLM used only as a zero-shot reranker over the CNN top-5."),
        ("Gradio + Hugging Face Space", "Presentation UI and deployment layer for the class demo."),
    ]

    for idx, (title, desc) in enumerate(trained_items):
        svg.append(scope_row(145, 280 + idx * 165, title, desc, "#D97B37", "TRAINED"))
    for idx, (title, desc) in enumerate(reused_items):
        svg.append(scope_row(885, 280 + idx * 165, title, desc, "#457BFF", "REUSED"))

    svg.append(
        note_callout(
            110,
            815,
            1385,
            55,
            "One-sentence takeaway",
            "We trained the recognition models ourselves, but we compared them against a frozen local VLM instead of fine-tuning the VLM. "
            "That still satisfies the project scope because the trained CNN is the main model and the VLM is an evaluated comparison method.",
            "#11203C",
            "#F7F1E8",
        )
    )
    svg.append(svg_footer())
    (OUTPUT_DIR / "project_scope_board.svg").write_text("".join(svg), encoding="utf-8")


def write_comparison_svg(metrics: ComparisonMetrics) -> None:
    vlm_value = metrics.vlm_accuracy if metrics.vlm_accuracy is not None else 0.0
    bars = [
        ("CNN top-1", metrics.cnn_top1, "#E86D3D"),
        ("CNN top-5", metrics.cnn_top5, "#3FB7A8"),
        ("Qwen rerank", vlm_value, "#4D7CFE"),
    ]
    max_width = 620
    start_x = 210
    start_y = 280
    row_gap = 150

    svg = [
        svg_header(1600, 900),
        dark_background(),
        glow(1240, 160, 280, "#4D7CFE", 0.22),
        f'<text x="110" y="118" font-family="{DISPLAY_FONT}" font-size="54" fill="#F8F3EA" font-weight="700">CNN vs VLM: What Actually Happened</text>',
        f'<text x="110" y="164" font-family="{BODY_FONT}" font-size="22" fill="#C9D0E0">Held-out WLASL-25 reranking set: {metrics.samples} clips across {metrics.classes} classes.</text>',
        stat_pill_dark(111, 205, "Key message", "Qwen matched CNN top-1, but did not beat it."),
        stat_pill_dark(540, 205, "Failures", f"{metrics.vlm_failures if metrics.vlm_failures is not None else 0} wrapper failures"),
    ]

    for index, (label, value, color) in enumerate(bars):
        y = start_y + index * row_gap
        width = max(18, int(max_width * value))
        svg.extend(
            [
                f'<text x="210" y="{y - 28}" font-family="{BODY_FONT}" font-size="30" fill="#F8F3EA" font-weight="600">{escape(label)}</text>',
                f'<rect x="{start_x}" y="{y}" width="{max_width}" height="44" rx="22" fill="#1D2637" stroke="#2B3347" stroke-width="2"/>',
                f'<rect x="{start_x}" y="{y}" width="{width}" height="44" rx="22" fill="{color}"/>',
                f'<text x="{start_x + max_width + 35}" y="{y + 31}" font-family="{MONO_FONT}" font-size="28" fill="#F8F3EA">{value * 100:.1f}%</text>',
            ]
        )

    svg.append(
        note_callout(
            960,
            325,
            500,
            250,
            "Interpretation",
            "The CNN learned enough to surface the right sign in its candidate list more than half the time. "
            "The local VLM successfully ran end to end, but it did not improve accuracy over the trained CNN baseline on this small evaluation set.",
            "#0F172A",
            "#F8F3EA",
        )
    )
    svg.append(
        mini_table(
            210,
            640,
            [
                ("Metric", "Value"),
                ("Evaluation clips", str(metrics.samples)),
                ("Unique classes", str(metrics.classes)),
                ("CNN top-1", f"{metrics.cnn_top1 * 100:.1f}%"),
                ("CNN top-5", f"{metrics.cnn_top5 * 100:.1f}%"),
                ("Qwen rerank", f"{vlm_value * 100:.1f}%"),
            ],
        )
    )
    svg.append(svg_footer())
    (OUTPUT_DIR / "cnn_vs_vlm_comparison.svg").write_text("".join(svg), encoding="utf-8")


def write_readme(metrics: ComparisonMetrics) -> None:
    text = f"""# Presentation Visuals

Generated slide-ready visuals for BridgeLink ASL.

## Files

- `bridgelink_pipeline.svg`: methodology / system diagram
- `project_scope_board.svg`: what the team trained versus what was reused
- `cnn_vs_vlm_comparison.svg`: final comparison numbers for the presentation
- `metrics-summary.json`: source values used to render the comparison slide

## Current values

- Evaluation set: {metrics.samples} clips
- Unique classes: {metrics.classes}
- CNN top-1: {metrics.cnn_top1 * 100:.1f}%
- CNN top-5: {metrics.cnn_top5 * 100:.1f}%
- Qwen rerank: {(metrics.vlm_accuracy or 0.0) * 100:.1f}%
- VLM wrapper failures: {metrics.vlm_failures if metrics.vlm_failures is not None else "unknown"}

## Regenerate

```powershell
python scripts\\generate_presentation_visuals.py
```
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" fill="none">'
    )


def svg_footer() -> str:
    return "</svg>"


def dark_background() -> str:
    return """
    <defs>
      <linearGradient id="bg-dark" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0C1020"/>
        <stop offset="100%" stop-color="#121A2C"/>
      </linearGradient>
    </defs>
    <rect width="1600" height="900" fill="url(#bg-dark)"/>
    <rect x="42" y="42" width="1516" height="816" rx="34" stroke="#263247" stroke-width="2"/>
    """


def light_background() -> str:
    return """
    <defs>
      <linearGradient id="bg-light" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#FFF8EE"/>
        <stop offset="100%" stop-color="#F4F7FB"/>
      </linearGradient>
    </defs>
    <rect width="1600" height="900" fill="url(#bg-light)"/>
    """


def editorial_grid() -> str:
    lines = []
    for x in range(80, 1600, 140):
        lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="900" stroke="#E7E1D5" stroke-opacity="0.55" />')
    for y in range(120, 900, 120):
        lines.append(f'<line x1="0" y1="{y}" x2="1600" y2="{y}" stroke="#E7E1D5" stroke-opacity="0.35" />')
    return "".join(lines)


def glow(cx: int, cy: int, radius: int, color: str, opacity: float) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" opacity="{opacity}" />'
    )


def title_block(title: str, subtitle: str) -> str:
    return (
        f'<text x="110" y="118" font-family="{DISPLAY_FONT}" font-size="56" fill="#F8F3EA" font-weight="700">{escape(title)}</text>'
        f'<text x="110" y="166" font-family="{BODY_FONT}" font-size="22" fill="#C9D0E0">{escape(subtitle)}</text>'
    )


def stat_pill(x: int, y: int, label: str, value: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="310" height="62" rx="31" fill="#162133" stroke="#2B3850" stroke-width="2"/>'
        f'<text x="{x + 24}" y="{y + 25}" font-family="{BODY_FONT}" font-size="15" fill="#9AB0D0" font-weight="600">{escape(label.upper())}</text>'
        f'<text x="{x + 24}" y="{y + 47}" font-family="{BODY_FONT}" font-size="22" fill="#F8F3EA" font-weight="600">{escape(value)}</text>'
    )


def stat_pill_dark(x: int, y: int, label: str, value: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="390" height="64" rx="32" fill="#121B2B" stroke="#2D3B56" stroke-width="2"/>'
        f'<text x="{x + 24}" y="{y + 24}" font-family="{BODY_FONT}" font-size="15" fill="#AAB7D2" font-weight="600">{escape(label.upper())}</text>'
        f'<text x="{x + 24}" y="{y + 47}" font-family="{BODY_FONT}" font-size="22" fill="#F8F3EA" font-weight="600">{escape(value)}</text>'
    )


def card(x: int, y: int, w: int, h: int, heading: str, body: str, accent: str) -> str:
    body_lines = body.split("\n")
    text = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="#11192A" stroke="#2A3650" stroke-width="2"/>',
        f'<rect x="{x + 24}" y="{y + 24}" width="72" height="8" rx="4" fill="{accent}"/>',
        f'<text x="{x + 24}" y="{y + 72}" font-family="{DISPLAY_FONT}" font-size="34" fill="#F8F3EA" font-weight="700">{escape(heading)}</text>',
    ]
    for idx, line in enumerate(body_lines):
        text.append(
            f'<text x="{x + 24}" y="{y + 118 + idx * 34}" font-family="{BODY_FONT}" font-size="24" fill="#D8DDE8">{escape(line)}</text>'
        )
    return "".join(text)


def arrow(x1: int, y1: int, x2: int, y2: int, color: str) -> str:
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_x = x2 - 18 * math.cos(angle)
    arrow_y = y2 - 18 * math.sin(angle)
    wing = 12
    left_x = arrow_x - wing * math.cos(angle - math.pi / 2)
    left_y = arrow_y - wing * math.sin(angle - math.pi / 2)
    right_x = arrow_x - wing * math.cos(angle + math.pi / 2)
    right_y = arrow_y - wing * math.sin(angle + math.pi / 2)
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
        f'<polygon points="{x2},{y2} {left_x},{left_y} {right_x},{right_y}" fill="{color}"/>'
    )


def note_callout(x: int, y: int, w: int, h: int, title: str, text: str, bg: str, fg: str) -> str:
    lines = wrap_lines(text, 55)
    nodes = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{bg}" opacity="0.94"/>',
        f'<text x="{x + 28}" y="{y + 42}" font-family="{BODY_FONT}" font-size="17" fill="{fg}" font-weight="700">{escape(title.upper())}</text>',
    ]
    for idx, line in enumerate(lines[:4]):
        nodes.append(
            f'<text x="{x + 28}" y="{y + 82 + idx * 30}" font-family="{BODY_FONT}" font-size="22" fill="{fg}">{escape(line)}</text>'
        )
    return "".join(nodes)


def panel(x: int, y: int, w: int, h: int, title: str, fill: str, accent: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="30" fill="{fill}" stroke="{accent}" stroke-width="3"/>'
        f'<text x="{x + 36}" y="{y + 64}" font-family="{DISPLAY_FONT}" font-size="42" fill="#11203C" font-weight="700">{escape(title)}</text>'
        f'<rect x="{x + 36}" y="{y + 84}" width="120" height="8" rx="4" fill="{accent}"/>'
    )


def scope_row(x: int, y: int, title: str, desc: str, accent: str, badge: str) -> str:
    lines = wrap_lines(desc, 42)
    parts = [
        f'<rect x="{x}" y="{y}" width="570" height="118" rx="22" fill="#FFFFFF" stroke="#E7EAF2" stroke-width="2"/>',
        f'<rect x="{x + 26}" y="{y + 28}" width="94" height="28" rx="14" fill="{accent}"/>',
        f'<text x="{x + 42}" y="{y + 48}" font-family="{MONO_FONT}" font-size="14" fill="#FFFFFF" font-weight="700">{escape(badge)}</text>',
        f'<text x="{x + 142}" y="{y + 46}" font-family="{BODY_FONT}" font-size="28" fill="#11203C" font-weight="700">{escape(title)}</text>',
    ]
    for idx, line in enumerate(lines[:2]):
        parts.append(
            f'<text x="{x + 142}" y="{y + 78 + idx * 26}" font-family="{BODY_FONT}" font-size="20" fill="#4B587C">{escape(line)}</text>'
        )
    return "".join(parts)


def mini_table(x: int, y: int, rows: list[tuple[str, str]]) -> str:
    row_height = 42
    width = 530
    height = row_height * len(rows) + 24
    out = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="22" fill="#141D2F" stroke="#2A3650" stroke-width="2"/>'
    ]
    for idx, (left, right) in enumerate(rows):
        row_y = y + 28 + idx * row_height
        if idx == 0:
            out.append(f'<rect x="{x + 14}" y="{row_y - 24}" width="{width - 28}" height="{row_height}" rx="14" fill="#1E2A40"/>')
        out.append(
            f'<text x="{x + 28}" y="{row_y}" font-family="{BODY_FONT}" font-size="20" fill="#F8F3EA" font-weight="{"700" if idx == 0 else "500"}">{escape(left)}</text>'
        )
        out.append(
            f'<text x="{x + width - 28}" y="{row_y}" text-anchor="end" font-family="{MONO_FONT}" font-size="20" fill="#F8F3EA" font-weight="{"700" if idx == 0 else "500"}">{escape(right)}</text>'
        )
    return "".join(out)


def wrap_lines(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        tentative = " ".join(current + [word])
        if current and len(tentative) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    main()
