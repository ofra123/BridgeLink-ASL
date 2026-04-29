"""Generate report artifacts for the imported How2Sign VLM experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VLM_ARCHIVE = (
    PROJECT_ROOT
    / "vlm_hf_space"
    / "outputs"
    / "how2sign_qwen25vl_3b_qlora"
    / "archive_500train_eval100"
)
REPORT_FIGURES = PROJECT_ROOT / "report" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = load_json(VLM_ARCHIVE / "baseline_metrics.json")
    finetuned = load_json(VLM_ARCHIVE / "finetuned_metrics.json")

    metrics = ("bleu", "chrf", "rouge_l")
    labels = ("BLEU", "chrF", "ROUGE-L")
    baseline_values = [float(baseline[name]) for name in metrics]
    finetuned_values = [float(finetuned[name]) for name in metrics]
    deltas = [after - before for before, after in zip(baseline_values, finetuned_values)]

    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_positions = range(len(labels))
    width = 0.34

    ax.bar(
        [x - width / 2 for x in x_positions],
        baseline_values,
        width=width,
        label="Base Qwen2.5-VL-3B",
        color="#5E81AC",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        finetuned_values,
        width=width,
        label="QLoRA fine-tuned",
        color="#D08770",
    )

    for index, delta in enumerate(deltas):
        ax.text(
            index,
            max(baseline_values[index], finetuned_values[index]) + 0.35,
            f"+{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2E3440",
        )

    ax.set_title("How2Sign VLM Translation Metrics", fontsize=14, pad=12)
    ax.set_ylabel("Score")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    figure_path = REPORT_FIGURES / "vlm_how2sign_metrics.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source_dir": str(VLM_ARCHIVE),
        "num_samples": int(finetuned["num_samples"]),
        "baseline": baseline,
        "finetuned": finetuned,
        "delta": {
            name: round(after - before, 6)
            for name, before, after in zip(metrics, baseline_values, finetuned_values)
        },
        "figure_path": str(figure_path),
    }
    summary_path = RESULTS_DIR / "vlm_how2sign_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote figure to {figure_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
