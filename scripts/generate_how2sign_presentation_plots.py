"""Generate slide-ready matplotlib graphics for the How2Sign sentence CNN work."""

from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from bridgelink_asl.clip_dataset import load_clip_dataset, summarize_clip_splits
from bridgelink_asl.cnn import CnnModelConfig, _build_tf_dataset, _require_tensorflow


TRANSLATION_DIR = PROJECT_ROOT / "data" / "raw" / "how2sign" / "translations"
CLIP_DIR = PROJECT_ROOT / "data" / "raw" / "how2sign" / "clips" / "raw_videos"
OUTPUT_DIR = PROJECT_ROOT / "presentation" / "visuals"

TOP12_MANIFEST = PROJECT_ROOT / "data" / "processed" / "how2sign_sentences_top12.frames.jsonl"
TOP25_MANIFEST = PROJECT_ROOT / "data" / "processed" / "how2sign_sentences_top25.frames.jsonl"
TOP25_NORMALIZED_MANIFEST = PROJECT_ROOT / "data" / "processed" / "how2sign_sentences_top25.normalized.frames.jsonl"
TOP12_MODEL = PROJECT_ROOT / "models" / "cnn-3d-sentence.keras"
TOP25_MODEL = PROJECT_ROOT / "models" / "cnn-3d-sentence-top25.keras"
TOP25_NORMALIZED_MODEL = PROJECT_ROOT / "models" / "cnn-3d-sentence-top25-normalized.keras"
TOP25_E30_MODEL = PROJECT_ROOT / "models" / "cnn-3d-sentence-top25-e30.keras"
TOP25_WEIGHTED_MODEL = PROJECT_ROOT / "models" / "cnn-3d-sentence-top25-weighted.keras"
TOP25_NORMALIZATION_SUMMARY = PROJECT_ROOT / "results" / "how2sign_top25_normalization_summary.json"

BACKGROUND = "#F7F4EE"
TEXT = "#18222F"
MUTED = "#5D6B7A"
ACCENT = "#1E88E5"
ACCENT_ALT = "#F05A28"
ACCENT_SOFT = "#6CC3A0"
GRID = "#D8D2C6"


@dataclass(frozen=True)
class ThresholdStat:
    label: str
    clip_count: int
    class_count: int


@dataclass(frozen=True)
class BenchmarkStat:
    name: str
    total_clips: int
    num_classes: int
    train_clips: int
    val_clips: int
    test_clips: int
    val_accuracy: float
    test_accuracy: float
    val_loss: float
    test_loss: float


@dataclass(frozen=True)
class ExperimentStat:
    name: str
    training_change: str
    val_accuracy: float
    test_accuracy: float
    test_loss: float


BASE_THRESHOLD_STATS = [
    ThresholdStat(label="All matched clips", clip_count=31047, class_count=30008),
    ThresholdStat(label="Repeated >= 2", clip_count=1487, class_count=448),
    ThresholdStat(label="Repeated >= 3", clip_count=769, class_count=89),
    ThresholdStat(label="Repeated >= 5", clip_count=611, class_count=41),
    ThresholdStat(label="Repeated >= 8", clip_count=531, class_count=26),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    threshold_stats = collect_threshold_stats()
    top25_summary = load_summary(TOP25_NORMALIZATION_SUMMARY)
    benchmark_stats = [
        build_benchmark_stat("Top-12 subset", TOP12_MANIFEST, TOP12_MODEL),
        build_benchmark_stat("Top-25 subset", TOP25_MANIFEST, TOP25_MODEL),
        build_benchmark_stat("Top-25 normalized", TOP25_NORMALIZED_MANIFEST, TOP25_NORMALIZED_MODEL),
    ]
    experiment_stats = collect_experiment_stats()

    write_summary_json(threshold_stats, benchmark_stats, experiment_stats)
    plot_dataset_constraint(threshold_stats)
    plot_subset_benchmark(benchmark_stats)
    plot_top25_class_distribution(top25_summary)
    plot_top25_experiment_comparison(experiment_stats)
    print(f"Generated matplotlib presentation plots in: {OUTPUT_DIR}")


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "font.size": 11,
            "savefig.facecolor": BACKGROUND,
            "savefig.bbox": "tight",
        }
    )


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_threshold_stats() -> list[ThresholdStat]:
    stats = list(BASE_THRESHOLD_STATS)
    top12_summary = summarize_manifest(TOP12_MANIFEST)
    top25_summary = summarize_manifest(TOP25_MANIFEST)
    top25_normalized_summary = summarize_manifest(TOP25_NORMALIZED_MANIFEST)
    stats.extend(
        [
            ThresholdStat(
                label="Top-12 subset",
                clip_count=int(top12_summary["total_clips"]),
                class_count=int(top12_summary["num_classes"]),
            ),
            ThresholdStat(
                label="Top-25 subset",
                clip_count=int(top25_summary["total_clips"]),
                class_count=int(top25_summary["num_classes"]),
            ),
            ThresholdStat(
                label="Top-25 normalized",
                clip_count=int(top25_normalized_summary["total_clips"]),
                class_count=int(top25_normalized_summary["num_classes"]),
            ),
        ]
    )
    return stats


def summarize_manifest(manifest_path: Path) -> dict[str, object]:
    records = load_clip_dataset(manifest_path)
    split_counts = summarize_clip_splits(records)
    class_counts = Counter(record.label.lower() for record in records)
    return {
        "total_clips": len(records),
        "num_classes": len(class_counts),
        "split_counts": split_counts,
        "class_counts": dict(sorted(class_counts.items())),
    }


def build_benchmark_stat(name: str, manifest_path: Path, model_path: Path) -> BenchmarkStat:
    summary = summarize_manifest(manifest_path)
    split_counts = summary["split_counts"]
    val_loss, val_accuracy = evaluate_saved_clip_cnn(manifest_path, model_path, split="val")
    test_loss, test_accuracy = evaluate_saved_clip_cnn(manifest_path, model_path, split="test")
    return BenchmarkStat(
        name=name,
        total_clips=int(summary["total_clips"]),
        num_classes=int(summary["num_classes"]),
        train_clips=int(split_counts["train"]),
        val_clips=int(split_counts["val"]),
        test_clips=int(split_counts["test"]),
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        val_loss=val_loss,
        test_loss=test_loss,
    )


def collect_experiment_stats() -> list[ExperimentStat]:
    experiments = [
        ("Top-25 best", "baseline checkpoint", TOP25_MODEL),
        ("Top-25 normalized", "merged duplicate labels", TOP25_NORMALIZED_MODEL),
        ("Top-25 continued", "18 more epochs", TOP25_E30_MODEL),
        ("Top-25 weighted", "balanced class weights", TOP25_WEIGHTED_MODEL),
    ]
    stats: list[ExperimentStat] = []
    for name, training_change, model_path in experiments:
        manifest_path = TOP25_NORMALIZED_MANIFEST if model_path == TOP25_NORMALIZED_MODEL else TOP25_MANIFEST
        val_loss, val_accuracy = evaluate_saved_clip_cnn(manifest_path, model_path, split="val")
        test_loss, test_accuracy = evaluate_saved_clip_cnn(manifest_path, model_path, split="test")
        stats.append(
            ExperimentStat(
                name=name,
                training_change=training_change,
                val_accuracy=val_accuracy,
                test_accuracy=test_accuracy,
                test_loss=test_loss,
            )
        )
    return stats


def evaluate_saved_clip_cnn(manifest_path: Path, model_path: Path, *, split: str) -> tuple[float, float]:
    tf = _require_tensorflow()
    records = load_clip_dataset(manifest_path)
    labels = tuple(sorted({record.label for record in records}))
    label_to_index = {label: index for index, label in enumerate(labels)}
    selected_records = [record for record in records if record.split == split]
    if not selected_records:
        raise ValueError(f"Manifest {manifest_path} does not contain split '{split}'.")

    model = tf.keras.models.load_model(model_path)
    _, frame_count, image_size, _, channels = model.input_shape
    dataset = _build_tf_dataset(
        tf,
        selected_records,
        label_to_index,
        CnnModelConfig(
            frame_count=int(frame_count),
            image_size=int(image_size),
            channels=int(channels),
        ),
        shuffle=False,
    )
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return float(loss), float(accuracy)


def plot_dataset_constraint(stats: list[ThresholdStat]) -> None:
    labels = [stat.label for stat in stats]
    clip_counts = [stat.clip_count for stat in stats]
    class_counts = [stat.class_count for stat in stats]
    colors = [ACCENT, "#4E79A7", "#59A14F", "#F28E2B", "#E15759", ACCENT_SOFT, ACCENT_ALT, "#A06CD5"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(
        "Why We Cannot Use All 31k How2Sign Clips as Sentence Classes",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )

    clip_ax, class_ax = axes
    clip_ax.bar(labels, clip_counts, color=colors, edgecolor="white", linewidth=1.2)
    clip_ax.set_yscale("log")
    clip_ax.set_ylabel("Clip count (log scale)")
    clip_ax.set_title("Clip volume after repeated-sentence filtering")
    clip_ax.tick_params(axis="x", rotation=35)
    annotate_bars(clip_ax, clip_counts, clip_ax.get_yscale(), percent=False)

    class_ax.bar(labels, class_counts, color=colors, edgecolor="white", linewidth=1.2)
    class_ax.set_yscale("log")
    class_ax.set_ylabel("Sentence classes (log scale)")
    class_ax.set_title("Unique sentence labels collapse once repetition is required")
    class_ax.tick_params(axis="x", rotation=35)
    annotate_bars(class_ax, class_counts, class_ax.get_yscale(), percent=False)

    clip_ax.yaxis.set_major_formatter(FuncFormatter(format_compact_count))
    class_ax.yaxis.set_major_formatter(FuncFormatter(format_compact_count))

    fig.text(
        0.5,
        -0.02,
        "The full dataset has 31,047 matched clips but roughly 30,008 unique English sentences. "
        "A sentence-classification CNN therefore needs a repeated-sentence subset rather than a full 31k-class softmax.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_plot(fig, "how2sign_dataset_constraint")


def plot_subset_benchmark(stats: list[BenchmarkStat]) -> None:
    names = [stat.name for stat in stats]
    x_positions = list(range(len(stats)))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle(
        "How2Sign 3D CNN Benchmark Progression",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )

    split_ax, metric_ax = axes
    split_ax.bar(names, [stat.train_clips for stat in stats], label="Train", color=ACCENT)
    split_ax.bar(
        names,
        [stat.val_clips for stat in stats],
        bottom=[stat.train_clips for stat in stats],
        label="Val",
        color=ACCENT_SOFT,
    )
    split_ax.bar(
        names,
        [stat.test_clips for stat in stats],
        bottom=[stat.train_clips + stat.val_clips for stat in stats],
        label="Test",
        color=ACCENT_ALT,
    )
    split_ax.set_ylabel("Clip count")
    split_ax.set_title("Dataset size grows meaningfully from Top-12 to Top-25")
    split_ax.legend(frameon=False)
    for index, stat in enumerate(stats):
        split_ax.text(
            index,
            stat.total_clips + 10,
            f"{stat.total_clips} clips\n{stat.num_classes} classes",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT,
        )

    val_bars = metric_ax.bar(
        [position - width / 2 for position in x_positions],
        [stat.val_accuracy for stat in stats],
        width=width,
        color=ACCENT_SOFT,
        label="Val accuracy",
    )
    test_bars = metric_ax.bar(
        [position + width / 2 for position in x_positions],
        [stat.test_accuracy for stat in stats],
        width=width,
        color=ACCENT_ALT,
        label="Test accuracy",
    )
    metric_ax.set_xticks(x_positions, names)
    metric_ax.set_ylim(0.0, max(max(stat.val_accuracy, stat.test_accuracy) for stat in stats) * 1.35)
    metric_ax.set_ylabel("Accuracy")
    metric_ax.set_title("Top-25 improves held-out accuracy while covering more classes")
    metric_ax.yaxis.set_major_formatter(FuncFormatter(format_percent))
    metric_ax.legend(frameon=False)
    annotate_patch_values(metric_ax, val_bars, percent=True)
    annotate_patch_values(metric_ax, test_bars, percent=True)

    fig.text(
        0.5,
        -0.02,
        "The sentence CNN pipeline is working end to end. Label normalization keeps the same 479 clips, cuts duplicate classes, and improves held-out accuracy.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_plot(fig, "how2sign_subset_benchmark")


def plot_top25_class_distribution(summary: dict[str, object]) -> None:
    class_counts = summary["label_counts_after"]
    items = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    labels = [textwrap.fill(label.title(), width=18) for label, _ in items]
    values = [count for _, count in items]

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        "Normalized Top-25 Class Distribution",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )
    colors = [ACCENT if index < 5 else ACCENT_SOFT if index < 12 else "#B8C4D0" for index in range(len(values))]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Clips per sentence class")
    ax.set_ylabel("Sentence label")
    ax.set_title("Label normalization helps, but the repeated-sentence subset is still imbalanced")
    annotate_patch_values(ax, bars, percent=False, horizontal=True)

    fig.text(
        0.5,
        -0.02,
        "After merging duplicate labels, classes like 'Okay.', 'Hi.', and 'Good.' still dominate the subset, so head-class bias remains.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_plot(fig, "how2sign_top25_class_distribution")


def plot_top25_experiment_comparison(stats: list[ExperimentStat]) -> None:
    names = [stat.name for stat in stats]
    x_positions = list(range(len(stats)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig.suptitle(
        "3D CNN Training Experiments",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )

    val_bars = ax.bar(
        [position - width / 2 for position in x_positions],
        [stat.val_accuracy for stat in stats],
        width=width,
        color=ACCENT_SOFT,
        label="Val accuracy",
    )
    test_bars = ax.bar(
        [position + width / 2 for position in x_positions],
        [stat.test_accuracy for stat in stats],
        width=width,
        color=ACCENT_ALT,
        label="Test accuracy",
    )
    ax.set_xticks(x_positions, names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, max(max(stat.val_accuracy, stat.test_accuracy) for stat in stats) * 1.35)
    ax.yaxis.set_major_formatter(FuncFormatter(format_percent))
    ax.set_title("The normalized checkpoint is now the strongest 3D CNN model")
    ax.legend(frameon=False)
    annotate_patch_values(ax, val_bars, percent=True)
    annotate_patch_values(ax, test_bars, percent=True)

    for index, stat in enumerate(stats):
        ax.text(
            index,
            -0.055,
            stat.training_change,
            ha="center",
            va="top",
            fontsize=9,
            color=MUTED,
            transform=ax.get_xaxis_transform(),
        )

    fig.text(
        0.5,
        -0.03,
        "Merging duplicate sentence labels produced the best held-out result; training longer or reweighting alone was less effective.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save_plot(fig, "how2sign_top25_experiment_comparison")


def annotate_bars(axis, values: list[int], scale: str, *, percent: bool) -> None:
    for patch, value in zip(axis.patches, values):
        height = patch.get_height()
        y = height * (1.18 if scale == "log" else 1.01)
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            y,
            format_percent_text(value) if percent else f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )


def annotate_patch_values(axis, patches, *, percent: bool, horizontal: bool = False) -> None:
    for patch in patches:
        value = patch.get_width() if horizontal else patch.get_height()
        if horizontal:
            axis.text(
                value + max(value * 0.01, 0.3),
                patch.get_y() + patch.get_height() / 2,
                format_percent_text(value) if percent else f"{int(round(value))}",
                va="center",
                ha="left",
                fontsize=9,
                color=TEXT,
            )
        else:
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                value + max(value * 0.03, 0.01),
                format_percent_text(value) if percent else f"{int(round(value))}",
                va="bottom",
                ha="center",
                fontsize=9,
                color=TEXT,
            )


def format_compact_count(value: float, _: object) -> str:
    if value >= 1000:
        return f"{int(value / 1000)}k"
    return f"{int(value)}"


def format_percent(value: float, _: object) -> str:
    return f"{value * 100:.0f}%"


def format_percent_text(value: float) -> str:
    return f"{value * 100:.1f}%"


def save_plot(fig, basename: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{basename}.png", dpi=220)
    fig.savefig(OUTPUT_DIR / f"{basename}.svg")
    plt.close(fig)


def write_summary_json(
    threshold_stats: list[ThresholdStat],
    benchmark_stats: list[BenchmarkStat],
    experiment_stats: list[ExperimentStat],
) -> None:
    payload = {
        "threshold_stats": [asdict(stat) for stat in threshold_stats],
        "benchmark_stats": [asdict(stat) for stat in benchmark_stats],
        "experiment_stats": [asdict(stat) for stat in experiment_stats],
    }
    (OUTPUT_DIR / "how2sign_plot_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
