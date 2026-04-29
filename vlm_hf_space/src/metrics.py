from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score

from .label_utils import normalize_label, normalize_text


def asl_citizen_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    exact = [str(r).strip() == str(p).strip() for r, p in zip(references, predictions)]
    norm_refs = [normalize_label(r) for r in references]
    norm_preds = [normalize_label(p) for p in predictions]
    norm_exact = [r == p for r, p in zip(norm_refs, norm_preds)]
    return {
        "num_samples": len(references),
        "exact_match": float(accuracy_score([1] * len(exact), exact)) if exact else 0.0,
        "normalized_exact_match": float(accuracy_score([1] * len(norm_exact), norm_exact)) if norm_exact else 0.0,
    }


def how2sign_metrics(references: list[str], predictions: list[str]) -> dict[str, Any]:
    try:
        import sacrebleu
        from rouge_score import rouge_scorer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "How2Sign metrics require sacrebleu and rouge-score. Install the project "
            "requirements into the same environment used by accelerate: "
            "python -m pip install -r requirements.txt"
        ) from exc

    refs = [normalize_text(r) for r in references]
    preds = [normalize_text(p) for p in predictions]
    if not refs:
        return {"num_samples": 0, "bleu": 0.0, "chrf": 0.0, "rouge_l": 0.0}
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)) / len(refs)
    return {
        "num_samples": len(refs),
        "bleu": float(bleu),
        "chrf": float(chrf),
        "rouge_l": float(rouge_l),
    }


def metrics_for_task(task_name: str, references: list[str], predictions: list[str]) -> dict[str, Any]:
    if task_name == "asl_citizen":
        return asl_citizen_metrics(references, predictions)
    if task_name == "how2sign":
        return how2sign_metrics(references, predictions)
    raise ValueError(f"Unsupported task_name for metrics: {task_name}")
