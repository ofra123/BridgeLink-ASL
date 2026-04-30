from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or REPO_ROOT) / p


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path, Path.cwd())
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out = resolve_path(path, Path.cwd())
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


def apply_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    out = dict(config)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {p}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def output_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return resolve_path(override or config["output_dir"])


def sample_rows(rows: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None or max_samples <= 0:
        return rows
    return rows[:max_samples]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
