from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "datasets"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "normalize_datasets.py"


class _NullLogger:
    def info(self, *args, **kwargs):
        return None


def _load_fixture_records(name: str) -> list[dict]:
    payload = json.loads((FIXTURE_DIR / name).read_text())
    return payload["records"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("normalize_datasets_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_helpers_use_adapters_and_emit_musique_hop_metadata(tmp_path):
    mod = _load_script_module()
    raw_dir = tmp_path / "raw"
    norm_dir = tmp_path / "norm"
    logger = _NullLogger()

    _write_jsonl(raw_dir / "hotpot" / "raw.jsonl", _load_fixture_records("hotpot_raw_samples.json"))
    _write_jsonl(raw_dir / "musique" / "raw.jsonl", _load_fixture_records("musique_raw_samples.json"))

    mod._normalize_hotpot(raw_dir, norm_dir / "hotpot.jsonl", logger)
    mod._normalize_musique(raw_dir, norm_dir / "musique.jsonl", logger)

    hotpot_rows = [json.loads(line) for line in (norm_dir / "hotpot.jsonl").read_text().splitlines()]
    musique_rows = [json.loads(line) for line in (norm_dir / "musique.jsonl").read_text().splitlines()]

    assert len(hotpot_rows) == 2
    assert all(r["dataset"] == "hotpot" for r in hotpot_rows)
    assert all("candidates" in r for r in hotpot_rows)

    assert len(musique_rows) == 3
    hop_counts = sorted(r["num_hops"] for r in musique_rows)
    assert hop_counts == [2, 3, 4]
    assert all(r["dataset"] == "musique" for r in musique_rows)
    assert all("question_decomposition" in r for r in musique_rows)

    # Clean-up request in scope: these old helpers are gone from this script.
    assert not hasattr(mod, "_normalize_scifact")
    assert not hasattr(mod, "_normalize_msmarco")


def test_main_only_writes_hotpot_and_musique_and_force_overwrites(tmp_path, monkeypatch):
    mod = _load_script_module()
    raw_dir = tmp_path / "raw"
    norm_dir = tmp_path / "norm"
    logs_dir = tmp_path / "logs"

    hotpot_rows = _load_fixture_records("hotpot_raw_samples.json")
    musique_rows = _load_fixture_records("musique_raw_samples.json")
    _write_jsonl(raw_dir / "hotpot" / "raw.jsonl", hotpot_rows)
    _write_jsonl(raw_dir / "musique" / "raw.jsonl", musique_rows[:1])

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(raw_dir),
                "norm_dir": str(norm_dir),
                "logs_dir": str(logs_dir),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["normalize_datasets.py", "--config", str(config_path)])
    mod.main()

    assert (norm_dir / "hotpot.jsonl").exists()
    assert (norm_dir / "musique.jsonl").exists()
    assert not (norm_dir / "scifact.jsonl").exists()
    assert not (norm_dir / "msmarco.jsonl").exists()

    first_pass = [json.loads(line) for line in (norm_dir / "musique.jsonl").read_text().splitlines()]
    assert len(first_pass) == 1

    # Without --force, existing output should be preserved.
    _write_jsonl(raw_dir / "musique" / "raw.jsonl", musique_rows)
    monkeypatch.setattr(sys, "argv", ["normalize_datasets.py", "--config", str(config_path)])
    mod.main()
    second_pass = [json.loads(line) for line in (norm_dir / "musique.jsonl").read_text().splitlines()]
    assert len(second_pass) == 1

    # With --force, output should be overwritten.
    monkeypatch.setattr(sys, "argv", ["normalize_datasets.py", "--config", str(config_path), "--force"])
    mod.main()
    forced_pass = [json.loads(line) for line in (norm_dir / "musique.jsonl").read_text().splitlines()]
    assert len(forced_pass) == 3
