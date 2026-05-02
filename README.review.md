# SEER Review Bundle

This is an anonymized, code-first review bundle for the SEER Evaluations & Datasets submission.

## Quick Start

```bash
uv sync
make env-check
make release-smoke
uv run pytest tests/test_sft_prepare_data.py tests/test_dpo_utils.py tests/test_dpo_validate.py tests/test_normalize_datasets_script.py
```

Use `README.md` for the repo surface, `REPRO.md` for the paper reproduction map,
and `RELEASE.md` for release-policy / asset-redistribution constraints.

## Canonical Pipeline Surface

- `scripts/rollouts/generate.py`
- `scripts/sft/prepare_data.py`
- `scripts/sft/train.py`
- `scripts/sft/eval.py`
- `scripts/dpo/generate.py`
- `scripts/dpo/score.py`
- `scripts/dpo/prepare.py`
- `scripts/dpo/train.py`
- `scripts/eval/compare_rollout_reader_em.py`
- `scripts/eval/monitoring_signal_report.py`

## External Dependencies

API-backed runs require credentials in environment variables documented in `.env.example`.
Raw datasets, generated rollouts, checkpoints, and private logs are intentionally not included.

## Dataset Position

This bundle does not claim or host a new dataset. It contains code and configuration for reproducing the evaluation method and experiments.
