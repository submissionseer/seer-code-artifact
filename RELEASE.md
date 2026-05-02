# Release And Review Bundle

This repository is being prepared for an anonymized NeurIPS Evaluations & Datasets submission as a method/evaluation artifact, not as a new-dataset contribution.

## What We Release For Review

The review bundle is code-first:

- core `seer/` package code
- canonical pipeline scripts for rollouts, SFT, IPO/DPO, evaluation, and RunPod ColBERT serving
- configs needed to reproduce the paper experiments
- top-level reproduction docs (`README.md`, `REPRO.md`, `RELEASE.md`)
- tests for the canonical prep, scoring, and evaluation utilities
- top-level setup docs, license, lock files, and environment-variable template

Raw datasets, generated rollouts, checkpoints, local logs, W&B exports, private context notes, and API keys are excluded.

## License And Redistribution Policy

The code bundle is released under the repository MIT license. External assets remain under their upstream terms:

- HotpotQA data and processed Wikipedia are CC BY-SA 4.0; HotpotQA code is Apache-2.0.
- MuSiQue is CC BY 4.0 and should be used with the official split/leakage cautions.
- Wikimedia/Wikipedia dump text is generally CC BY-SA 4.0 and GFDL, with documented exceptions.
- ColBERT code is MIT; ColBERT checkpoints and third-party Hugging Face indexes should be obtained from their source repositories.
- Meta Llama 3 weights are governed by the Meta Llama 3 Community License and Acceptable Use Policy.
- OpenRouter, OpenAI, Google Gemini, Anthropic, and Jina API use is governed by their provider terms at the time of use.

Generated rollouts, scored files, Jina caches, reader outputs, and LoRA adapters are derived artifacts. They should only be redistributed when the upstream dataset, corpus, model, and API terms permit it. Otherwise, the release should provide schemas, manifests, hashes, prompts, configs, and scripts so users can regenerate the artifacts from official sources using their own access credentials.

This applies directly to retrieval assets:

- Hotpot local retrieval needs a compatible ColBERT index and matching
  `collection.tsv`
- MuSiQue retrieval needs a separately provisioned `wiki20m` ColBERT service
- the review bundle provides the surrounding infrastructure and exact commands,
  but not third-party retrieval corpora whose redistribution status is unclear

## Build A Review Bundle

```bash
make review_bundle
```

Equivalent direct command:

```bash
uv run python scripts/release/export_review_bundle.py --output dist/seer_review_bundle.zip
```

Use `--include-paper-source` only if we explicitly decide to ship paper LaTeX source in the supplemental bundle. Use `--include-evidence` only after the evidence docs have passed the anonymity scan.

## Anonymity And Secret Checks

The bundle exporter scans copied text files for common release hazards:

- hard-coded API keys or provider tokens
- local home-directory absolute paths
- user identifiers from local machine paths
- credential-helper file references
- S3 URIs that could expose private bucket names

The exporter fails by default if it finds a high-risk pattern. Use `--allow-warnings` only for local debugging, not for submission.

## Non-Dataset Position

We are not claiming the generated rollouts or preference pairs as a primary new dataset contribution. That means NeurIPS dataset-hosting and Croissant metadata requirements should not apply, but the paper still needs reproducibility details, exact commands, dependencies, and artifact access instructions.
