PYTHON ?= uv run python
CONFIG ?= configs/default.yaml
ARGS ?=

.PHONY: bootstrap env-check download-index download-datasets normalize release-smoke \
	rollouts rollouts_hotpot rollouts_musique \
	sft_prepare sft_train sft_eval \
	dpo_generate dpo_score dpo_prepare dpo_train \
	em_compare review_bundle \
	setup download label_tasks label_app decomp_app \
	optimize_seer optimize_seer_gold optimize_seer_human optimize_seer_dry \
	optimize_rewriter optimize_rewriter_dry \
	eval figures export

bootstrap:
	./scripts/bootstrap.sh

env-check:
	$(PYTHON) scripts/env_check.py --config $(CONFIG)

download-index:
	$(PYTHON) scripts/download_colbert_index.py --config $(CONFIG)

download-datasets:
	$(PYTHON) scripts/download_datasets.py --config $(CONFIG)

normalize:
	$(PYTHON) scripts/normalize_datasets.py --config $(CONFIG)

release-smoke:
	$(PYTHON) scripts/release/release_smoke.py $(ARGS)

rollouts:
	$(PYTHON) scripts/rollouts/generate.py --config $(CONFIG) $(ARGS)

rollouts_hotpot:
	$(PYTHON) scripts/rollouts/generate.py --config $(CONFIG) --dataset hotpot $(ARGS)

rollouts_musique:
	$(PYTHON) scripts/rollouts/generate.py --config $(CONFIG) --dataset musique $(ARGS)

sft_prepare:
	$(PYTHON) scripts/sft/prepare_data.py $(ARGS)

sft_train:
	$(PYTHON) scripts/sft/train.py $(ARGS)

sft_eval:
	$(PYTHON) scripts/sft/eval.py --config $(CONFIG) $(ARGS)

dpo_generate:
	$(PYTHON) scripts/dpo/generate.py $(ARGS)

dpo_score:
	$(PYTHON) scripts/dpo/score.py $(ARGS)

dpo_prepare:
	$(PYTHON) scripts/dpo/prepare.py $(ARGS)

dpo_train:
	$(PYTHON) scripts/dpo/train.py $(ARGS)

em_compare:
	$(PYTHON) scripts/eval/compare_rollout_reader_em.py $(ARGS)

review_bundle:
	$(PYTHON) scripts/release/export_review_bundle.py $(ARGS)

setup download label_tasks label_app decomp_app optimize_seer optimize_seer_gold optimize_seer_human optimize_seer_dry optimize_rewriter optimize_rewriter_dry eval figures export:
	@echo "Target '$@' is hard-retired."
	@echo "Use canonical targets (env-check, download-datasets, download-index, normalize, release-smoke, rollouts, sft_*, dpo_*, em_compare, review_bundle) and the top-level README/REPRO docs."
	@exit 1
