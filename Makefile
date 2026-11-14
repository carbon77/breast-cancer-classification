.DEFAULT_GOAL := help
RUN := uv run --locked
.PHONY: help install pull push download prepare test verify-dvc clean clean-preview

help:
	@$(info Usage: make [target])
	@$(info )
	@$(info Setup and data:)
	@$(info   install        Install project dependencies from uv.lock)
	@$(info   pull           Restore the BUSI dataset from DVC (Yandex Object Storage, S3))
	@$(info   push           Upload the BUSI dataset to DVC (Yandex Object Storage, S3))
	@$(info   download       Download BUSI from Kaggle and prepare the dataset)
	@$(info   prepare        Prepare the local BUSI dataset and generated artifacts)
	@$(info )
	@$(info Validation:)
	@$(info   test           Run the unit test suite)
	@$(info   verify-dvc     Verify dataset and binary artifact recovery via local DVC storage)
	@$(info )
	@$(info Cleanup:)
	@$(info   clean-preview  Show which local data and artifacts clean would remove)
	@$(info   clean          Remove local data and artifacts, preserving splits and DVC pointers)
	@$(info )
	@$(info   help           Show this help (also the default target))
	@$(info )
	@$(info Typical workflow: make install, make pull, make prepare)
	@$(info First download from Kaggle: make install, make download)

install:
	uv sync --locked

pull:
	$(RUN) dvc pull data/busi.dvc

push:
	$(RUN) dvc push data/busi.dvc

download:
	$(RUN) python -m busi.prepare --kaggle

prepare:
	$(RUN) python -m busi.prepare

test:
	$(RUN) python -m unittest discover -s tests

verify-dvc:
	$(RUN) python -m tests.verify_dvc

clean-preview:
	uv run --no-sync scripts/clean.py --dry-run

clean:
	uv run --no-sync scripts/clean.py
