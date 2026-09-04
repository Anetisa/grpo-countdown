.PHONY: install test train

install:
	pip install -e ".[dev]"

test:
	pytest -q

train:
	python -m grpo_countdown.train
