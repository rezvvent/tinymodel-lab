.PHONY: install lint format test profile-resnet quickstart clean

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -v --cov=tinymodel

profile-resnet:
	uv run tinymodel profile --model resnet18 --details

quickstart:
	uv run python examples/quickstart_resnet18.py

clean:
	rm -rf .pytest_cache .coverage htmlcov artifacts build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
