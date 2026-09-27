# Thin wrapper around the CLI. Without make (e.g. plain Windows), run the same
# commands directly: `uv run accessmap <command>`, `uv run pytest`.
UV ?= uv
RUN = $(UV) run

.PHONY: setup check coverage test lint

setup:
	$(UV) sync --extra dev

check:
	$(RUN) accessmap check

coverage:
	$(RUN) accessmap coverage

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests
