# Thin wrapper around the CLI. Without make (e.g. plain Windows), run the same
# commands directly: `uv run accessmap <command>`, `uv run pytest`.
UV ?= uv
RUN = $(UV) run

.PHONY: setup check coverage fetch-osm fetch-images pilot analyze geolocate evaluate serve export-site deploy test lint

setup:
	$(UV) sync --extra dev

check:
	$(RUN) accessmap check

coverage:
	$(RUN) accessmap coverage

fetch-osm:
	$(RUN) accessmap fetch-osm

fetch-images:
	$(RUN) accessmap fetch-images

pilot:
	$(RUN) accessmap pilot

analyze:
	$(RUN) accessmap analyze --all

geolocate:
	$(RUN) accessmap geolocate

evaluate:
	$(RUN) accessmap evaluate

serve:
	$(RUN) accessmap serve --host $${HOST:-127.0.0.1}

export-site:
	$(RUN) accessmap export-site

deploy: export-site
	npx vercel deploy site --prod

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests
