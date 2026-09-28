# Thin wrapper around the CLI. Without make (e.g. plain Windows), run the same
# commands directly: `uv run accessmap <command>`, `uv run pytest`.
UV ?= uv
RUN = $(UV) run

.PHONY: setup check coverage pipeline fetch-osm fetch-images pilot analyze geolocate evaluate build-graph route-demo export-osm serve demo build-demo export-site deploy test lint

setup:
	$(UV) sync --extra dev

check:
	$(RUN) accessmap check

coverage:
	$(RUN) accessmap coverage

# Every step skips work that is already done (cached Gemini answers cost nothing).
pipeline: fetch-osm fetch-images analyze geolocate evaluate build-graph

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
	$(RUN) accessmap evaluate --final

build-graph:
	$(RUN) accessmap build-graph

route-demo:
	$(RUN) accessmap route-demo

export-osm:
	$(RUN) accessmap export-osm

serve:
	$(RUN) accessmap serve --host $${HOST:-127.0.0.1}

# Offline demo on tests/fixtures/demo: no keys, no pipeline run.
demo:
	$(RUN) accessmap demo --host $${HOST:-127.0.0.1}

build-demo:
	$(RUN) accessmap build-demo

export-site:
	$(RUN) accessmap export-site

deploy: export-site
	npx vercel deploy site --prod

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests
