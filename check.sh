#!/usr/bin/env bash
set -eEuo pipefail

poetry run black .
poetry run python -m mypy -p zarrita
poetry run python -m pylint -j2 zarrita
