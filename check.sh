#!/usr/bin/env bash
set -eEuo pipefail

poetry run black .
poetry run isort .
poetry run python -m mypy -p zarrita
