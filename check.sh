#!/usr/bin/env bash
set -eEuo pipefail

if [ "$#" = 1 ] && [ "$1" = "--fix" ]; then
    poetry run black .
    poetry run isort .
    poetry run ruff check . --fix
    poetry run mypy .
else
    poetry run black --check .
    poetry run isort --check .
    poetry run ruff check .
    poetry run mypy .
fi
