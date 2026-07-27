#!/bin/sh
set -eu

python_bin="${VIRTUAL_ENV:-$(dirname "$0")/.venv}/bin/python"
exec "$python_bin" -m agent.pre_start "$@"
