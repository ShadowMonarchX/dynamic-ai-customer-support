#!/usr/bin/env bash
set -euo pipefail

uv run python -c "import backend.app.main as m; assert m.app is not None; print('startup smoke passed')"
