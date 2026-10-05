#!/usr/bin/env bash
set -euo pipefail
SANW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SANW_IMAGE=sanw-v10-local
if [[ "${1:-}" == fresh-encode ]]; then SANW_IMAGE=sanw-v10-encoder; fi
exec docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$SANW_ROOT,dst=/workspace/scratch/81995298881b/SANW_practical_work" \
  "$SANW_IMAGE" "$@"
