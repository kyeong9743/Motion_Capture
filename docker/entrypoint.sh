#!/usr/bin/env bash
# 바인드 마운트된 GVHMR 레포를 첫 시작 시 editable 패키지로 설치한 후,
# 컨테이너 명령어로 제어를 넘긴다.
set -e

if [[ -f /app/GVHMR/setup.py && ! -f /app/.gvhmr_installed ]]; then
  echo "[entrypoint] installing GVHMR (editable, no-deps)..."
  pip install -e /app/GVHMR --no-deps
  touch /app/.gvhmr_installed
fi

exec "$@"
