#!/usr/bin/env bash
# 다운로드한 바디 모델을 GVHMR이 기대하는 폴더 구조로 배치한다.
# 재실행해도 안전하다. 레포 루트에서 실행:  bash scripts/arrange_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$ROOT/GVHMR-main/inputs/checkpoints"

echo "[*] Repo root: $ROOT"

# --- 1. 전체 체크포인트 트리 생성 ---
mkdir -p "$CKPT/body_models/smplx" \
         "$CKPT/body_models/smpl" \
         "$CKPT/gvhmr" \
         "$CKPT/hmr2" \
         "$CKPT/vitpose" \
         "$CKPT/yolo" \
         "$ROOT/GVHMR-main/outputs"

# --- 2. SMPL-X neutral (npz) 배치 ---
SMPLX_SRC="$ROOT/models/smplx/SMPLX_NEUTRAL.npz"
if [[ -f "$SMPLX_SRC" ]]; then
  cp -f "$SMPLX_SRC" "$CKPT/body_models/smplx/SMPLX_NEUTRAL.npz"
  echo "[ok] SMPLX_NEUTRAL.npz placed"
else
  echo "[!!] missing: $SMPLX_SRC"
fi

# --- 3. SMPL neutral (pkl, basicmodel_*에서 이름 변경) ---
SMPL_SRC="$ROOT/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
if [[ -f "$SMPL_SRC" ]]; then
  cp -f "$SMPL_SRC" "$CKPT/body_models/smpl/SMPL_NEUTRAL.pkl"
  echo "[ok] SMPL_NEUTRAL.pkl placed"
else
  echo "[!!] missing: $SMPL_SRC"
fi

echo
echo "=== Body models done. Still MISSING (download from Google Drive) ==="
echo "  $CKPT/gvhmr/gvhmr_siga24_release.ckpt"
echo "  $CKPT/hmr2/epoch=10-step=25000.ckpt"
echo "  $CKPT/vitpose/vitpose-h-multi-coco.pth"
echo "  $CKPT/yolo/yolov8x.pt"
echo "  (DPVO skipped for MVP)"
echo
echo "Drive: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
