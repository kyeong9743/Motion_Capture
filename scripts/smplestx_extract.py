"""
단계 2b (M5): 영상 -> 프레임별 expressive 파라미터 (SMPLest-X 경유).

smplestx 컨테이너 내부에서 실행한다 (작업 디렉터리 /app/SMPLest-X, 레포 import 가능).
cv2로 영상을 직접 읽고 (프레임 덤프 없이), YOLO 최대 인물 bbox +
SMPLest-X를 프레임마다 실행하여, 그래프팅 단계에 필요한 것만 저장한다:

  smplestx_params.npz:
    left_hand_pose  [F,45]  axis-angle
    right_hand_pose [F,45]
    jaw_pose        [F,3]
    expression      [F,10]
    valid           [F]     bool — 해당 프레임에서 인물 탐지 여부
    (눈 시선은 SMPLest-X가 예측하지 않는다 — 내부적으로 zero eye pose 사용)

사용법:
  python /app/scripts/smplestx_extract.py --video <in.mp4> --out <smplestx_params.npz> \
      [--ckpt-name smplest_x_h]
"""
import argparse
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

sys.path.insert(0, "/app/SMPLest-X")
from main.base import Tester              # noqa: E402
from main.config import Config            # noqa: E402
from human_models.human_models import SMPLX  # noqa: E402
from utils.data_utils import process_bbox, generate_patch_image  # noqa: E402
from utils.inference_utils import non_max_suppression            # noqa: E402
from ultralytics import YOLO              # noqa: E402
from tqdm import tqdm                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-name", default="smplest_x_h")
    args = ap.parse_args()

    root = Path("/app/SMPLest-X")
    cfg = Config.load_config(str(root / "pretrained_models" / args.ckpt_name / "config_base.py"))
    ckpt = str(root / "pretrained_models" / args.ckpt_name / f"{args.ckpt_name}.pth.tar")
    cfg.update_config({
        "model": {"pretrained_model_path": ckpt},
        "log": {"exp_name": "extract", "log_dir": "/tmp/smplestx_log"},
    })
    cfg.prepare_log()

    # SMPLX 싱글톤을 바디 모델 경로로 초기화해야 한다 —
    # get_model()이 디코더를 구성하기 *전에* (업스트림 inference.py와 동일 순서)
    SMPLX(cfg.model.human_model_path)

    tester = Tester(cfg)
    tester._make_model()
    model = tester.model

    det_path = getattr(cfg.inference.detection, "model_path", "./pretrained_models/yolov8x.pt")
    detector = YOLO(det_path)
    to_tensor = transforms.ToTensor()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[smplestx_extract] cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[smplestx_extract] video frames={total}")

    lhand, rhand, jaw, expr, valid = [], [], [], [], []
    Z45, Z3, Z10 = np.zeros(45, np.float32), np.zeros(3, np.float32), np.zeros(10, np.float32)

    for _ in tqdm(range(total), desc="SMPLest-X"):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        img = frame_bgr[:, :, ::-1].astype(np.float32)  # RGB float, like load_img
        H, W = img.shape[:2]

        boxes = detector.predict(img, device="cuda", classes=0, conf=0.5,
                                 save=False, verbose=False)[0].boxes.xyxy.detach().cpu().numpy()
        if len(boxes) < 1:
            lhand.append(Z45); rhand.append(Z45); jaw.append(Z3); expr.append(Z10)
            valid.append(False)
            continue
        # 최대 인물 선택
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes[int(np.argmax(areas))]
        xywh = np.array([b[0], b[1], abs(b[2] - b[0]), abs(b[3] - b[1])])

        bbox = process_bbox(bbox=xywh, img_width=W, img_height=H,
                            input_img_shape=cfg.model.input_img_shape,
                            ratio=getattr(cfg.data, "bbox_ratio", 1.25))
        if bbox is None:
            lhand.append(Z45); rhand.append(Z45); jaw.append(Z3); expr.append(Z10)
            valid.append(False)
            continue
        patch, _, _ = generate_patch_image(cvimg=img, bbox=bbox, scale=1.0, rot=0.0,
                                           do_flip=False, out_shape=cfg.model.input_img_shape)
        t = to_tensor(patch.astype(np.float32)) / 255.0
        with torch.no_grad():
            out = model({"img": t.cuda()[None]}, {}, {}, "test")

        lhand.append(out["smplx_lhand_pose"][0].detach().cpu().numpy().reshape(-1).astype(np.float32))
        rhand.append(out["smplx_rhand_pose"][0].detach().cpu().numpy().reshape(-1).astype(np.float32))
        jaw.append(out["smplx_jaw_pose"][0].detach().cpu().numpy().reshape(-1).astype(np.float32))
        expr.append(out["smplx_expr"][0].detach().cpu().numpy().reshape(-1)[:10].astype(np.float32))
        valid.append(True)
    cap.release()

    F = len(valid)
    n_ok = int(np.sum(valid))
    print(f"[smplestx_extract] frames={F} detected={n_ok} ({100.0 * n_ok / max(F,1):.1f}%)")
    if n_ok == 0:
        raise SystemExit("[smplestx_extract] no person detected in any frame; aborting.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             left_hand_pose=np.stack(lhand), right_hand_pose=np.stack(rhand),
             jaw_pose=np.stack(jaw), expression=np.stack(expr),
             valid=np.array(valid, dtype=bool))
    print(f"[smplestx_extract] saved -> {out_path}")


if __name__ == "__main__":
    main()
