# 설정 가이드 — GVHMR Docker 환경 (RTX 5070 / Blackwell)

## 업스트림 설치 방식을 대체하고 Docker를 사용하는 이유
GVHMR의 `requirements.txt`는 `torch==2.3.0+cu121` 및 cu121 기반 `pytorch3d` 휠(wheel) 파일을 사용하도록 고정되어 있다. 
이 버전들은 RTX 5070 (Blackwell, sm_120) 아키텍처에서 동작하지 **않는다**. 따라서 본 프로젝트의 Docker 이미지는 **CUDA 12.8 + PyTorch 2.7 (cu128)** 환경을 구성하고 `pytorch3d`를 소스에서 직접 빌드한다. MVP 버전에서는 **DPVO를 건너뛴다** (선택 사항이며, GVHMR은 DPVO 없이도 동작한다).

---

## 단계 0 — 사전 준비 사항 (최초 1회)
1. **Docker Desktop**을 실행하고, **WSL2 백엔드** 및 **GPU 지원**이 활성화되어 있어야 한다.
   (Settings → Resources → WSL Integration; 최신 버전에서는 GPU 지원이 자동 적용된다.)
2. 데몬이 정상 작동하는지 확인한다: `docker info` 명령어가 성공적으로 실행되어야 한다.
   (데몬이 설치되어 있지만 실행 중이 아닐 수 있으므로 Docker Desktop을 먼저 켜야 한다.)

## 단계 1 — 바디 모델 배치 ✅ 완료됨
`bash scripts/arrange_models.sh` 스크립트를 통해 다음 파일들이 복사되었다:
- `GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`
- `GVHMR-main/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`

## 단계 2 — 누락된 4개의 가중치 파일 다운로드 ⬅️ 사용자 수행
GVHMR Google Drive에서 파일을 다운로드한다:
https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD

다운로드한 파일들을 정확히 다음 위치에 배치한다:
```
GVHMR-main/inputs/checkpoints/
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```
(`dpvo/` 디렉터리는 필요하지 않으므로 건너뛴다.)

## 단계 3 — 컨테이너 빌드 및 시작
```bash
docker compose -f docker/docker-compose.yml up -d --build
```
⚠️ 첫 번째 빌드는 시간이 오래 소요된다 (10~40분). 소스 코드로부터 `pytorch3d`를 직접 컴파일하기 때문이다.

## 단계 4 — 컨테이너 내부 GPU 인식 검증
```bash
docker compose -f docker/docker-compose.yml exec gvhmr \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
다음과 같은 출력이 예상된다: `2.7.1+cu128 True NVIDIA GeForce RTX 5070`

## 단계 5 — 데모 실행 (단계 2 가중치 배치 후) ✅ 정상 작동 확인됨
```bash
docker compose -f docker/docker-compose.yml exec gvhmr \
  python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s
```
(`-s` 플래그 = 정적 카메라 설정, DPVO 스킵)
출력 결과는 `GVHMR-main/outputs/demo/<name>/` 위치에 저장된다:
- `tennis_3_incam_global_horiz.mp4` — 원본 입력 영상과 3D 오버레이가 나란히 배치된 영상
- `hmr4d_results.pt` — **SMPL 파라미터** (M1 단계에서 FBX 변환을 위한 소스 파일)

---

## 단계 6 — FBX 내보내기 (M1)
총 2단계로 나뉜다. 첫 번째 단계는 GVHMR (GPU) 컨테이너에서 실행되고, 두 번째 단계는 Blender 컨테이너에서 실행된다.

```bash
# 새로운 ../scripts 마운트를 활성화하기 위해 gvhmr 컨테이너를 재생성한다 (재빌드 아님):
docker compose -f docker/docker-compose.yml up -d gvhmr

# Blender 이미지를 빌드한다 (컴파일이 아닌 다운로드 위주이므로 비교적 빠름):
docker compose -f docker/docker-compose.yml build blender
docker compose -f docker/docker-compose.yml up -d blender

# 1단계: hmr4d_results.pt -> anim.npz 변환 (FK와 SMPL-X 간 교차 검증 포함)
docker compose -f docker/docker-compose.yml exec gvhmr \
  python /app/scripts/pt_to_anim.py \
    --pt outputs/demo/tennis/hmr4d_results.pt \
    --out outputs/demo/tennis/anim.npz --fps 30 --space global

# 2단계: anim.npz -> FBX + GLB 변환 (자세가 적용된 본과 소스 데이터 간 교차 검증 포함)
docker compose -f docker/docker-compose.yml exec blender \
  python /work/scripts/anim_to_fbx.py \
    --anim /work/outputs/demo/tennis/anim.npz \
    --fbx /work/outputs/demo/tennis/tennis.fbx \
    --glb /work/outputs/demo/tennis/tennis.glb --zup
```
두 단계 모두 최대 오차 자가 검증(max-error self-check)을 수행하며, 오차가 허용 임계치를 초과할 경우 조용히 잘못된 파일을 생성하는 대신 작업을 중단(abort)시킨다.

---

## 해결된 호환성 이슈 (Blackwell / torch 2.7 포팅)
- **pytorch3d** — cu128 용 휠 파일이 존재하지 않아 소스에서 직접 빌드하였다. 컴파일 중 WSL 메모리 부족(OOM)을 방지하기 위해 `--no-build-isolation` 옵션과 `MAX_JOBS=4` 환경 변수를 사용하였다.
- **chumpy** — 빌드 격리(build isolation) 환경에서 `import pip`가 실패하는 문제를 해결하기 위해, 최신 pip API를 지원하는 GitHub 마스터 브랜치 버전을 `--no-build-isolation` 옵션으로 설치하였다.
- **tkinter** — GVHMR의 `body_model.py` 파일 내 불필요한 `from turtle import forward` 구문이 존재하며, `turtle` 라이브러리는 tkinter를 요구하므로 `python3-tk` 패키지를 추가하였다.
- **weights_only** — torch 2.6 버전 이상부터 `torch.load(weights_only=True)`가 기본값으로 적용된다. 하지만 torch 2.3 시절의 체크포인트(YOLO/ViTPose/HMR2/GVHMR)를 불러오기 위해서는 이전 동작 방식이 필요하므로 환경 변수 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 설정하였다 (`docker-compose.yml` 내 적용됨).
