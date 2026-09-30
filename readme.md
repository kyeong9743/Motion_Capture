# MediaMotion

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r167-000000?logo=threedotjs&logoColor=white)
![Blender](https://img.shields.io/badge/Blender-4.2_(bpy)-E87D0D?logo=blender&logoColor=white)
![License](https://img.shields.io/badge/License-Research_Only-red)

**단일 카메라 영상 → 3D 휴머노이드 애니메이션(FBX/GLB) 자동 변환 파이프라인**

단일 RGB 비디오 하나로 전신 3D 모션캡처를 수행하고, 산업 표준 FBX 포맷으로 내보내는 로컬 GPU 파이프라인이다.  
상용 모션캡처 서비스(DeepMotion 등)와 동등한 워크플로를 RTX 5070 단일 GPU 위에 재현하되, 모든 후처리 파라미터를 노출하여 연구·실무 양쪽에서 자유롭게 튜닝할 수 있도록 설계하였다.

### Demo (Input vs Output)

| 원본 비디오 (Input) | 3D 모션 캡처 결과 (Output) |
|:---:|:---:|
| <img src="./assets/demo_input.gif" height="300"/> | <img src="./assets/demo_output.gif" height="300"/> |
| *단일 RGB 카메라 영상* | *자동 생성된 FBX (Blender 렌더링)* |

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [핵심 모델과 모듈](#3-핵심-모델과-모듈)
4. [파이프라인 상세](#4-파이프라인-상세)
5. [후처리 엔진](#5-후처리-엔진)
6. [웹 인터페이스](#6-웹-인터페이스)
7. [프로젝트 파일 구조](#7-프로젝트-파일-구조)
8. [문서 파일 가이드](#8-문서-파일-가이드)
9. [환경 설정 및 실행](#9-환경-설정-및-실행)
10. [알려진 한계와 향후 과제](#10-알려진-한계와-향후-과제)
11. [라이선스와 참고 문헌](#11-라이선스와-참고-문헌)

---

## 1. 프로젝트 개요

### 풀고자 하는 문제

단일 카메라 비디오에서 3D 인체 모션을 추출(Human Motion Recovery)하여, 게임·VFX·VTubing 등에서 바로 사용할 수 있는 골격 애니메이션 파일(FBX)로 변환한다.  
기존 상용 솔루션(DeepMotion, Rokoko Video 등)은 클라우드 의존·비용·파라미터 비공개라는 한계가 있으며, 본 프로젝트는 이를 **완전 로컬, 오픈 파이프라인**으로 대체한다.

### 핵심 기여

| 기여 | 설명 |
|------|------|
| **GVHMR + SMPL-X → FBX 엔드투엔드 자동화** | 학술 모델의 `.pt` 출력을 산업 표준 FBX까지 자동 변환하는 4단계 파이프라인 |
| **SMPLest-X 손/얼굴 그래프팅** | 22-관절 바디와 별도 예측한 손가락/턱을 55-관절로 합성, flat_hand_mean 변환 포함 |
| **속도 적응형 One-Euro 스무딩** | 기존 Gaussian 대비 빠른 동작의 75%를 보존하면서 동일 수준의 지터 제거 |
| **이중 자기검증(Self-Validation)** | FK ↔ SMPL-X 모델 교차검증 + Blender 본 위치 교차검증으로 침묵 오류 차단 |
| **Blackwell GPU 포팅** | CUDA 12.8 / PyTorch 2.7 기반 Docker 이미지로 RTX 5070(sm_120) 지원 |
| **Web UI (Next.js + Three.js)** | 업로드 → 실시간 진행률 → 입력/3D 나란히 비교 프리뷰 → FBX 다운로드 |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js :5000)                                          │
│  upload · options panel · real-time stepper · split preview · FBX  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP (REST)
┌───────────────────────────▼─────────────────────────────────────────┐
│  Backend (FastAPI :8010)                                            │
│  job queue · stage orchestration · tqdm log parsing · file serving  │
└───────┬────────────┬────────────┬───────────────────────────────────┘
        │docker exec │docker exec │docker exec
┌───────▼──────┐┌────▼───────┐┌───▼──────────┐
│  GVHMR       ││ SMPLest-X  ││  Blender     │
│  (GPU)       ││ (GPU)      ││  (CPU)       │
│  cu128/2.7   ││ cu128/2.7  ││  bpy 4.2     │
│              ││            ││              │
│ • demo.py    ││ • extract  ││ • anim→FBX   │
│ • pt_to_anim ││   .py      ││ • anim→GLB   │
└──────────────┘└────────────┘└──────────────┘
        │               │              │
        └───────────────┴──────────────┘
                   공유 볼륨: /data/jobs/<id>/
```

**핵심 설계 결정: Docker 컨테이너 격리**

각 모델의 Python 의존성이 상호 충돌한다(GVHMR은 numpy<1.24 + pytorch3d 소스빌드, Blender는 Python 3.11 + bpy 4.2). 이를 Docker Compose로 격리하고, 백엔드가 `docker exec`로 오케스트레이션한다. 모든 컨테이너는 `/data` 볼륨을 공유하여 중간 산출물을 직접 파일로 전달한다.

---

## 3. 핵심 모델과 모듈

### 3.1 GVHMR — Gravity-View-Aware Human Motion Recovery

| 항목 | 내용 |
|------|------|
| **논문** | World-Grounded Human Motion Recovery via Gravity-View Augmentation (SIGGRAPH Asia 2024) |
| **역할** | 영상 전체의 시간적 맥락을 활용한 3D 전신 포즈·궤적 추정 (22-관절 SMPL-X) |
| **선정 이유** | 프레임별 추정(HMR2 등) 대비 **시간적 모션 사전(temporal motion prior)**을 탑재하여, 가려진 팔·다리가 자연스럽게 복원됨. 정적 카메라(`-s`) 모드에서 DPVO를 건너뛰어 추론 속도 향상 |
| **내부 서브모델** | YOLOv8x (인물 탐지) → ViTPose-H (2D 키포인트) → HMR2 (초기 SMPL 추정) → GVHMR Transformer (시간적 정제 + 월드 좌표 변환) |
| **출력** | `hmr4d_results.pt` — 프레임별 `global_orient`, `body_pose`, `transl`, `betas` (SMPL-X 파라미터) |

**왜 GVHMR인가?**

단일 영상 3D HMR 분야에서 GVHMR은 "World-Grounded" 복원을 제공하는 최초의 모델 중 하나이다.  
기존 HMR2, WHAM 등은 카메라 좌표계에서의 상대 포즈만 복원하거나, 글로벌 궤적 추정에 별도 SLAM이 필요했다.  
GVHMR은 중력 방향 추정을 네트워크 내부에 통합하여, 카메라 움직임 없이도(정적 카메라) 월드 좌표의 포즈와 이동 궤적을 직접 출력한다.

### 3.2 SMPLest-X — 손/얼굴 Expressive 추정

| 항목 | 내용 |
|------|------|
| **논문** | SMPLest-X (TPAMI 2025), SMPLer-X의 후속 |
| **역할** | 프레임별 손가락(좌우 각 15관절, 45 axis-angle) + 턱(3 axis-angle) + 표정(10 blendshape) 추정 |
| **선정 이유** | 선행 SMPLer-X 대비 **Hand-PA-MPJPE 15% 향상**, **mmcv/mmdet/mmpose 의존성 제거**로 Docker 포팅 난이도가 GVHMR과 동급 |
| **한계** | 프레임 단위 추정이므로 시간적 안정성 없음 → 후처리 One-Euro 필터로 보정. 눈 시선(eye gaze) 미지원 (내부에서 zero pose 사용) |

**왜 SMPLer-X가 아닌 SMPLest-X인가?**

SMPLer-X는 mmcv, mmdet, mmpose 의존성 지옥(dependency hell)을 수반하여 CUDA 12.8 환경에서 빌드가 극히 어렵다. SMPLest-X는 이 의존성을 완전히 제거하면서 정량 성능이 더 높아, 동일 Docker 포팅 레시피(chumpy no-build-isolation, numpy 1.23.5)를 재활용할 수 있다.

### 3.3 SMPL-X 바디 모델

| 항목 | 내용 |
|------|------|
| **역할** | 파라메트릭 인체 3D 메쉬 모델. 10개 shape parameter(β)로 체형을, 관절 axis-angle로 포즈를 표현 |
| **사용처** | Forward Kinematics 검증, rest-pose 관절 위치 추출, LBS(Linear Blend Skinning) 가중치로 메쉬 스킨닝 |
| **flat_hand_mean 이슈** | GVHMR은 `flat_hand_mean=True`(편 손)로 바디를 예측하고, SMPLest-X는 `flat_hand_mean=False`(MANO 기본 컬 자세) 기준으로 손을 예측. 이 불일치를 `pt_to_anim.py`에서 MANO hand_mean 벡터를 더하는 것으로 해결. 이 변환 없이는 손가락이 절반만 쥐어진 부정확한 자세가 됨 |

### 3.4 Blender (bpy 4.2, Headless)

| 항목 | 내용 |
|------|------|
| **역할** | `anim.npz`(관절별 글로벌 회전/위치) → FBX + GLB 베이킹 |
| **선정 이유** | FBX SDK가 사실상 Blender bpy 또는 Autodesk SDK로만 접근 가능. bpy는 PyPI 패키지로 제공되어 GUI 없이 Python 스크립트로 구동 가능 |
| **메쉬 모드** | `export_mesh=True` 시 SMPL-X LBS 가중치로 스킨된 메쉬를 함께 베이킹 |

### 3.5 YOLOv8x

두 단계에서 공유됩니다:
- **GVHMR**: 영상 내 인물 바운딩박스 검출 → ViTPose로 전달
- **SMPLest-X**: 가장 큰 인물 바운딩박스 선택 → 크롭 후 모델 입력

### 3.6 ViTPose-H (Multi-COCO)

GVHMR 내부에서 YOLOv8 바운딩박스 → 17-keypoint 2D 포즈 추정. 이 2D 관절이 HMR2의 입력으로 사용된다.

---

## 4. 파이프라인 상세

전체 파이프라인은 4개 단계(Stage)로 구성되며, `backend/pipeline.py`가 순차적으로 오케스트레이션한다.

### Stage 1: 3D 포즈 추정 (GVHMR)

```
입력: input.mp4
실행: docker exec mediamotion-gvhmr → demo.py --video <path> -s
출력: hmr4d_results.pt
```

- `-s` 플래그: 정적 카메라 가정 → DPVO(Dense Point-VO) 스킵으로 추론 시간 단축
- 내부적으로 YOLOv8x → ViTPose-H → HMR2 → GVHMR Transformer 순서로 동작
- 출력 `.pt`에는 `smpl_params_global`과 `smpl_params_incam` 두 좌표계 결과가 모두 포함

### Stage 2: 손/얼굴 추정 (SMPLest-X) — 선택적

```
입력: input.mp4
실행: docker exec mediamotion-smplestx → smplestx_extract.py
출력: smplestx_params.npz
조건: options.hand_tracking == true
```

- 프레임별 YOLO 탐지 → 가장 큰 인물 크롭 → SMPLest-X 추론
- 출력: `left_hand_pose [F,45]`, `right_hand_pose [F,45]`, `jaw_pose [F,3]`, `expression [F,10]`, `valid [F] bool`
- 미탐지 프레임은 `valid=False`로 기록되며, 후속 그래프팅에서 hold-last-valid로 보간

### Stage 3: 후처리 + Forward Kinematics (`pt_to_anim.py`)

```
입력: hmr4d_results.pt [+ smplestx_params.npz]
실행: docker exec mediamotion-gvhmr → pt_to_anim.py
출력: anim.npz [+ mesh.npz]
```

이 단계가 파이프라인의 핵심이며, 10개의 서브스텝을 포함한다:

1. **SMPL-X 파라미터 로드** — `global_orient`, `body_pose`, `transl`, `betas` 추출
2. **Expressive 그래프팅 (M5)** — SMPLest-X의 손/턱을 GVHMR 바디에 접합. `flat_hand_mean` 좌표계 변환 수행
3. **Fallback Pose (Despike)** — 인접 프레임 간 관절 회전 변화가 45°를 초과하는 "물리적으로 비현실적인" 프레임을 6D 회전 표현 기반 slerp 보간으로 교체
4. **Upper Body Only** — 앉은 자세/웹캠 등 하체 불필요 시 다리 관절을 단위 행렬로 고정
5. **Motion Smoothing (One-Euro)** — 모든 관절 + 루트 translation에 속도 적응형 One-Euro 필터 적용 (6D 회전 표현 위에서 동작)
6. **Forward Kinematics** — rest-pose 관절 위치 + 부모-자식 체인으로 각 관절의 글로벌 회전·위치 계산
7. **Self-Validation** — FK 결과를 SMPL-X 모델의 순전파 결과와 비교, 최대 오차 1mm 초과 시 abort
8. **Foot Locking** — 접지 프레임(높이 < 6cm, 수평 속도 < 2cm/frame)의 수평 드리프트를 루트 translation 보정으로 상쇄
9. **Physics Filter** — 발이 추정된 지면 아래로 관통하는 프레임을 y축 방향으로 들어올림
10. **Forward Alignment** — 프레임 0의 골반 방향을 정면(+Z)으로 회전 보정

### Stage 4: FBX/GLB 베이킹 (Blender)

```
입력: anim.npz [+ mesh.npz]
실행: docker exec mediamotion-blender → anim_to_fbx.py
출력: result.fbx, preview.glb
```

- Blender Armature를 `joint_names`/`parents` 기반으로 동적 생성 (22관절 또는 55관절)
- 메쉬 모드: identity rest orientation → LBS 스킨닝이 SMPL-X와 수학적으로 동일
- 스켈레톤 모드: bone head→tail 방향을 자식 관절로 지정 → 시각적으로 자연스러움
- `pose_bone.matrix`에 글로벌 변환을 직접 설정하고 매 프레임 키프레임 삽입
- **Self-Validation**: 8개 샘플 프레임에서 Blender 본 위치 vs 소스 `global_pos` 비교, 0.1mm 초과 시 abort
- Y-up → Z-up 변환 (Blender/게임 엔진 표준)
- GLB: PBR 머티리얼 설정 (metalness=0, roughness=0.6) — 뷰어에서 검게 보이는 문제 방지

---

## 5. 후처리 엔진

### 5.1 One-Euro Filter — 속도 적응형 스무딩

기존 Gaussian 커널의 문제: 느린 동작의 지터와 빠른 동작의 임팩트를 **동일한 강도로** 감쇠시켜, 펀치·댄스 등 빠른 모션이 "수중 촬영"처럼 흐릿해짐.

**One-Euro Filter (Casiez et al., CHI 2012)**:
- 느린 신호 → `min_cutoff` 유지 → 지터 강하게 제거
- 빠른 신호 → `cutoff = min_cutoff + β × |velocity|` → 컷오프가 속도에 비례하여 상승, 날카로운 동작 통과
- **Zero-Phase 구현**: 오프라인 데이터이므로 forward + backward 패스 평균으로 위상 지연 완전 상쇄

| 신호 | Raw | One-Euro | 기존 Gaussian |
|------|-----|----------|--------------|
| 정적 포즈 지터 (RMS) | 0.0056 | **0.0020 (−64%)** | 0.0022 |
| 펀치 최대 속도 | 15.0 | **11.3 (75% 보존)** | 4.2 (28% 보존) |

UI 슬라이더(0~1)는 로그 스케일로 `min_cutoff`에 매핑된다:
- 0.1 → 6.3 Hz (거의 원본), 0.55 → 0.8 Hz (강한 지터 제거), 1.0 → 0.1 Hz (매우 강함)

### 5.2 Despike (Fallback Pose)

인접 프레임 간 어떤 관절이든 45°/frame을 초과하는 회전 변화가 감지되면, 해당 프레임을 전후 유효 프레임의 6D 회전 선형 보간으로 교체한다. GVHMR의 시간적 사전이 대부분의 경우 이를 방지하지만, 극심한 가림(occlusion) 상황에서 간헐적으로 발생한다.

### 5.3 Foot Locking

- 접지 판정: 발 관절 높이 < `foot_lock_height`(기본 6cm) AND 수평 속도 < 2cm/frame
- 보정: 연속 접지 프레임의 수평 드리프트를 누적하여 루트 translation에서 역보정
- 효과: 걸음 시 "미끄러지는 발" 현상 제거

### 5.4 Physics Filter (Ground Clamp)

추정 지면(발 높이 하위 5% 분위수) 아래로 관통하는 프레임을 y축으로 들어올린다. 물리 시뮬레이션이 아닌 간단한 클램핑이지만, 시각적으로 "땅 아래에 발이 있는" 부자연스러움을 효과적으로 제거한다.

### 5.5 Forward Alignment

GVHMR의 월드 좌표 방향은 첫 프레임 카메라 방향에 의존하므로, 캐릭터가 뒷모습을 보일 수 있다. 프레임 0의 골반 forward 벡터를 기준으로 yaw 회전을 전체 시퀀스에 적용하여 캐릭터가 정면(+Z)을 향하도록 보정한다.

---

## 6. 웹 인터페이스

### 프론트엔드 (Next.js 14 + React 18 + Three.js)

| 컴포넌트 | 역할 |
|---------|------|
| `Uploader.js` | 드래그&드롭/클릭 영상 업로드 (MP4/MOV) |
| `OptionsPanel.js` | 10개 파이프라인 옵션 UI (토글, 슬라이더, 셀렉트) |
| `Progress.js` | 4단계 스테퍼 + 실시간 tqdm 진행률 바 (1.5초 폴링) |
| `Preview.js` | 완료 후 입력 영상(좌) / 3D GLB(우) 나란히 비교, Three.js OrbitControls + 골반 자동 추적 카메라 |
| `api.js` | FastAPI 백엔드 REST 클라이언트 |

### 백엔드 (FastAPI)

- `POST /api/jobs` — 영상 업로드 + 옵션 JSON → 백그라운드 스레드로 파이프라인 실행
- `GET /api/jobs/{id}` — 상태·스테이지·tqdm 진행률 조회
- `GET /api/jobs/{id}/result.fbx` — FBX 다운로드
- `GET /api/jobs/{id}/preview.glb` — GLB 프리뷰 서빙
- `GET /api/jobs/{id}/input.mp4` — 원본 영상 스트리밍
- `GET /api/health` — 헬스체크

**tqdm 로그 파싱**: 각 단계의 로그 파일 꼬리를 읽어 `n%|` 패턴으로 진행률을 실시간 추출한다.

---

## 7. 프로젝트 파일 구조

```
mediamotion/
├── backend/
│   ├── app.py                 # FastAPI 서버: 작업 생성·상태 조회·파일 서빙
│   ├── pipeline.py            # 4단계 오케스트레이터 (docker exec 호출)
│   └── Dockerfile             # Python 3.11 + docker-ce-cli + FastAPI
│
├── frontend/
│   ├── app/
│   │   ├── page.js            # 메인 SPA: 상태 머신 (upload → process → preview)
│   │   ├── layout.js          # 루트 레이아웃 + SEO 메타
│   │   └── globals.css        # 다크 테마 디자인 시스템 (glassmorphism)
│   ├── components/
│   │   ├── Uploader.js        # 드래그&드롭 영상 업로더
│   │   ├── OptionsPanel.js    # 파이프라인 옵션 패널 (10개 파라미터)
│   │   ├── Progress.js        # 4단계 스테퍼 + 실시간 진행률
│   │   └── Preview.js         # Three.js 기반 입력/3D 나란히 비교
│   ├── lib/api.js             # REST API 클라이언트
│   ├── package.json           # Next.js 14, React 18, Three.js 0.167
│   ├── next.config.js         # Three.js transpile 설정
│   └── Dockerfile             # Node 20 + Next.js dev 서버
│
├── scripts/
│   ├── pt_to_anim.py          # [핵심] HMR .pt → anim.npz 변환 + 10단계 후처리
│   ├── anim_to_fbx.py         # anim.npz → FBX/GLB (Blender bpy 구동)
│   ├── smplestx_extract.py    # 영상 → SMPLest-X 손/얼굴 추출
│   ├── pipeline_config.py     # PipelineOptions 데이터클래스 (15개 파라미터)
│   └── arrange_models.sh      # 바디 모델 파일 배치 자동화 스크립트
│
├── docker/
│   ├── Dockerfile.gvhmr       # CUDA 12.8 + PyTorch 2.7 + pytorch3d 소스빌드
│   ├── Dockerfile.smplestx    # CUDA 12.8 + PyTorch 2.7 (mmcv 불필요)
│   ├── Dockerfile.blender     # Python 3.11 + bpy 4.2 (헤드리스)
│   ├── docker-compose.yml     # 5개 서비스 오케스트레이션
│   └── entrypoint.sh          # GVHMR editable install 자동화
│
├── docs/
│   ├── M5_hands_face.md       # 손/얼굴 통합 설계 문서 + flat_hand_mean 트랩 해결
│   └── smoothing_one_euro.md  # Gaussian→One-Euro 전환 근거 + 정량 비교
│
├── GVHMR-main/                # [.gitignore] GVHMR 원본 레포 (체크포인트 포함)
├── SMPLest-X/                 # [.gitignore] SMPLest-X 원본 레포
├── SMPLer-X-main/             # [.gitignore] SMPLer-X 참조용 (미사용, 비교 목적)
├── SMPL_python_v.1.1.0/       # [.gitignore] SMPL 바디 모델 원본 배포판
├── models/                    # [.gitignore] SMPL-X 바디 모델 파일 (.npz)
├── data/                      # [.gitignore] 런타임 작업 디렉토리 (업로드/결과)
│
├── INSTALL.md                 # GVHMR 원본 설치 가이드 (참조용)
├── setup.md                   # Docker 기반 설치 + RTX 5070 포팅 가이드
├── WEB.md                     # 웹 스택 빌드·실행 가이드
└── readme.md                  # ← 이 파일
```

---

## 8. 문서 파일 가이드

### `setup.md` — Docker 설치 및 Blackwell GPU 포팅 가이드

RTX 5070(Blackwell, sm_120)에서 GVHMR을 실행하기 위한 Docker 이미지 빌드 가이드.  
CUDA 12.8 / PyTorch 2.7 포팅 과정에서 해결한 호환성 문제들을 기록:
- pytorch3d: cu128 프리빌트 휠 미존재 → `--no-build-isolation` 소스 빌드, `MAX_JOBS=4`로 WSL OOM 방지
- chumpy: `import pip` 빌드 격리 실패 → `--no-build-isolation` + GitHub master 사용
- tkinter: GVHMR의 stray `from turtle import forward` → `python3-tk` 설치
- weights_only: PyTorch≥2.6의 `torch.load(weights_only=True)` 기본값 변경 → 환경변수 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

### `INSTALL.md` — GVHMR 원본 설치 가이드

GVHMR 레포지토리의 원본 설치 문서. conda 환경 + cu121 기반이므로 본 프로젝트에서는 Docker 레시피(`setup.md`)로 대체됨. 필요한 체크포인트 목록과 다운로드 링크를 참조하는 용도.

### `WEB.md` — 웹 스택 빌드·실행 가이드

프론트엔드(Next.js) + 백엔드(FastAPI) + 워커 컨테이너 4개의 빌드·실행·접속 순서. `docker-compose.yml` 기반 원커맨드 빌드 설명.

### `docs/M5_hands_face.md` — 손/얼굴 통합 설계 문서

SMPLest-X를 선택한 이유(SMPLer-X 대비 성능·의존성 비교), 22→55 관절 그래프팅 아키텍처, `flat_hand_mean` 좌표계 트랩과 해결법, 알려진 한계(눈 시선 미지원, 표정 FBX 미반영) 기록.

### `docs/smoothing_one_euro.md` — 스무딩 기법 전환 근거

기존 Gaussian 커널이 빠른 동작을 과도하게 감쇠시키는 증상 분석, One-Euro 필터의 속도 적응 원리 설명, 정량 비교 결과(지터 제거 동등, 빠른 동작 보존 75% vs 28%), 파라미터 튜닝 가이드.

---

## 9. 환경 설정 및 실행

### 필수 요구사항

- **GPU**: NVIDIA GPU (CUDA Compute Capability ≥ 7.0 권장, Blackwell sm_120 검증 완료)
- **VRAM**: 8GB 이상 (SMPLest-X 동시 사용 시 12GB 이상 권장)
- **Docker Desktop** (WSL2 백엔드 + GPU 지원 활성화)
- **디스크**: GVHMR 체크포인트 ~2GB, SMPLest-X-Huge ~8.2GB, SMPL/SMPL-X 바디 모델 ~200MB

### 검증된 테스트 환경

| 항목 | 사양 |
|------|------|
| **CPU** | AMD Ryzen 9 9950X3D (16C/32T, 3D V-Cache) |
| **RAM** | 16GB DDR5 |
| **GPU** | NVIDIA GeForce RTX 5070 (12GB GDDR7, Blackwell sm_120) |
| **OS** | Windows 11 Pro (24H2) |
| **가상화** | Docker Desktop (WSL2 백엔드) |
| **CUDA** | 12.8 (컨테이너 내부) / 호스트 드라이버 ≥ 570.x |
| **PyTorch** | 2.7.1+cu128 (컨테이너 내부) |

> **참고**: RAM 16GB는 Docker 빌드 시 pytorch3d 소스 컴파일(`MAX_JOBS=4`)과 WSL2 메모리 할당을 감안하면 최소 수준이다. 32GB 이상을 권장하며, 16GB 환경에서는 WSL2의 `.wslconfig`에서 메모리 제한을 적절히 설정해야 빌드 중 OOM이 발생하지 않는다.

### 설치 단계

```bash
# 1. 레포 클론
git clone https://github.com/<user>/mediamotion.git
cd mediamotion

# 2. 서브 레포지토리 배치 (각각 별도 다운로드/클론 필요)
#    GVHMR-main/   → https://github.com/zju3dv/GVHMR
#    SMPLest-X/    → https://github.com/SMPLCap/SMPLest-X
#    SMPL 바디 모델 → https://smpl-x.is.tue.mpg.de/ (라이선스 동의 필요)

# 3. 바디 모델 배치
bash scripts/arrange_models.sh

# 4. GVHMR 체크포인트 다운로드 (Google Drive)
#    https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD
#    → GVHMR-main/inputs/checkpoints/{gvhmr, hmr2, vitpose, yolo}/

# 5. (선택) SMPLest-X 체크포인트 다운로드
#    https://huggingface.co/waanqii/SMPLest-X/tree/main
#    → SMPLest-X/pretrained_models/smplest_x_h/

# 6. Docker 빌드 및 실행
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d

# 7. GPU 확인
docker compose -f docker/docker-compose.yml exec gvhmr \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 웹 UI 접속

```
http://localhost:5000
```

1. 영상 업로드 (MP4/MOV)
2. 옵션 조정 (스무딩 강도, 발 고정, 손가락 추적 등)
3. "모션 캡처 생성" 클릭
4. 실시간 진행률 확인
5. 완료 후 입력/3D 나란히 비교 프리뷰
6. FBX 다운로드

### CLI 직접 실행

```bash
# Stage 1: GVHMR 추론
docker compose -f docker/docker-compose.yml exec gvhmr \
  python tools/demo/demo.py --video <입력영상> -s

# Stage 3: 후처리 + FK
docker compose -f docker/docker-compose.yml exec gvhmr \
  python /app/scripts/pt_to_anim.py \
    --pt outputs/demo/<이름>/hmr4d_results.pt \
    --out outputs/demo/<이름>/anim.npz

# Stage 4: FBX/GLB 베이킹
docker compose -f docker/docker-compose.yml exec blender \
  python /work/scripts/anim_to_fbx.py \
    --anim /work/outputs/demo/<이름>/anim.npz \
    --fbx /work/outputs/demo/<이름>/result.fbx \
    --glb /work/outputs/demo/<이름>/preview.glb
```

---

## 10. 알려진 한계와 향후 과제

### 현재 한계

| 항목 | 설명 |
|------|------|
| **눈 시선(Eye Gaze)** | SMPLest-X가 내부적으로 zero eye pose를 사용하여 예측 불가 |
| **표정(Expression)** | blendshape 계수는 추출·저장되나, FBX shape-key export는 미구현 |
| **손가락 정확도** | 전신 원거리 촬영 시 손이 수 픽셀 → 정확도 저하 (단일 뷰 한계) |
| **다중 인물** | 현재 가장 큰 바운딩박스의 1인만 처리 |
| **동적 카메라** | DPVO 스킵 상태이므로 카메라가 크게 움직이는 영상은 부정확 |
| **실시간 처리** | GVHMR 추론 + 오버레이 렌더링이 분 단위 소요 (비실시간) |

### 향후 과제

- GVHMR `--no-render` 패스트 패스 구현 (오버레이 렌더링 스킵)
- DPVO 통합으로 동적 카메라 지원
- Expression → FBX shape-key export
- 다중 인물 동시 추적
- WebSocket 기반 실시간 진행률 (현재 HTTP 폴링)
- Mixamo 호환 리깅으로 자동 리타게팅

---

## 11. 라이선스와 참고 문헌

### 핵심 의존 프로젝트

| 프로젝트 | 라이선스 | 용도 |
|---------|---------|------|
| [GVHMR](https://github.com/zju3dv/GVHMR) | 비상업적 | 3D 전신 포즈 추정 |
| [SMPLest-X](https://github.com/SMPLCap/SMPLest-X) | Apache 2.0 | 손/얼굴 추정 |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | 비상업적 (별도 라이선스) | 파라메트릭 바디 모델 |
| [Blender / bpy](https://www.blender.org/) | GPL-2.0 | FBX/GLB 베이킹 |
| [Three.js](https://threejs.org/) | MIT | 웹 3D 프리뷰 |

### 학술 참고 문헌

| # | 논문 제목 | 저자 | 발표처 | 연도 |
|---|----------|------|--------|------|
| 1 | World-Grounded Human Motion Recovery via Gravity-View Augmented Generation | Liang, Zehong 외 7인 | SIGGRAPH Asia 2024 | 2024 |
| 2 | SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation | Cai, Zhongang 외 | IEEE TPAMI | 2025 |
| 3 | 1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems | Casiez, Géry; Roussel, Nicolas; Vogel, Daniel | CHI | 2012 |

<details>
<summary>BibTeX 인용 (논문 작성 시 사용)</summary>

```bibtex
@inproceedings{liang2024gvhmr,
  title     = {World-Grounded Human Motion Recovery via Gravity-View Augmented Generation},
  author    = {Liang, Zehong and Xue, Jianan and Li, Lei and Lu, Haomin and Dong, Jinglong and Li, Shuai and Loy, Chen Change and Fan, Haoqian},
  booktitle = {SIGGRAPH Asia 2024},
  year      = {2024}
}

@article{cai2024smplest,
  title   = {SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation},
  author  = {Cai, Zhongang and others},
  journal = {IEEE TPAMI},
  year    = {2025}
}

@inproceedings{casiez2012oneeuro,
  title     = {1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems},
  author    = {Casiez, G{\'e}ry and Roussel, Nicolas and Vogel, Daniel},
  booktitle = {CHI},
  year      = {2012}
}
```

</details>

---

> **Note**: SMPL/SMPL-X 바디 모델과 GVHMR 체크포인트는 라이선스 제약으로 본 레포지토리에 포함되지 않는다. 설치 시 각 프로젝트의 라이선스에 동의한 후 별도 다운로드가 필요하다.
