# M5 — SMPLest-X를 통한 손/얼굴 통합 (구현 완료, 첫 빌드+실행 대기 중)

## 모델 선택: SMPLer-X 대신 SMPLest-X를 채택한 이유
- [SMPLest-X (TPAMI 2025)](https://github.com/SMPLCap/SMPLest-X)는 SMPLer-X의 후속 모델이다:
  SMPLer-X 대비 **hand-PA-MPE 15% 감소 / hand-MPE 13% 감소**를 달성하였으며, 아키텍처도 더 단순하다.
- 결정적으로, SMPLest-X의 requirements에는 **mmcv/mmdet/mmpose가 없다** — M5 도입을 미뤘던 의존성 지옥(dependency hell) 문제가 여기서는 존재하지 않는다. 포팅 난이도는 GVHMR과 동급이다(이미 해결된 레시피를 재활용할 수 있다).

## 아키텍처 (구현된 구조)
```
video ─► GVHMR (시간적으로 안정된 바디 22관절)     ─┐
     └─► SMPLest-X (프레임별 손 + 턱 + 표정)      ─┤ pt_to_anim.py 내부에서 그래프팅
                                                    ├─► 55-관절 FK (+ 검증)
                                                    ├─► 모든 관절에 대해 스무딩
                                                    └─► anim.npz ─► Blender ─► FBX/GLB
```
- 그래프팅은 GVHMR의 바디를 유지하면서, SMPLest-X로부터 `left/right_hand_pose`(45+45), `jaw_pose`(3)만을 접합한다. 미탐지 프레임은 hold-last-valid로 보간한다.
- **flat_hand_mean 트랩 (해결됨):** SMPLest-X는 *non-flat* hand mean 기준으로 손 포즈를 예측하지만, 본 프로젝트의 FK 모델은 `flat_hand_mean=True`를 사용한다. `pt_to_anim`에서 `model.left/right_hand_mean`을 더하여 좌표계를 변환한다 — 이 변환 없이는 손가락이 절반만 쥐어진 부정확한 자세가 된다.
- 6D 회전 스무딩은 이제 손/턱 관절까지 포함한다 (SMPLest-X는 프레임 단위 추정이므로 지터가 발생하며, 이 스무딩을 통해 GVHMR 수준의 시간적 안정성을 복원한다). 이후 필터 자체가 Gaussian에서 One-Euro로 변경되었다. 자세한 내용은 [smoothing_one_euro.md](smoothing_one_euro.md)를 참조한다.
- `anim_to_fbx.py`는 변경이 필요하지 않았다 — joint_names/parents에 대해 범용적으로 동작하므로, 아마추어(armature)가 자동으로 55개 본(손가락, 턱, 눈 관절)으로 확장된다.

## 솔직한 한계
- **눈 시선(Eye Gaze): 이 모델로는 불가능하다.** SMPLest-X는 내부적으로 zero eye pose를 사용한다 (`models/SMPLest_X.py`의 `zero_pose`). UI에서 해당 토글은 비활성화되어 있으며 사유가 표시된다.
- `expression` (10개의 blendshape 계수)은 추출되어 `smplestx_params.npz`에 저장되지만, **FBX로는 내보내지 않는다** — Blender에서 shape-key export를 구현해야 한다 (향후 과제).
- 품질은 화면 내 인물/손의 크기에 따라 결정된다. 전신 원거리 촬영 시 손가락이 수 픽셀에 불과하여 정확도가 떨어진다. 이는 단일 뷰(single-view)의 본질적 한계이며 버그가 아니다.

## 관련 파일
| 파일 | 역할 |
|------|------|
| `docker/Dockerfile.smplestx` | cu128/torch2.7 포팅 (GVHMR 레시피 재활용: chumpy no-build-isolation, numpy 1.23.5) |
| `scripts/smplestx_extract.py` | 영상 → 프레임별 `smplestx_params.npz` (최대 인물 YOLO + 모델 추론) |
| `scripts/pt_to_anim.py` | `--expressive` 플래그: 그래프팅 + 55-관절 FK + 검증 |
| `backend/pipeline.py` | `hand_tracking` 옵션 활성화 시 조건부 `hands` 단계 실행 |
| compose `smplestx` 서비스 | 레포, 스크립트, /data, 바디 모델(`../models`), GVHMR의 yolov8x.pt를 마운트 |

## 셋업 (사용자)
1. **SMPLest-X-Huge** (8.2GB)를 https://huggingface.co/waanqii/SMPLest-X/tree/main 에서 다운로드하여
   `SMPLest-X/pretrained_models/smplest_x_h/` 경로에 배치한다 → 반드시
   `smplest_x_h.pth.tar` **및** `config_base.py`가 포함되어 있어야 한다 (HuggingFace 레포에 없을 경우,
   `SMPLest-X/configs/config_smplest_x_h.py`를 해당 이름으로 복사한다).
2. `docker compose -f docker/docker-compose.yml build smplestx`
3. `docker compose -f docker/docker-compose.yml up -d smplestx`
4. 웹 UI에서 **손가락 추적**을 활성화하고 작업을 실행한다 (인물이 카메라에 가까울수록 결과가 좋다).
