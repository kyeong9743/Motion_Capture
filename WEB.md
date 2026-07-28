# MediaMotion Web (M3) — 실행 가이드

전체 스택: **프론트엔드 (Next.js)** → **백엔드 (FastAPI)** 구조로 이루어지며, Docker 소켓을 통해 **gvhmr** 및 **blender** 워커 컨테이너를 구동한다. 모든 컨테이너는 `./data` 볼륨을 공유한다.

```
[ :5000 frontend ]  업로드 · 옵션 · 진행률 · 분할 화면 미리보기 · FBX 다운로드
        │  http
[ :8010 backend  ]  FastAPI, 백그라운드 작업, 워커에 docker exec 호출 (8000번 포트는 deepfake-web이 사용 중)
        │  docker.sock
[ gvhmr ]  demo.py → hmr4d_results.pt → pt_to_anim.py → anim.npz(+mesh.npz)
[ blender ] anim_to_fbx.py → result.fbx + preview.glb
```

## 빌드 및 실행
```bash
# 워커 컨테이너가 먼저 존재해야 한다 (이전에 이미 빌드 완료됨):
docker compose -f docker/docker-compose.yml up -d gvhmr blender

# 2개의 새로운 서비스 빌드:
docker compose -f docker/docker-compose.yml build backend frontend
docker compose -f docker/docker-compose.yml up -d backend frontend
```
**http://localhost:5000**에 접속한다. 영상을 업로드하고 옵션을 선택한 뒤 "모션 캡처 생성"을 클릭한다.
스테퍼(stepper) 진행 상황을 확인하며, 완료되면 분할 화면 미리보기가 재생되고(좌측 입력 영상 / 우측 3D),
"⬇ FBX 다운로드" 버튼을 통해 `result.fbx`를 얻을 수 있다.

## 참고 사항 및 예상되는 한계 (엔드투엔드 미테스트 항목)
- 파이프라인의 **개별 단계(stage)들은 검증이 완료**되었으나, 웹 **오케스트레이션 연결**(백엔드에서 워커로의 `docker exec` 호출 및 Windows/WSL 환경에서의 docker.sock 마운트)은 최초 실행 시 확인이 필요하다. 특정 단계가 실패할 경우 `data/jobs/<id>/logs/*.log`에서 상세 로그를 확인할 수 있다.
- GVHMR은 추론 과정에서 여전히 오버레이 영상을 렌더링하므로 속도가 느리다. 추후 `--no-render` 옵션을 통한 고속 처리(fast path) 최적화가 권장된다.
- `docker.sock` 마운트는 Docker Desktop + WSL2 환경에서 `/var/run/docker.sock`을 노출한다고 가정한다.
- 첫 번째 작업은 GVHMR 초기화 및 렌더링으로 인해 느리게 진행된다. 진행률은 1.5초마다 폴링(poll)된다.
- 미리보기에서 기본 골격(skeleton) 대신 스킨(skinned)이 적용된 바디를 보려면 **미리보기 메쉬** 옵션을 활성화한다.
