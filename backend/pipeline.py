"""작업 오케스트레이션. 백엔드 컨테이너가 docker CLI + docker.sock을 보유하며,
`docker exec`로 워커 컨테이너를 구동한다. 모든 컨테이너는 /data 볼륨을 공유한다.

단계:
  1. gvhmr   : demo.py 영상 -> hmr4d_results.pt
  2. hands   : (선택, hand_tracking) smplestx_extract.py 영상 -> smplestx_params.npz
  3. pose    : pt_to_anim.py 결과 [+expressive] -> anim.npz (+ mesh.npz)
  4. retarget: anim_to_fbx.py anim.npz -> result.fbx + preview.glb
"""
import json
import subprocess
from pathlib import Path

GVHMR = "mediamotion-gvhmr"
SMPLESTX = "mediamotion-smplestx"
BLENDER = "mediamotion-blender"
DATA = "/data/jobs"  # 워커 컨테이너 *내부* 경로


def _exec(container: str, inner_cmd: str, log_file: Path):
    """워커 컨테이너 내부에서 셸 명령을 실행하고, 출력을 로그 파일에 기록한다."""
    cmd = ["docker", "exec", container, "bash", "-lc", inner_cmd]
    with open(log_file, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log_file.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"[{container}] step failed (exit {proc.returncode}):\n{tail}")


def run(job_id: str, host_job_dir: Path, set_stage):
    """host_job_dir은 호스트/백엔드에서 파일이 위치하는 경로이다. 워커는 /data/jobs/<id>를 참조한다."""
    d = f"{DATA}/{job_id}"
    logs = host_job_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1"

    try:
        opts = json.loads((host_job_dir / "options.json").read_text())
    except (OSError, json.JSONDecodeError):
        opts = {}
    hands = bool(opts.get("hand_tracking"))

    # 1) GVHMR 추론 (정적 카메라; DPVO 스킵)
    set_stage("gvhmr", "3D pose estimation (GVHMR)")
    _exec(GVHMR,
          f"cd /app/GVHMR && {env} python tools/demo/demo.py "
          f"--video {d}/input.mp4 --output_root {d}/gvhmr -s",
          logs / "1_gvhmr.log")

    # 2) 선택적 손/얼굴 추정 (SMPLest-X)
    expressive = ""
    if hands:
        set_stage("hands", "Hand/face estimation (SMPLest-X)")
        _exec(SMPLESTX,
              f"cd /app/SMPLest-X && {env} python /app/scripts/smplestx_extract.py "
              f"--video {d}/input.mp4 --out {d}/smplestx_params.npz",
              logs / "2_hands.log")
        expressive = f" --expressive {d}/smplestx_params.npz"

    # 3) 파라미터 -> anim.npz 후처리 + 옵션 적용
    set_stage("pose", "Post-processing (smoothing / foot-lock)")
    _exec(GVHMR,
          f"cd /app/GVHMR && {env} python /app/scripts/pt_to_anim.py "
          f"--pt {d}/gvhmr/input/hmr4d_results.pt --out {d}/anim.npz "
          f"--options {d}/options.json{expressive}",
          logs / "3_pose.log")

    # 4) anim.npz -> FBX + GLB 변환
    set_stage("retarget", "Baking FBX + preview (Blender)")
    _exec(BLENDER,
          f"cd /work && python /work/scripts/anim_to_fbx.py "
          f"--anim {d}/anim.npz --fbx {d}/result.fbx --glb {d}/preview.glb",
          logs / "4_retarget.log")

    set_stage("done", "Complete")
