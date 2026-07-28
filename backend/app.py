"""MediaMotion 백엔드 (FastAPI). 업로드를 처리하고, 파이프라인을 백그라운드
작업으로 실행하며, 진행률 및 결과 파일(FBX / 프리뷰 GLB / 원본 영상)을 서빙한다.
"""
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import pipeline

DATA_ROOT = Path("/data/jobs")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MediaMotion")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 인메모리 작업 저장소. <job>/job.json에 미러링하여 영속성을 보장한다
JOBS: dict[str, dict] = {}


def _save(job: dict):
    JOBS[job["id"]] = job
    (DATA_ROOT / job["id"] / "job.json").write_text(json.dumps(job, indent=2))


def _run_job(job_id: str, job_dir: Path):
    job = JOBS[job_id]
    job["status"] = "running"
    _save(job)

    def set_stage(key, label):
        job["stage"], job["stage_label"] = key, label
        _save(job)

    try:
        pipeline.run(job_id, job_dir, set_stage)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)
    job["finished_at"] = time.time()
    _save(job)


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...), options: str = Form("{}")):
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with open(job_dir / "input.mp4", "wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        opts = json.loads(options)
    except json.JSONDecodeError:
        opts = {}
    (job_dir / "options.json").write_text(json.dumps(opts, indent=2))

    job = {
        "id": job_id, "status": "queued", "stage": "queued",
        "stage_label": "Queued", "options": opts, "error": None,
        "created_at": time.time(), "filename": video.filename,
    }
    _save(job)
    threading.Thread(target=_run_job, args=(job_id, job_dir), daemon=True).start()
    return {"id": job_id}


STAGE_LOG = {"gvhmr": "1_gvhmr.log", "hands": "2_hands.log",
             "pose": "3_pose.log", "retarget": "4_retarget.log"}
_PCT = re.compile(r"(\d+)%\|")
_ETA = re.compile(r"<(\d+:\d+)")
_DESC = re.compile(r"^\s*([A-Za-z][\w ]*?):")


def parse_progress(job_id: str, stage: str):
    """현재 단계의 로그 꼬리를 읽어 최신 tqdm 진행률 바를 추출한다."""
    fn = STAGE_LOG.get(stage)
    if not fn:
        return None
    logf = DATA_ROOT / job_id / "logs" / fn
    if not logf.exists():
        return None
    try:
        data = logf.read_bytes()[-8000:].decode("utf-8", "replace")
    except OSError:
        return None
    for tok in reversed(re.split(r"[\r\n]", data)):
        m = _PCT.search(tok)
        if m:
            eta = _ETA.search(tok)
            desc = _DESC.match(tok)
            return {
                "percent": int(m.group(1)),
                "eta": eta.group(1) if eta else None,
                "desc": desc.group(1).strip() if desc else "",
                "line": tok.strip()[:140],
            }
    return None


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        jf = DATA_ROOT / job_id / "job.json"
        if not jf.exists():
            raise HTTPException(404, "job not found")
        job = json.loads(jf.read_text())
    if job.get("status") == "running":
        job = {**job, "progress": parse_progress(job_id, job.get("stage"))}
    return JSONResponse(job)


def _file(job_id: str, name: str, media: str, download=False):
    p = DATA_ROOT / job_id / name
    if not p.exists():
        raise HTTPException(404, f"{name} not ready")
    return FileResponse(p, media_type=media,
                        filename=name if download else None)


@app.get("/api/jobs/{job_id}/input.mp4")
def input_video(job_id: str):
    return _file(job_id, "input.mp4", "video/mp4")


@app.get("/api/jobs/{job_id}/preview.glb")
def preview_glb(job_id: str):
    return _file(job_id, "preview.glb", "model/gltf-binary")


@app.get("/api/jobs/{job_id}/result.fbx")
def result_fbx(job_id: str):
    return _file(job_id, "result.fbx", "application/octet-stream", download=True)


@app.get("/api/health")
def health():
    return {"ok": True}
