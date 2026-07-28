"use client";
import { useEffect, useState } from "react";
import Uploader from "../components/Uploader";
import OptionsPanel from "../components/OptionsPanel";
import Progress from "../components/Progress";
import Preview from "../components/Preview";
import { createJob, getJob, fileUrl } from "../lib/api";

const DEFAULTS = {
  fps: 30, space: "global", source_speed: 1.0,
  motion_smoothing: 0.4, foot_locking: true, foot_lock_height: 0.06,
  physics_filter: true, upper_body_only: false,
  hand_tracking: false, head_rotation: true,
  fallback_pose: true, zup: true, export_mesh: false, start_pose: "none",
  align_forward: true,
};

export default function Home() {
  const [file, setFile] = useState(null);
  const [options, setOptions] = useState(DEFAULTS);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  const busy = jobId && job?.status !== "done" && job?.status !== "error";

  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (!stop) setJob(j);
        if (j.status === "done" || j.status === "error") return;
      } catch (e) { /* 폴링 유지 */ }
      if (!stop) setTimeout(tick, 1500);
    };
    tick();
    return () => { stop = true; };
  }, [jobId]);

  const generate = async () => {
    setError(null); setJob(null);
    try {
      const { id } = await createJob(file, options);
      setJobId(id);
    } catch (e) { setError(e.message); }
  };
  const reset = () => { setJobId(null); setJob(null); setFile(null); setError(null); };

  return (
    <div className="container">
      <header className="topbar">
        <div className="logo">M</div>
        <div className="brand">Media<span>Motion</span></div>
        <div className="tag">local · GVHMR → SMPL-X → FBX</div>
      </header>

      {error && <div className="banner err">에러: {error}</div>}

      <div className="grid">
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <Uploader file={file} setFile={setFile} disabled={!!busy} />
          <OptionsPanel options={options} setOptions={setOptions} disabled={!!busy} />
          {job?.status === "done" ? (
            <div className="row">
              <a className="btn" href={fileUrl(jobId, "result.fbx")}>⬇ FBX 다운로드</a>
              <button className="btn ghost" onClick={reset}>새 작업</button>
            </div>
          ) : (
            <button className="btn" disabled={!file || !!busy} onClick={generate}>
              {busy ? "처리 중…" : "모션 캡처 생성"}
            </button>
          )}
        </div>

        <div>
          {!jobId && (
            <div className="card">
              <h2>시작하기</h2>
              <div className="sub">영상을 올리고 옵션을 고른 뒤 “모션 캡처 생성”을 누르세요.</div>
              <p className="muted">
                단일 카메라 영상 하나로 3D 휴머노이드 애니메이션(FBX)을 만듭니다.
                가려진 팔 등은 GVHMR의 시간적 모션 사전(temporal prior)이 이어서 복원하고,
                후처리에서 떨림과 발 미끄러짐을 정리합니다.
              </p>
            </div>
          )}
          {jobId && job?.status !== "done" && <Progress job={job} />}
          {job?.status === "error" && (
            <div className="banner err" style={{ marginTop: 16 }}>
              파이프라인 실패: {job.error}
            </div>
          )}
          {job?.status === "done" && <Preview jobId={jobId} />}
        </div>
      </div>
    </div>
  );
}
