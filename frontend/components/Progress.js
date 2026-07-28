"use client";

const STAGES = [
  { key: "gvhmr", label: "3D 포즈 추정 (GVHMR)" },
  { key: "hands", label: "손·얼굴 추정 (SMPLest-X)", optional: true },
  { key: "pose", label: "후처리 (스무딩 · 발고정)" },
  { key: "retarget", label: "FBX + 프리뷰 베이킹" },
];
const ORDER = ["queued", "gvhmr", "hands", "pose", "retarget", "done"];

export default function Progress({ job }) {
  const cur = ORDER.indexOf(job?.stage || "queued");
  const p = job?.progress;
  const stages = STAGES.filter((s) => !s.optional || job?.options?.hand_tracking);
  return (
    <div className="card">
      <h2>진행 상황</h2>
      <div className="sub">{job?.stage_label || "대기 중"}</div>
      <div className="steps">
        {stages.map((s) => {
          const idx = ORDER.indexOf(s.key);
          const state = job?.status === "done" || idx < cur ? "done" : idx === cur ? "active" : "pending";
          return (
            <div key={s.key}>
              <div className={`step ${state}`}>
                <div className="dot">
                  {state === "done" ? "✓" : state === "active" ? <span className="spin" /> : idx}
                </div>
                <div className="label">{s.label}</div>
              </div>
              {state === "active" && p && (
                <div className="pbar-wrap">
                  <div className="pbar"><div className="pbar-fill" style={{ width: `${p.percent}%` }} /></div>
                  <div className="pbar-meta">
                    <span>{p.desc || "처리"} · {p.percent}%</span>
                    <span>{p.eta ? `남은 시간 ~${p.eta}` : "…"}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {job?.status === "running" && !p && (
        <div className="muted" style={{ marginTop: 10 }}>진행률 집계 중… (첫 프레임 로딩)</div>
      )}
    </div>
  );
}
