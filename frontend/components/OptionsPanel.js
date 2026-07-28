"use client";

function Toggle({ label, hint, value, onChange, disabled }) {
  return (
    <div className="opt">
      <div>
        <label>{label}</label>
        {hint && <div className="hint">{hint}</div>}
      </div>
      <label className="switch">
        <input type="checkbox" checked={!!value} disabled={disabled}
          onChange={(e) => onChange(e.target.checked)} />
        <span className="slider" />
      </label>
    </div>
  );
}

function Range({ label, hint, value, onChange, disabled }) {
  return (
    <div className="opt">
      <div>
        <label>{label}</label>
        {hint && <div className="hint">{hint}</div>}
      </div>
      <input className="range" type="range" min="0" max="1" step="0.05"
        value={value} disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

function Select({ label, hint, value, onChange, disabled, choices }) {
  return (
    <div className="opt">
      <div>
        <label>{label}</label>
        {hint && <div className="hint">{hint}</div>}
      </div>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {choices.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
      </select>
    </div>
  );
}

export default function OptionsPanel({ options, setOptions, disabled }) {
  const set = (k) => (v) => setOptions({ ...options, [k]: v });
  return (
    <div className="card">
      <h2>옵션</h2>
      <div className="sub">값수정</div>

      <Range label="모션 스무딩" hint="떨림(jitter) 완화 — 빠른 동작은 자동 보존 (One-Euro)"
        value={options.motion_smoothing} onChange={set("motion_smoothing")} disabled={disabled} />
      <Toggle label="발 고정 (Foot Lock)" hint="땅에서 발 미끄러짐 제거"
        value={options.foot_locking} onChange={set("foot_locking")} disabled={disabled} />
      <Toggle label="물리 필터" hint="발이 땅을 뚫지 않도록 클램프"
        value={options.physics_filter} onChange={set("physics_filter")} disabled={disabled} />
      <Toggle label="상체만" hint="앉은 자세 / 웹캠 — 다리 고정"
        value={options.upper_body_only} onChange={set("upper_body_only")} disabled={disabled} />
      <Toggle label="머리 회전" hint="목·머리 방향 추적 (GVHMR 내장)"
        value={options.head_rotation} onChange={set("head_rotation")} disabled={disabled} />
      <Toggle label="폴백 포즈" hint="탐지 실패 프레임 보간"
        value={options.fallback_pose} onChange={set("fallback_pose")} disabled={disabled} />
      <Toggle label="미리보기 메쉬" hint="뼈대 대신 사람 형태로 프리뷰"
        value={options.export_mesh} onChange={set("export_mesh")} disabled={disabled} />

      <Toggle label="손가락 추적" hint="SMPLest-X · 인물이 화면에 클수록 정확 (처리시간 증가)"
        value={options.hand_tracking} onChange={set("hand_tracking")} disabled={disabled} />
      <Select label="시작 포즈" hint="프레임 0에 기준 자세 삽입 — 리타게팅 정렬용"
        value={options.start_pose} onChange={set("start_pose")} disabled={disabled}
        choices={[["none", "없음"], ["T", "T-포즈"], ["A", "A-포즈"]]} />
      <Toggle label="정면 정렬" hint="시작 시 캐릭터가 정면을 보도록 전체 회전 보정"
        value={options.align_forward} onChange={set("align_forward")} disabled={disabled} />
    </div>
  );
}
