"use client";
import { useRef, useState } from "react";

export default function Uploader({ file, setFile, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const pick = (f) => { if (f && f.type.startsWith("video/")) setFile(f); };

  return (
    <div className="card">
      <h2>영상 업로드</h2>
      <div className="sub">단일 카메라 영상 하나. 사람이 프레임에 잘 담길수록 정확합니다.</div>
      {!file ? (
        <div
          className={`drop${drag ? " drag" : ""}`}
          onClick={() => !disabled && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
        >
          <div className="big">영상을 끌어다 놓거나 클릭</div>
          <div className="small">MP4 / MOV · 짧을수록 빠릅니다</div>
          <input ref={inputRef} type="file" accept="video/*" hidden
                 onChange={(e) => pick(e.target.files?.[0])} />
        </div>
      ) : (
        <div className="filechip">
          <span>🎬</span>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {file.name}
          </span>
          {!disabled && <span className="x" onClick={() => setFile(null)}>✕</span>}
        </div>
      )}
    </div>
  );
}
