"use client";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { fileUrl } from "../lib/api";

const fmt = (t) => {
  if (!isFinite(t)) return "0:00.0";
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

export default function Preview({ jobId }) {
  const videoRef = useRef(null);
  const mountRef = useRef(null);
  const mixerRef = useRef(null);
  const clipDurRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [volume, setVolume] = useState(0.5);
  const [muted, setMuted] = useState(false);
  const prevVolRef = useRef(0.5);

  const videoUrl = fileUrl(jobId, "input.mp4");
  const glbUrl = fileUrl(jobId, "preview.glb");

  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth, h = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    camera.position.set(0, 1.1, 3.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    // 이미지 기반 환경 조명 — PBR 메터리얼을 에셋 없이도 읽을 수 있게 한다
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.add(new THREE.HemisphereLight(0xbfe3ff, 0x223, 1.6));
    const dir = new THREE.DirectionalLight(0xffffff, 2.4);
    dir.position.set(2, 4, 3); scene.add(dir);
    const grid = new THREE.GridHelper(10, 20, 0x22d3ee, 0x1a2230);
    grid.material.opacity = 0.25; grid.material.transparent = true; scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 0.9, 0);

    let raf;
    let trackBone = null;           // 카메라가 추적할 본 (골반)
    const followPos = new THREE.Vector3();
    const tmp = new THREE.Vector3();
    const loader = new GLTFLoader();
    loader.load(glbUrl, (gltf) => {
      // 이전 익스포트는 glTF 기본 메터리얼(metallic=1)을 포함할 수 있다 → 검게 렌더링됨;
      // 무조건 무광택 룩을 강제한다
      gltf.scene.traverse((o) => {
        if (o.isMesh && o.material) {
          o.material.metalness = 0.0;
          o.material.roughness = 0.65;
          o.frustumCulled = false;   // 스킨드 메쉬는 rest bbox에서 멀리 이동한다
        }
      });
      scene.add(gltf.scene);
      if (gltf.animations?.length) {
        const mixer = new THREE.AnimationMixer(gltf.scene);
        mixer.clipAction(gltf.animations[0]).play();
        mixer.setTime(0);           // 카메라 피팅 *전에* 프레임 0 포즈를 설정한다
        mixerRef.current = mixer;
        clipDurRef.current = gltf.animations[0].duration;
      }
      gltf.scene.updateMatrixWorld(true);
      // POSED 스켈레톤에 카메라를 맞춘다 (rest-pose bbox는 원점에 있으나
      // 월드 기반 애니메이션은 캐릭터를 수 미터 떨어진 곳에 배치한다)
      const pts = [];
      gltf.scene.traverse((o) => {
        if (o.isBone) {
          pts.push(o.getWorldPosition(new THREE.Vector3()));
          if (!trackBone && o.name.toLowerCase().includes("pelvis")) trackBone = o;
        }
      });
      const box = pts.length
        ? new THREE.Box3().setFromPoints(pts)
        : new THREE.Box3().setFromObject(gltf.scene);
      const c = box.getCenter(new THREE.Vector3());
      const size = Math.max(box.getSize(new THREE.Vector3()).length(), 1.5);
      controls.target.copy(c);
      followPos.copy(c);
      camera.position.set(c.x, c.y + size * 0.15, c.z + size * 1.15);
      camera.near = size / 100; camera.far = size * 20; camera.updateProjectionMatrix();
    });

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const v = videoRef.current;
      if (mixerRef.current && v) {
        const d = clipDurRef.current || 1;
        mixerRef.current.setTime(Math.min(v.currentTime, d));
      }
      // 골반을 추적하여 걷거나 춤추는 캐릭터가 프레임 안에 머물도록 한다
      if (trackBone) {
        trackBone.getWorldPosition(tmp);
        followPos.lerp(tmp, 0.08);
        const delta = tmp.copy(followPos).sub(controls.target);
        controls.target.add(delta);
        camera.position.add(delta);
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const nw = mount.clientWidth, nh = mount.clientHeight;
      camera.aspect = nw / nh; camera.updateProjectionMatrix(); renderer.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose(); renderer.dispose();
      if (renderer.domElement.parentNode) mount.removeChild(renderer.domElement);
    };
  }, [glbUrl]);

  const togglePlay = () => {
    const v = videoRef.current; if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); } else { v.pause(); setPlaying(false); }
  };
  const scrub = (e) => {
    const v = videoRef.current; if (!v) return;
    v.currentTime = parseFloat(e.target.value); setT(v.currentTime);
  };
  const changeVolume = (e) => {
    const val = parseFloat(e.target.value);
    const v = videoRef.current; if (!v) return;
    v.volume = val; setVolume(val);
    if (val === 0) { setMuted(true); } else { setMuted(false); prevVolRef.current = val; }
  };
  const toggleMute = () => {
    const v = videoRef.current; if (!v) return;
    if (muted) {
      const restore = prevVolRef.current || 0.5;
      v.volume = restore; setVolume(restore); setMuted(false);
    } else {
      prevVolRef.current = volume; v.volume = 0; setVolume(0); setMuted(true);
    }
  };
  const volIcon = muted || volume === 0 ? "🔇" : volume < 0.4 ? "🔈" : volume < 0.75 ? "🔉" : "🔊";

  return (
    <div className="card preview">
      <h2>미리보기</h2>
      <div className="sub">왼쪽 원본 · 오른쪽 3D 결과 (드래그로 회전 · 스크롤로 확대)</div>
      <div className="split">
        <div className="pane">
          <span className="plabel">Input</span>
          <video
            ref={videoRef} src={videoUrl} playsInline
            onLoadedMetadata={(e) => { e.target.volume = volume; setDur(e.target.duration); }}
            onTimeUpdate={(e) => setT(e.target.currentTime)}
            onEnded={() => setPlaying(false)}
          />
        </div>
        <div className="pane">
          <span className="plabel">3D · SMPL-X</span>
          <div ref={mountRef} style={{ width: "100%", height: "100%" }} />
        </div>
      </div>
      <div className="transport">
        <button className="playbtn" onClick={togglePlay}>{playing ? "❚❚" : "▶"}</button>
        <input className="scrub" type="range" min="0" max={dur || 0} step="0.01"
               value={t} onChange={scrub} />
        <span className="time">{fmt(t)} / {fmt(dur)}</span>
        <div className="vol-group">
          <button className="vol-btn" onClick={toggleMute} title={muted ? "음소거 해제" : "음소거"}>{volIcon}</button>
          <input className="vol-slider" type="range" min="0" max="1" step="0.01"
                 value={volume} onChange={changeVolume} />
        </div>
      </div>
    </div>
  );
}
