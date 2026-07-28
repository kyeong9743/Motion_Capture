"""
단계: GVHMR hmr4d_results.pt  ->  anim.npz  (M2 후처리 + M4 옵션,
선택적으로 SMPLest-X 손/얼굴을 그래프팅하여 55-관절 리그를 구성한다 — M5).

이 단계의 파이프라인:
  1. GVHMR SMPL-X 바디 파라미터 읽기 (global_orient, body_pose, transl)
  2. [--expressive] SMPLest-X 손/턱을 바디에 그래프팅 (미탐지 프레임은 hold-last 처리)
  3. [fallback_pose]   물리적으로 비현실적인 바디 프레임 제거 (slerp 보간)
  4. [upper_body_only] 다리를 rest 포즈로 고정
  5. [motion_smoothing] 회전 안전 속도 적응형 One-Euro 필터 (6D 표현)
     모든 관절 + 이동량에 적용 — 지터를 제거하고 빠른 동작은 보존한다
  6. 전방 운동학 -> 관절별 글로벌 회전 + 위치 (22 또는 55 관절)
  7. FK 결과를 SMPL-X 모델과 교차검증한다 (스무딩된 파라미터 기준)
  8. [foot_locking]    접지 발의 수평 드리프트를 루트 이동량으로 상쇄한다
  9. [physics_filter]  지면 클램프 (발이 추정 지면 아래로 내려가지 않도록)
 10. anim.npz 저장 (+ export_mesh 시 mesh.npz)

GVHMR (GPU) 컨테이너에서 실행:
  python /app/scripts/pt_to_anim.py --pt <results.pt> --out <anim.npz> \
      [--options <json>] [--expressive <smplestx_params.npz>]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import smplx
from pytorch3d.transforms import (
    axis_angle_to_matrix, matrix_to_axis_angle,
    matrix_to_rotation_6d, rotation_6d_to_matrix,
)

from pipeline_config import PipelineOptions

BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
FACE_JOINT_NAMES = ["jaw", "left_eye_smplhf", "right_eye_smplhf"]
HAND_JOINT_NAMES = [
    f"{side}_{finger}{i}"
    for side in ("left", "right")
    for finger in ("index", "middle", "pinky", "ring", "thumb")
    for i in (1, 2, 3)
]
FULL_JOINT_NAMES = BODY_JOINT_NAMES + FACE_JOINT_NAMES + HAND_JOINT_NAMES  # 55 (SMPL-X order)

NUM_BODY = len(BODY_JOINT_NAMES)          # 22
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
FOOT_JOINTS = [10, 11]


# ----------------------------- 유틸리티 ---------------------------------------
# One-Euro 필터 (Casiez et al. 2012): 속도 적응형 저역 통과. 느린 신호 ->
# 강한 스무딩 (지터 제거); 빠른 신호 -> 속도에 비례하여 컷오프 상승,
# 펀치/급정지 등은 거의 그대로 통과한다. 기존 균일 Gaussian이
# 지터와 동일하게 빠른 동작도 흐렸던 문제를 해결한다.
ONE_EURO_BETA = 1.5    # 단위 속도당 컷오프 증가량 (6D-rot 단위 & 미터 모두 O(1) 수준)
ONE_EURO_DCUT = 1.0    # Hz, 속도 추정치 자체에 대한 저역 통과


def smoothing_min_cutoff(strength: float) -> float:
    """UI 강도 0..1 -> One-Euro min_cutoff (Hz), 로그 스케일.
    0.1 -> 6.3 Hz (거의 원본), 0.55 -> 0.8 Hz, 1.0 -> 0.1 Hz (매우 강함)."""
    return 10.0 ** (1.0 - 2.0 * strength)


def _one_euro_pass(x: torch.Tensor, fps: float, min_cutoff: float) -> torch.Tensor:
    """단일 인과 패스. x: [F, C]."""
    def alpha(cutoff):
        return 1.0 / (1.0 + fps / (2.0 * np.pi * cutoff))
    a_d = alpha(ONE_EURO_DCUT)
    out = torch.empty_like(x)
    x_hat = x[0].clone()
    dx_hat = torch.zeros_like(x[0])
    out[0] = x_hat
    for i in range(1, x.shape[0]):
        dx = (x[i] - x_hat) * fps
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        a = alpha(min_cutoff + ONE_EURO_BETA * dx_hat.abs())  # 채널별 컷오프
        x_hat = a * x[i] + (1.0 - a) * x_hat
        out[i] = x_hat
    return out


def smooth_time(x: torch.Tensor, fps: float, strength: float) -> torch.Tensor:
    """Zero-phase One-Euro: 순방향과 역방향 인과 패스의 평균.
    오프라인 데이터이므로 필터 지연을 상쇄할 수 있다."""
    if strength <= 1e-3:
        return x
    min_cutoff = smoothing_min_cutoff(strength)
    flat = x.reshape(x.shape[0], -1)
    fwd = _one_euro_pass(flat, fps, min_cutoff)
    bwd = _one_euro_pass(flat.flip(0), fps, min_cutoff).flip(0)
    return ((fwd + bwd) * 0.5).reshape(x.shape)


def smooth_rotations(rotmats: torch.Tensor, fps: float, strength: float) -> torch.Tensor:
    if strength <= 1e-3:
        return rotmats
    r6 = matrix_to_rotation_6d(rotmats)
    r6 = smooth_time(r6, fps, strength)
    return rotation_6d_to_matrix(r6)


def despike(rotmats: torch.Tensor, transl: torch.Tensor, deg_thresh=45.0):
    """바디 관절 스파이크 제거 (fallback_pose). rotmats:[F,J,3,3] (바디 슬라이스만)."""
    F = rotmats.shape[0]
    rel = rotmats[1:] @ rotmats[:-1].transpose(-1, -2)
    ang = matrix_to_axis_angle(rel).norm(dim=-1)
    frame_jump = ang.max(dim=1).values
    bad = torch.zeros(F, dtype=torch.bool)
    bad[1:] = frame_jump > np.deg2rad(deg_thresh)
    bad[0] = False
    n_bad = int(bad.sum())
    if n_bad == 0:
        return rotmats, transl, 0
    good_idx = torch.where(~bad)[0]
    r6 = matrix_to_rotation_6d(rotmats)
    for f in torch.where(bad)[0]:
        lo = good_idx[good_idx < f]
        hi = good_idx[good_idx > f]
        if len(lo) == 0 or len(hi) == 0:
            continue
        a, b = lo[-1].item(), hi[0].item()
        t = (f.item() - a) / (b - a)
        r6[f] = (1 - t) * r6[a] + t * r6[b]
        transl[f] = (1 - t) * transl[a] + t * transl[b]
    return rotation_6d_to_matrix(r6), transl, n_bad


def hold_last_valid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """유효하지 않은 프레임을 마지막 유효 값으로 채운다 (첫 유효 프레임 이전은 neutral)."""
    out = x.copy()
    last = None
    for f in range(len(valid)):
        if valid[f]:
            last = out[f]
        elif last is not None:
            out[f] = last
        else:
            out[f] = 0.0
    return out


def forward_kinematics(rest_J, parents, rotmats, transl):
    F, J = rotmats.shape[0], rotmats.shape[1]
    g_rot = torch.zeros((F, J, 3, 3))
    g_pos = torch.zeros((F, J, 3))
    g_rot[:, 0] = rotmats[:, 0]
    g_pos[:, 0] = rest_J[0]
    for i in range(1, J):
        p = int(parents[i])
        g_rot[:, i] = g_rot[:, p] @ rotmats[:, i]
        offset = (rest_J[i] - rest_J[p]).view(1, 3, 1)
        g_pos[:, i] = g_pos[:, p] + (g_rot[:, p] @ offset).squeeze(-1)
    g_pos = g_pos + transl[:, None, :]
    return g_rot, g_pos


def foot_lock(g_pos, opt: PipelineOptions):
    F = g_pos.shape[0]
    feet = g_pos[:, FOOT_JOINTS]
    heights = feet[..., 1]
    ground = torch.quantile(heights.reshape(-1), 0.05)
    contact = (heights - ground) < opt.foot_lock_height
    horiz = feet[..., [0, 2]]
    speed = torch.zeros(F, 2)
    speed[1:] = (horiz[1:] - horiz[:-1]).norm(dim=-1)
    contact &= speed < 0.02

    corr = torch.zeros(F, 3)
    for t in range(1, F):
        both = contact[t] & contact[t - 1]
        if both.any():
            fi = FOOT_JOINTS[int(torch.where(both)[0][0])]
            drift = g_pos[t, fi, [0, 2]] - g_pos[t - 1, fi, [0, 2]]
            corr[t, 0] = corr[t - 1, 0] - drift[0]
            corr[t, 2] = corr[t - 1, 2] - drift[1]
        else:
            corr[t] = corr[t - 1]
    return corr, ground


# ------------------------------- 메인 ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--options", default=None, help="JSON 문자열 또는 옵션 파일 경로")
    ap.add_argument("--expressive", default=None,
                    help="smplestx_params.npz (손/턱, SMPLest-X) -> 55-관절 리그")
    ap.add_argument("--model-root", default="inputs/checkpoints/body_models")
    args = ap.parse_args()

    if args.options:
        raw = Path(args.options).read_text() if Path(args.options).exists() else args.options
        opt = PipelineOptions.from_json(raw)
    else:
        opt = PipelineOptions()
    print(f"[pt_to_anim] options: {opt.to_json()}")

    pred = torch.load(args.pt, map_location="cpu")
    params = pred[f"smpl_params_{opt.space}"]
    body_pose = params["body_pose"].float()
    global_orient = params["global_orient"].float()
    transl = params["transl"].float().clone()
    betas = params["betas"].float()
    F = body_pose.shape[0]
    mean_betas = betas.mean(0, keepdim=True)
    print(f"[pt_to_anim] frames={F} space={opt.space}")

    # ---- expressive 그래프팅 (M5) ----
    use_hands = bool(args.expressive) and opt.hand_tracking
    if args.expressive and not opt.hand_tracking:
        print("[pt_to_anim] expressive npz given but hand_tracking=False -> body-only rig")
    expr10 = None
    lhand = rhand = jaw = None
    if use_hands:
        ed = np.load(args.expressive)
        valid = ed["valid"]
        if len(valid) != F:
            # 프레임 수 불일치 (예: 비디오 리더 오차): F에 맞춰 잘라내거나 패딩한다
            n = min(len(valid), F)
            print(f"[pt_to_anim] WARNING expressive frames={len(valid)} vs body={F}; aligning to {n}")
            def fit(a):
                out = np.zeros((F,) + a.shape[1:], a.dtype)
                out[:n] = a[:n]
                return out
            lhand_r, rhand_r, jaw_r, expr_r = map(fit, (ed["left_hand_pose"], ed["right_hand_pose"],
                                                        ed["jaw_pose"], ed["expression"]))
            valid_f = np.zeros(F, bool); valid_f[:n] = valid[:n]; valid = valid_f
        else:
            lhand_r, rhand_r, jaw_r, expr_r = (ed["left_hand_pose"], ed["right_hand_pose"],
                                               ed["jaw_pose"], ed["expression"])
        lhand_r = hold_last_valid(lhand_r, valid)
        rhand_r = hold_last_valid(rhand_r, valid)
        jaw_r = hold_last_valid(jaw_r, valid)
        expr10 = hold_last_valid(expr_r, valid)
        print(f"[pt_to_anim] graft: hands/jaw from SMPLest-X "
              f"({int(valid.sum())}/{F} detected, rest held)")
        lhand, rhand, jaw = map(lambda a: torch.from_numpy(a).float(), (lhand_r, rhand_r, jaw_r))

    joint_names = FULL_JOINT_NAMES if use_hands else BODY_JOINT_NAMES
    NUM_J = len(joint_names)

    # flat_hand_mean 선택은 손 포즈가 예측된 방식과 일치해야 한다:
    # GVHMR 바디는 편 손을 가정한다. SMPLest-X는 비-flat 손 평균 기준으로
    # 손 포즈를 예측한다. flat_hand_mean=True로 모델을 생성하고,
    # SMPLest-X 포즈에 손 평균을 더해 flat-hand 공간으로 변환한다.
    model = smplx.create(model_path=args.model_root, model_type="smplx", gender="neutral",
                         num_betas=10, use_pca=False, flat_hand_mean=True, batch_size=1)
    parents = model.parents[:NUM_J].clone().numpy()
    with torch.no_grad():
        rest_out = model(betas=mean_betas, body_pose=torch.zeros(1, 63),
                         global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                         left_hand_pose=torch.zeros(1, 45), right_hand_pose=torch.zeros(1, 45),
                         jaw_pose=torch.zeros(1, 3), leye_pose=torch.zeros(1, 3),
                         reye_pose=torch.zeros(1, 3), expression=torch.zeros(1, 10))
    rest_J = rest_out.joints[0, :NUM_J].clone()

    if use_hands:
        # 참고: flat_hand_mean=True일 때 모델의 left/right_hand_mean 버퍼는
        # 전부 제로이다 — non-flat 인스턴스에서 진짜 MANO 평균을 가져와야 하며,
        # 그렇지 않으면 변환이 no-op가 되어 손이 너무 편평하게 나온다.
        mean_src = smplx.create(model_path=args.model_root, model_type="smplx",
                                gender="neutral", num_betas=10, use_pca=False,
                                flat_hand_mean=False, batch_size=1)
        lhand = lhand + mean_src.left_hand_mean.view(1, 45)   # non-flat -> flat 공간 변환
        rhand = rhand + mean_src.right_hand_mean.view(1, 45)
        print(f"[pt_to_anim] hand-mean conversion: |Lmean|={mean_src.left_hand_mean.abs().sum():.2f} rad total")
        del mean_src
        zeros3 = torch.zeros(F, 1, 3)
        aa = torch.cat([global_orient[:, None, :], body_pose.view(F, 21, 3),
                        jaw.view(F, 1, 3), zeros3, zeros3,          # jaw, leye, reye
                        lhand.view(F, 15, 3), rhand.view(F, 15, 3)], dim=1)  # [F,55,3]
    else:
        aa = torch.cat([global_orient[:, None, :], body_pose.view(F, 21, 3)], dim=1)  # [F,22,3]
    rotmats = axis_angle_to_matrix(aa)

    # 폴백 despike는 바디 슬라이스에만 적용한다 (손 스파이크는 스무딩으로 처리한다)
    if opt.fallback_pose:
        body_slice, transl, n_bad = despike(rotmats[:, :NUM_BODY].clone(), transl)
        rotmats[:, :NUM_BODY] = body_slice
        if n_bad:
            print(f"[pt_to_anim] fallback: interpolated {n_bad} implausible frame(s)")

    if opt.upper_body_only:
        for j in LEG_JOINTS:
            rotmats[:, j] = torch.eye(3)
        print("[pt_to_anim] upper_body_only: legs frozen")

    fps_eff = opt.fps
    rotmats = smooth_rotations(rotmats, fps_eff, opt.motion_smoothing)
    transl = smooth_time(transl, fps_eff, opt.motion_smoothing)
    if opt.motion_smoothing > 1e-3:
        print(f"[pt_to_anim] smoothing: one-euro min_cutoff="
              f"{smoothing_min_cutoff(opt.motion_smoothing):.2f} Hz, "
              f"beta={ONE_EURO_BETA} (all {NUM_J} joints, zero-phase)")

    g_rot, g_pos = forward_kinematics(rest_J, parents, rotmats, transl)

    # ---- FK 결과를 스무딩된 파라미터 기준으로 SMPL-X와 교차검증 ----
    aa_s = matrix_to_axis_angle(rotmats)
    go_s = aa_s[:, 0]
    bp_s = aa_s[:, 1:NUM_BODY].reshape(F, 63)
    if use_hands:
        jaw_s = aa_s[:, 22].reshape(F, 3)
        lh_s = aa_s[:, 25:40].reshape(F, 45)
        rh_s = aa_s[:, 40:55].reshape(F, 45)
    else:
        jaw_s = torch.zeros(F, 3)
        lh_s = torch.zeros(F, 45)
        rh_s = torch.zeros(F, 45)

    max_err = 0.0
    for s in range(0, F, 64):
        e = min(s + 64, F)
        n = e - s
        with torch.no_grad():
            out = model(betas=mean_betas.expand(n, -1), body_pose=bp_s[s:e],
                        global_orient=go_s[s:e], transl=transl[s:e],
                        jaw_pose=jaw_s[s:e], left_hand_pose=lh_s[s:e],
                        right_hand_pose=rh_s[s:e],
                        leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                        expression=torch.zeros(n, 10))
        max_err = max(max_err, (out.joints[:, :NUM_J] - g_pos[s:e]).abs().max().item())
    print(f"[pt_to_anim] FK vs SMPL-X max abs error ({NUM_J} joints): {max_err:.3e} m")
    if max_err > 1e-3:
        raise SystemExit(f"[pt_to_anim] FK mismatch too large ({max_err}); aborting.")

    if opt.foot_locking:
        corr, ground = foot_lock(g_pos, opt)
        transl = transl + corr
        g_pos = g_pos + corr[:, None, :]
        print(f"[pt_to_anim] foot_locking: max root correction {corr.norm(dim=1).max():.3f} m")

    if opt.physics_filter:
        feet_y = g_pos[:, FOOT_JOINTS, 1]
        ground = torch.quantile(feet_y.reshape(-1), 0.05)
        penetration = (ground - feet_y.min(dim=1).values).clamp(min=0)
        if penetration.max() > 1e-4:
            g_pos[:, :, 1] += penetration[:, None]
            transl[:, 1] += penetration
            print(f"[pt_to_anim] physics: lifted {int((penetration>1e-4).sum())} penetrating frame(s)")

    # ---- 전방 정렬: 프레임 0이 정방향 +Z를 향하도록 전체 시퀀스를 yaw 회전 ----
    # (프레임 0 골반을 기준점으로 하여 캐릭터 위치는 유지하고 방향만 변경한다)
    if opt.align_forward:
        fwd = g_rot[0, 0] @ torch.tensor([0.0, 0.0, 1.0])   # 골반 전방 벡터, 프레임 0
        yaw = float(torch.atan2(fwd[0], fwd[2]))            # +Z로부터의 각도
        c, s = np.cos(-yaw), np.sin(-yaw)
        Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        pivot = g_pos[0, 0].clone(); pivot[1] = 0.0
        g_rot = Ry @ g_rot
        g_pos = (g_pos - pivot) @ Ry.T + pivot
        transl = (transl - pivot) @ Ry.T + pivot
        print(f"[pt_to_anim] align_forward: corrected heading by {np.degrees(yaw):.1f} deg")

    # ---- 선택적 보정 프레임 (리타게팅 보조): 프레임 0 = T 또는 A 포즈 ----
    if opt.start_pose in ("T", "A"):
        cal = torch.eye(3).repeat(NUM_J, 1, 1)[None]        # 제로 포즈 = SMPL T-포즈
        if opt.start_pose == "A":
            # 양쪽 팔을 어깨에서 ~40도 내린다 (표준 A-포즈)
            L_SH, R_SH = 16, 17
            cal[0, L_SH] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, -0.7]))
            cal[0, R_SH] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 0.7]))
        cal_rot, cal_pos = forward_kinematics(rest_J, parents, cal, transl[:1])
        g_rot = torch.cat([cal_rot, g_rot], dim=0)
        g_pos = torch.cat([cal_pos, g_pos], dim=0)
        print(f"[pt_to_anim] start_pose: prepended 1 calibration frame ({opt.start_pose}-pose)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        joint_names=np.array(joint_names),
        parents=parents.astype(np.int64),
        rest_joints=rest_J.numpy().astype(np.float32),
        global_rot=g_rot.numpy().astype(np.float32),
        global_pos=g_pos.numpy().astype(np.float32),
        fps=np.float32(opt.fps * opt.source_speed),
        betas=mean_betas.numpy().astype(np.float32),
        zup=bool(opt.zup),
        export_mesh=bool(opt.export_mesh),
    )
    print(f"[pt_to_anim] saved -> {out_path}  ({NUM_J} joints, validated, err={max_err:.2e})")

    # 메쉬 에셋: 스킨드 프리뷰용으로 SMPL-X 55 LBS 관절을 리그의 관절에 축소한다
    if opt.export_mesh:
        full_parents = model.parents.numpy()

        def rig_ancestor(j):
            while j >= NUM_J:
                j = int(full_parents[j])
            return j

        lbs = model.lbs_weights.detach().numpy()
        V = lbs.shape[0]
        w = np.zeros((V, NUM_J), dtype=np.float32)
        for j in range(lbs.shape[1]):
            w[:, rig_ancestor(j)] += lbs[:, j]
        rest_verts = rest_out.vertices[0].numpy().astype(np.float32)
        faces = model.faces.astype(np.int64)
        mesh_path = out_path.with_name("mesh.npz")
        np.savez(mesh_path, rest_verts=rest_verts, faces=faces, weights=w)
        print(f"[pt_to_anim] saved mesh assets -> {mesh_path}  (V={V}, joints={NUM_J})")


if __name__ == "__main__":
    main()
