"""
단계: anim.npz -> FBX (+ GLB), 선택적으로 스킨드 SMPL-X 메쉬를 포함한다.

Blender 컨테이너에서 실행한다 (bpy 모듈, 헤드리스):
  python /work/scripts/anim_to_fbx.py --anim <anim.npz> --fbx <out.fbx> --glb <out.glb>

본 구성은 조건부로 처리한다:
  - 메쉬 모드  : 본이 +Y 방향, identity rest -> Blender LBS == SMPL LBS (정확한 스키닝)
  - 스켈레톤 모드: 본이 head->tail 방향 (시각적으로 자연스러움); 애니메이션은 동일하다
양쪽 모두 pose_bone.matrix에 FK 글로벌 변환을 직접 설정하므로, 관절 위치는
소스와 정확히 일치한다. 내장된 재검증이 최대 위치 오차를 출력한다.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import bpy
from mathutils import Matrix, Vector

Y_UP_TO_Z_UP = Matrix.Rotation(np.pi / 2.0, 4, "X")


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat_from(rot3x3, pos3):
    m = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            m[r][c] = float(rot3x3[r][c])
        m[r][3] = float(pos3[r])
    return m


def build_armature(joint_names, parents, rest_joints, identity_rest):
    arm_data = bpy.data.armatures.new("SMPLX")
    arm_obj = bpy.data.objects.new("SMPLX", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    children = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    ebones = []
    for i, name in enumerate(joint_names):
        b = arm_data.edit_bones.new(name)
        head = Vector(rest_joints[i].tolist())
        if identity_rest:
            # +Y 본, identity rest 방향 -> 정확한 LBS 스키닝
            b.head = head
            b.tail = head + Vector((0, 0.1, 0))
            b.use_connect = False
        else:
            if children[i]:
                tail = Vector(rest_joints[children[i][0]].tolist())
                if (tail - head).length < 1e-4:
                    tail = head + Vector((0, 0.05, 0))
            else:
                p = parents[i]
                d = (head - Vector(rest_joints[p].tolist())) if p >= 0 else Vector((0, 0.05, 0))
                d = d.normalized() * 0.05 if d.length > 1e-4 else Vector((0, 0.05, 0))
                tail = head + d
            b.head, b.tail, b.use_connect = head, tail, False
        ebones.append(b)

    for i, p in enumerate(parents):
        if p >= 0:
            ebones[i].parent = arm_data.edit_bones[joint_names[p]]
    bpy.ops.object.mode_set(mode="OBJECT")

    if identity_rest:  # identity rest 회전을 명시적으로 강제한다
        bpy.ops.object.mode_set(mode="EDIT")
        for i, name in enumerate(joint_names):
            arm_data.edit_bones[name].matrix = Matrix.Translation(Vector(rest_joints[i].tolist()))
        bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def build_mesh(arm_obj, joint_names, mesh_npz):
    d = np.load(mesh_npz)
    verts = d["rest_verts"]           # [V,3]
    faces = d["faces"]                # [F,3]
    weights = d["weights"]            # [V,22]
    mesh = bpy.data.meshes.new("body")
    mesh.from_pydata([v.tolist() for v in verts], [], [f.tolist() for f in faces])
    mesh.update()
    obj = bpy.data.objects.new("body", mesh)
    bpy.context.collection.objects.link(obj)

    # 명시적 비금속 매터리얼 — 없으면 glTF가 기본값(metallicFactor=1.0)을
    # 사용하여 환경 맵 없는 뷰어에서 거의 검게 렌더링된다
    mat = bpy.data.materials.new("body_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.78, 0.80, 0.85, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(mat)

    vgs = [obj.vertex_groups.new(name=n) for n in joint_names]
    nz = np.nonzero(weights > 1e-4)
    for v, j in zip(*nz):
        vgs[j].add([int(v)], float(weights[v, j]), "REPLACE")

    obj.parent = arm_obj
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    return obj


def apply_animation(arm_obj, joint_names, global_rot, global_pos, fps):
    F = global_rot.shape[0]
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 0, F - 1
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    pbones = [arm_obj.pose.bones[n] for n in joint_names]
    for pb in pbones:
        pb.rotation_mode = "QUATERNION"
    order = list(range(len(joint_names)))
    for f in range(F):
        scene.frame_set(f)
        for i in order:
            pbones[i].matrix = mat_from(global_rot[f, i], global_pos[f, i])
            bpy.context.view_layer.update()
        for i in order:
            pbones[i].keyframe_insert("location", frame=f)
            pbones[i].keyframe_insert("rotation_quaternion", frame=f)
    bpy.ops.object.mode_set(mode="OBJECT")


def verify(arm_obj, joint_names, global_pos, n_check=8):
    scene = bpy.context.scene
    F = global_pos.shape[0]
    frames = np.linspace(0, F - 1, num=min(n_check, F)).astype(int)
    max_err = 0.0
    for f in frames:
        scene.frame_set(int(f))
        bpy.context.view_layer.update()
        for i, n in enumerate(joint_names):
            got = np.array(arm_obj.pose.bones[n].matrix.translation)
            max_err = max(max_err, float(np.abs(got - global_pos[f, i]).max()))
    return max_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anim", required=True)
    ap.add_argument("--fbx", default=None)
    ap.add_argument("--glb", default=None)
    args = ap.parse_args()

    d = np.load(args.anim, allow_pickle=True)
    joint_names = [str(x) for x in d["joint_names"]]
    parents = d["parents"].astype(int)
    rest_joints = d["rest_joints"].astype(np.float32)
    global_rot = d["global_rot"].astype(np.float32)
    global_pos = d["global_pos"].astype(np.float32)
    fps = float(d["fps"])
    zup = bool(d["zup"]) if "zup" in d else True
    export_mesh = bool(d["export_mesh"]) if "export_mesh" in d else False
    mesh_npz = Path(args.anim).with_name("mesh.npz")
    has_mesh = export_mesh and mesh_npz.exists()
    print(f"[anim_to_fbx] joints={len(joint_names)} frames={global_rot.shape[0]} "
          f"fps={fps} mesh={has_mesh} zup={zup}")

    clear_scene()
    arm = build_armature(joint_names, parents, rest_joints, identity_rest=has_mesh)
    if has_mesh:
        build_mesh(arm, joint_names, mesh_npz)
    apply_animation(arm, joint_names, global_rot, global_pos, fps)

    err = verify(arm, joint_names, global_pos)
    print(f"[anim_to_fbx] posed-bone vs source max abs error: {err:.6e} m")
    if err > 1e-4:
        raise SystemExit(f"[anim_to_fbx] armature mismatch too large ({err}); aborting.")

    if zup:
        arm.matrix_world = Y_UP_TO_Z_UP @ arm.matrix_world

    if args.fbx:
        bpy.ops.export_scene.fbx(filepath=args.fbx, use_selection=False,
                                 add_leaf_bones=False, bake_anim=True,
                                 bake_anim_use_all_bones=True,
                                 bake_anim_use_nla_strips=False,
                                 bake_anim_use_all_actions=False)
        print(f"[anim_to_fbx] wrote {args.fbx}")
    if args.glb:
        bpy.ops.export_scene.gltf(filepath=args.glb, export_format="GLB",
                                  export_animations=True)
        print(f"[anim_to_fbx] wrote {args.glb}")


if __name__ == "__main__":
    main()
    # bpy는 모든 작업 완료 *후* 인터프리터 종료 시 SIGSEGV가 발생할 수 있다
    # (exit 139). 이미 모두 기록되었으므로 깔끔하게 종료하고
    # 버그 있는 정리 과정을 건너뛴다.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
