"""
공유 파이프라인 옵션. 웹 UI 컨트롤(M4) 및 DeepMotion 옵션과 1:1로 대응한다.
모든 단계 + 백엔드에서 사용하는 단일 진실 공급원이다.
"""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class PipelineOptions:
    # --- 입력 / 타이밍 ---
    fps: float = 30.0
    space: str = "global"          # "global" (월드) | "incam" (카메라)
    source_speed: float = 1.0      # DeepMotion "소스 비디오 속도" (리타임 계수)

    # --- 스무딩 / 안정화 (M2) ---
    motion_smoothing: float = 0.4  # 0..1 -> One-Euro min_cutoff (속도 적응형:
                                   # 지터를 스무딩하고 빠른 동작은 보존한다)
    foot_locking: bool = True      # 접지 프레임에서 발 미끄러짐 제거
    foot_lock_height: float = 0.06 # 추정 지면 위 m 단위, 접지 판정 기준
    physics_filter: bool = True    # 간단 지면 클램프 (발이 지면 아래로 내려가지 않음)

    # --- 바디 선택 ---
    upper_body_only: bool = False  # 다리를 rest로 고정 (앉은 자세 / 웹캠)

    # --- 억스프레시브 파츠 (M5, SMPLest-X) ---
    hand_tracking: bool = False    # 손가락 + 턱 (SMPLest-X 단계)
    head_rotation: bool = True     # 머리/목은 GVHMR 바디에 이미 포함됨
    # (눈 시선 제거됨: SMPLest-X가 예측하지 않음)

    # --- 폴백 ---
    fallback_pose: bool = True     # 저신뢰 프레임을 hold/보간 처리한다

    # --- 내보내기 ---
    zup: bool = True               # SMPL Y-up -> Blender Z-up 변환
    export_mesh: bool = False      # 스킨드 SMPL-X 메쉬 베이킹 (더 나은 프리뷰)
    start_pose: str = "none"       # "none" | "T" | "A" — 리타게팅 정렬을 위해
                                   # 프레임 0에 보정 자세를 삽입한다
    align_forward: bool = True     # 전체 시퀀스를 yaw 회전하여 캐릭터가
                                   # 프레임 0에서 정방향을 향하도록 한다 (GVHMR의
                                   # 월드 방향은 첫 프레임 카메라에 따라 결정되므로,
                                   # 피사체가 "뒷모습"을 보이는 경우가 있다)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PipelineOptions":
        return cls(**{k: v for k, v in json.loads(s).items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
