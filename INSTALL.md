# 설치 가이드 (원본 GVHMR 기준)

## 환경 설정

```bash
git clone https://github.com/zju3dv/GVHMR
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
# 다른 레포지토리에서 gvhmr을 편집 가능 모드(editable)로 설치하려면, settings.json 파일에 "python.analysis.extraPaths": ["path/to/your/package"] 구문을 추가해 본다.
```

### 선택 사항: DPVO (빠른 추론 속도를 원한다면 권장하지 않음)
```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
```

## 입력 및 출력

```bash
mkdir inputs
mkdir outputs
```

**가중치 (Weights)**

```bash
mkdir -p inputs/checkpoints

# 1. [SMPL](https://smpl.is.tue.mpg.de/) 및 [SMPLX](https://smpl-x.is.tue.mpg.de/)를 다운로드하려면 가입이 필요하다. 체크포인트는 다음 디렉터리 구조에 맞게 배치해야 한다:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (SMPLX 파라미터 예측 및 평가용)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (렌더링 및 평가용)

# 2. 구글 드라이브(Google-Drive)에서 다른 사전 훈련된 모델들을 다운로드한다 (다운로드 시 해당 라이선스에 동의하는 것으로 간주된다): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

**데이터 (Data)**

훈련 및 평가를 위해 전처리된 데이터를 제공한다.
원본 데이터셋(어노테이션, 비디오 등)을 직접 배포하지는 않으므로, 원본 웹사이트에서 직접 다운로드해야 한다.
*라이선스 제한으로 인해 원본 데이터를 직접 제공할 수 없다.*
전처리된 데이터를 다운로드함으로써 원본 데이터셋의 이용 약관에 동의하며, 연구 목적으로만 데이터를 사용해야 한다.

구글 드라이브([Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link))에서 다운로드할 수 있다. 파일을 "inputs" 폴더에 배치하고 다음 명령어들을 실행한다:

```bash
cd inputs
# 훈련 (Train)
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# 테스트 (Test)
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# 다음과 같은 폴더 구조를 갖추어야 한다:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```
