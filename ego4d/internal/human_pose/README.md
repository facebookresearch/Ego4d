# Human Pose Estimation Pipeline

Steps:
1. [x] Preprocess egoexo data
    - [x] download files
    - [x] frame extraction (TODO make optional):
        - [x] aria frame extraction
        - [x] exo camera frame extraction
    - [x] paths + camera calibrations to JSON
2. [ ] Human pose detection
    - [ ] (low-pri) read streaming frame data
    - [x] read preprocessed frame data
    - [ ] obtain bounding boxes 
        - [ ] FasterRCNN
        - [ ] Aria trajectory
            - [x] world space -> exo image space
            - [x] world space -> ego (aria) image space
    - [ ] run HRNet
3. [ ] Triangulation

## Setup

NOTE: python3.9 needed due to pycolmap, see: https://github.com/colmap/pycolmap#getting-started

```bash
cd ego4d/internal/human_pose/
conda create -n human_pose39 python=3.9 -y
conda activate human_pose39
pip install -r requirements.txt
```

On the FAIR cluster you will have to load CUDA and CUDNN before installing, via:

```bash
module load cuda/11.2 cudnn/v8.1.1.33-cuda.11.0
```

### mmpose, mmdetection

For the configurations, you need to clone the following repos somewhere on your filesystem, e.g.

```bash
mkdir tp
pushd tp
git clone https://github.com/open-mmlab/mmpose/
git clone https://github.com/open-mmlab/mmdetection
popd
```

## Usage

Configurations are written with Hydra/OmegaConf as YAML files. New
configurations must be added to `ego4d/internal/human_pose/configs`.

### Run Pipeline

Please ensure you have run the setup step first. From the repository root:

```bash
python3 ego4d/internal/human_pose/main.py --config-name unc_T1 mode=preprocess
```

- [ ] TODO: scale horizontally / run via SLURM
