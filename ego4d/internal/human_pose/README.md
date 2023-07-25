# Human Pose Estimation Pipeline

Steps:
1. Preprocess egoexo data
    - download files
    - frame extraction
        - aria frame extraction
        - exo camera frame extraction
    - paths + camera calibrations to JSON
2. Obtain human bounding boxes 
    - Uses the Aria trajectory + FasterRCNN + heuristics
3. 2D Pose Detection
4. Triangulation
5. Smoothing

## Setup

### Dependencies

We depend on @rawalkhirodkar's fork of mmlab and related repositories. This
will be installed with the `requirements.txt` via pip. If you want to manually
install it, you can follow the instructions below.

```bash
mkdir tp
pushd tp
git clone git@github.com:rawalkhirodkar/mmlab.git
popd

pushd tp/mmlab/mmcv
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e . -v
popd

pushd tp/mmlab/mmpose
pip install .
popd

pushd tp/mmlab/mmdetection
pip install .
popd

pip install "torch>=2.0.0"
```
NOTE: python3.9 needed due to pycolmap, see: https://github.com/colmap/pycolmap#getting-started

### Install

```bash
cd ego4d/internal/human_pose/
conda create -n human_pose39 python=3.9 -y
conda activate human_pose39
pip install -r requirements.txt
pip install --upgrade numpy
```

Alternatively there is an install script located under
`ego4d/internal/human_pose/scripts/_install/conda.sh` maintained by
@rawalkhirodkar.

### Notes for the FAIR cluster
On the FAIR cluster you will have to load CUDA and CUDNN before installing, via:

```bash
module load cuda/11.2 cudnn/v8.1.1.33-cuda.11.0
```

## Usage

Configurations are written with Hydra/OmegaConf as YAML files. New
configurations must be added to `ego4d/internal/human_pose/configs`.

### Run Pipeline

Please ensure you have run the setup step first. From the repository root:

```bash
python3 ego4d/internal/human_pose/main.py --config-name unc_T1 mode=preprocess repo_root_dir=$PWD
python3 ego4d/internal/human_pose/main.py --config-name unc_T1 mode=bbox repo_root_dir=$PWD
python3 ego4d/internal/human_pose/main.py --config-name unc_T1 mode=pose2d repo_root_dir=$PWD
```

# TODOs
- [ ] TODO: scale horizontally / run via SLURM
