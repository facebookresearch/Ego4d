source ~/anaconda3/etc/profile.d/conda.sh

## cd to root of the repository
cd ../..

conda_env=${1:-human_pose}

##-----------------------------------------------
conda create -n $conda_env python=3.9 -y ## pycolmap not supported for 3.10
conda activate $conda_env

##-----------------------------------------------
## install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y ## pytorch 2.0

##---------install dependencies---------------------------------
cd external

git clone git@github.com:rawalkhirodkar/mmlab.git ## host of all forks

cd mmlab/mmcv
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e . -v
cd ../..

cd mmlab/mmpose
pip install -r requirements.txt
pip install -v -e .
pip install flask
pip install timm==0.4.9
cd ../..

cd mmlab/mmdetection
pip install -v -e .
cd ../..

cd ..

##-----------------------------------------------
## install other dependencies
pip install yacs
pip install Rtree
pip install pyntcloud pyvista
pip install python-fcl
pip install hydra-core --upgrade
pip install av iopath
pip install pycolmap
pip install projectaria_tools

##-----------------------------------------------
## install ego4d locally
cd ../../..
pip install -e .
cd ego4d/internal/human_pose

##-----------------------------------------------
echo "Done Installing"
