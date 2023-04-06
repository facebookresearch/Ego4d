source ~/anaconda3/etc/profile.d/conda.sh

## cd to root of the repository
cd ../..

# # # ###-------------------------------------------
conda create -n mmd python=3.8 -y
conda activate mmd

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y ### pyotrch 1.11.0

## install mmcvfull
pip install "mmcv-full" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html    




### install mmdet
pip install -v -e .

