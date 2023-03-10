# COLMAP

This code is a pipeline which you can use to run COLMAP on an EgoExo take. There
is an optional but suggested requirement for running this code: QR timesync has
been run for the take. If it has not been run, you must input where the aria
walkthrough occurs in the take.

You do not require to have the take local on your machine. We can either stream
from S3 or the script will download all appropriate data to your machine (it
downloads this once). Streaming from S3 should be fast enough for the needs of
this script. The exception to this is the VRS file must be downloaded, as we
do not interact with VRS files remotely (as of yet).

We have written an associated notebook in `notebooks/COLMAP.ipynb` which will
allow you to proceed with the following steps:
1. Generate appropriate inputs to feed into COLMAP based on a supplied
   configuration
    - This does not require your videos to be downloaded locally (streaming via
      S3 can occur)
    - This step will generate a set of frames
2. Validate that the above configuration has generated appropriate frames
    - What to validate will
3. Upload your configuration to S3
4. Let Meta run COLMAP on your inputs
    - Please ensure you you ***do not*** modify the input files **unless** the
      script generates. This is such that we have reproducable
5. Optionally, you may run COLMAP and upload the files as appropriate. Again: ***do not modify generated inputs***, we want to enforce reproducibility
      - If you find that COLMAP does not generate a model appropriately without
        parameter tweaking, please communicate this and optionally add the
        option to the script generation code. COLMAP can be configured
        completely from the command line.
6. Finally validate your COLMAP outputs
    - Feel free to validate this yourself. The COLMAP script will run COLMAP's
      validation step, which will output a text file like so:
        ```
        Cameras: 6
        Images: 278
        Registered images: 278
        Points: 45824
        Observations: 750872
        Mean track length: 16.385999
        Mean observations per image: 2700.978417
        Mean reprojection error: 0.744607px
        ```
    - Please ensure the reprojection error is not large, we have 6 cameras and
      the registered images is approximately the same as the number of images.
7. Align with Aria's SLAM trajectory paths
    - TODO: integrate Rawal's pipeline
    - You can refer to Rawal's pipeline: https://github.com/rawalkhirodkar/ego4dv2/

## Setup and Installation

Please setup a conda environment with python 3.9 (this is due to pycolmap, see:
https://github.com/colmap/pycolmap/issues/111) and install appropriate packages
with the `ego4d/internal/colmap/requirements.txt` file:

```
conda create -n colmap python=3.9
conda activate colmap
pip install -r ego4d/internal/colmap/requirements.txt
```

### Setting up Aria Data Tool's: VRS

Please install vrs via following the instructions [here](https://github.com/facebookresearch/vrs#instructions-macos-and-ubuntu-and-container).

Once you have set this up you can set your vrs_bin in your configuration to use this.

### (Optional) Setting up COLMAP
Please see my
[notes](https://gist.github.com/miguelmartin75/8bf23bba1f8eaf29a6e8a98e293501a4) for compiling.

## Running on the CLI (DEPRECATED)

Parameter documentation is defined in juypter, inlined with code.

### `preprocess.py` documentation

Please the notebook in `notebooks/COLMAP.ipynb` instead of using this file
directly unless you have a specific reason to (e.g. for scripting purposes). The
notebook will output a YAML file you can feed directly into this script.

notes:
- as hydra is used, you should use absolute paths.
- this script will generate a bash script which will execute COLMAP on the frames
produced by the `preprocess.py` script. It will additionally download and cache
data (you can force a download, see below).
- please refer to the section below for more granular docs. Please place your
YAML configuration files in `ego4d/internal/colmap/configs`

### Examples

```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example
```

specific frames for each file:
```
python3 ego4d/internal/colmap/preprocess.py --config-name frame_example
```

force download:
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example force_download=True
```

colmap bin
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example colmap_bin="path_to_colmap"
```

sync exo views (requires syncing to be complete)
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example sync_exo_views=True
```
