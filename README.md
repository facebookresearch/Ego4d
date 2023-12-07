> [!NOTE]
> **DATESET UPDATE:** Ego4D **V2.1** has been released due to the addition of the [Goal-Step](https://openreview.net/pdf?id=3BxYAaovKr) annotations and accompanying "grouped videos". Please refer to the [documentation](https://ego4d-data.org/docs/updates/) for more information.
>
> You can proceed to download via `--benchmark goalstep` using `--datasets full_scale annotations`, [see CLI docs](https://ego4d-data.org/docs/CLI/) or [Getting Started](https://ego4d-data.org/docs/start-here/) if you are new to the dataset. As of writing, the PyPi package is not up to date, you will have to download/clone the repository & run the python script: `python3 -m ego4d.cli.cli --datasets full_scale annotations --benchmarks goalstep -o <out-dir>`


# Ego4D

EGO4D is the world's largest egocentric (first person) video ML dataset and benchmark suite, with 3,600 hrs (and counting) of densely narrated video and a wide range of annotations across five new benchmark tasks.  It covers hundreds of scenarios (household, outdoor, workplace, leisure, etc.) of daily life activity captured in-the-wild by 926 unique camera wearers from 74 worldwide locations and 9 different countries.  Portions of the video are accompanied by audio, 3D meshes of the environment, eye gaze, stereo, and/or synchronized videos from multiple egocentric cameras at the same event.  The approach to data collection was designed to uphold rigorous privacy and ethics standards with consenting participants and robust de-identification procedures where relevant.

## Getting Started
- To **access** the data, please refer to the Documentation's [Getting Started](https://ego4d-data.org/docs/start-here/) page.
- To **download** the data, refer to the [CLI README](ego4d/cli/README.md)
- **Explore** the dataset here (you'll need a license): [Ego4D Visualizer](https://visualize.ego4d-data.org/)
- Read the [Summary](#summary) below for details about this repository.

## Summary

The Ego4d repository (`ego4d` python module) includes: 
- [Downloader CLI](ego4d/cli/README.md) for the Ego4D dataset
- A simple API abstracting common video reading libraries
([TorchAudio](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/research/readers.py#L69),
[PyAV](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/research/readers.py#L136)),
- An API for [feature
extraction](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/README.md#as-an-api), which includes [lightweight wrappers for common models](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/features/models), such as: [Omnivore](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/models/omnivore.py) and [SlowFast](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/models/slowfast.py)
- [Notebooks](https://github.com/facebookresearch/Ego4d/tree/main/notebooks) serving as examples/tutorials to analyze & use the dataset
    - Colab notebooks serving as additional examples for the benchmarks (VQ, NLQ and STA) can be found on: https://ego4d-data.org/docs/challenge/
- Research code to train models on the dataset, e.g. [clep](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/research/clep)
    - **NOTE:** baseline code for the benchmarks exists on separate GitHub repositories, see the [EGO4D organization](https://github.com/EGO4D/) and [docs](https://ego4d-data.org/docs/benchmarks/overview/)


Please see [**Structure of the Repository**](#structure-of-the-repository) below for details.

## Setup

Please follow the below instructions to setup the downloader CLI for Ego4d and
to install the `ego4d` python module. 


### Option 1: From the PyPi package

> [!WARNING]
> This is currently an out of date package, we are working on updating this.
> For now, please follow [Option 2](#option-2-clonedownload-the-code).

```
pip install ego4d
```

### Option 2: Clone/Download the Code

Ensure you have cloned or downloaded the code to your local disk. All
instructions assume you are the **root of the repository**.

#### Step 1: Create or Use an Environment

Create a conda environment to enable pip installation:
```
conda create -n ego4d python=3.11 -y
conda activate ego4d
```

If you are using an existing conda (or pyenv) environment: please ensure you
have installed *at least* Python 3.10.

#### Step 2: 

```
pip install .  # run from the root of Ego4d
```

Now you should be able to import ego4d:

```
python3 -c 'import ego4d; print(ego4d)'
```

You can check that the ego4d module links to the correct file on your file system from the output of the above command.

## Structure of the Repository
The repository contains multiple directories covering a specific theme. Each
theme contains an associated `README.md` file, please refer to them.


All python code is located in the `ego4d` and associated subdirectories. The
goal for each subdirectory is to cover one specific theme. 

- `ego4d`: the `ego4d` *python* module exists
    - [`cli`](ego4d/cli/README.md): The Ego4D CLI for downloading the dataset
    - [`features`](ego4d/features/README.md): Feature extraction across the dataset
    - [`research`](ego4d/research/README.md): Everything related to research and
      usage of the dataset (dataloaders, etc).
        - [`research/clep`](ego4d/research/clep/README.md): Contrastive Language Ego-centric video Pre-training
- [`viz`](viz/narrations/README.md): visualization engine

## Visualization and Demo
- For a demo notebook: [Annotation Notebook](notebooks/annotation_visualization.ipynb)
- For the visualization engine: [Viz README](viz/narrations/README.md)

# License

Ego4D is released under the [MIT License](LICENSE).
