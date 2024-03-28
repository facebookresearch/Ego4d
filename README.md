> [!IMPORTANT]
> **EGO-EXO4D DATASET ANNOUNCEMENT:** *Ego-Exo4D* **V2** is now *available to
> the public*. V2 contains **1286.30 video hours** (**221.26 ego-hours**) across 5035
> takes with **more annotations** Please refer to the
> [changelog](https://docs.ego-exo4d-data.org/changelog) for details on what
> has changed.
>
> **EGO4D UPDATE:** *V2.1* has been released due to the addition of the
> [Goal-Step](https://openreview.net/pdf?id=3BxYAaovKr) annotations and
> accompanying "grouped videos". Please refer to the
> [documentation](https://ego4d-data.org/docs/updates/#ego4d-goal-step--grouped-videos) for more information.

# Ego4D & Ego-Exo4D

**Ego-Exo4D** is a large-scale multi-modal multi-view video dataset (including 3D) and benchmark challenge. The dataset consists of time-synchronized videos of participants recorded with at least one first-person (egocentric Aria glasses) and third-person (exocentric GoPro cameras) perspective cameras. 
- Please refer to the [website](https://ego-exo4d-data.org/),
  [documentation](https://docs.ego-exo4d-data.org/),
  [paper](https://arxiv.org/abs/2311.18259), [blog
  post](https://ai.meta.com/blog/ego-exo4d-video-learning-perception/) and
  [video introduction](https://www.youtube.com/watch?v=GdooXEBAnI8).

**Ego4D** is the world's largest egocentric (first person) video ML dataset and benchmark suite, including over 3700 hours of annotated first-person video data. 
- Please refer to the [website](https://ego4d-data.org/),
  [documentation](https://ego4d-data.org/docs/) or
  [paper](https://arxiv.org/abs/2110.07058) for more information.

## Getting Started
- To **access** the data, please refer to the Documentation:
    - For Ego-Exo4D: refer to the [Getting Started](https://docs.ego-exo4d-data.org/getting-started/) page.
    - For Ego4D: refer to the [Start Here](https://ego4d-data.org/docs/start-here/) page.
- To **download** the data, refer to:
    - For Ego-Exo4D: [Ego-Exo4D's Downloader CLI README](ego4d/egoexo/download/README.md)
    - For Ego4D: [Ego4D's CLI README](ego4d/cli/README.md)
- **Explore** Ego4D or Ego-Exo4D here (you'll need a license): [Ego4D Visualizer](https://visualize.ego4d-data.org/)
- Read the [Summary](#summary) below for details about this repository.

## Summary

The Ego4d repository (`ego4d` python module) includes: 
- [Ego-Exo4D Downloader CLI](ego4d/egoexo/download/README.md) for the Ego-Ego4D dataset (available as the command `egoexo`)
- [Ego4D Downloader CLI](ego4d/cli/README.md) for the Ego4D dataset (available as the command `ego4d`)
- A simple API abstracting common video reading libraries
([TorchAudio](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/research/readers.py#L69),
[PyAV](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/research/readers.py#L136)),
- An API for [feature
extraction](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/README.md#as-an-api), which includes [lightweight wrappers for common models](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/features/models), such as: [Omnivore](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/models/omnivore.py) and [SlowFast](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/features/models/slowfast.py)
- Notebooks (for [Ego4D](https://github.com/facebookresearch/Ego4d/tree/main/notebooks) and [Ego-Exo4D]()) serving as examples/tutorials to analyze & use the dataset
    - Colab notebooks for Ego4D serve as additional examples for the benchmarks (VQ, NLQ and STA) can be found on: https://ego4d-data.org/docs/challenge/
- Research code to train models on the dataset, e.g. [clep](https://github.com/facebookresearch/Ego4d/tree/main/ego4d/research/clep)
    - **NOTE:** baseline code for Ego-Exo4D is coming soon!
    - **NOTE:** baseline code for the Ego4D benchmarks exists on separate GitHub repositories, see the [EGO4D organization](https://github.com/EGO4D/) and [docs](https://ego4d-data.org/docs/benchmarks/overview/)


Please see [**Structure of the Repository**](#structure-of-the-repository) below for details.

## Setup

Please follow the below instructions to setup the downloader CLI for Ego4d and
to install the `ego4d` python module. 

### Option 1: From the PyPi package


>[!TIP]
>Please ensure you have a conda or pyenv environment created & activated. If you're unsure
>on how to do so, you can follow [Option 2: Step 1](step-1-create-or-use-an-environment).

```
pip install ego4d --upgrade
```

**NOTE:** Please ensure you are on at least Python 3.10

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
