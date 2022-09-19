**Please note VQ test annotations (for the challenge) were recently released. If needed, please download the annotations dataset again, e.g. `python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets annotations`**

# Ego4D

EGO4D is the world's largest egocentric (first person) video ML dataset and benchmark suite, with 3,600 hrs (and counting) of densely narrated video and a wide range of annotations across five new benchmark tasks.  It covers hundreds of scenarios (household, outdoor, workplace, leisure, etc.) of daily life activity captured in-the-wild by 926 unique camera wearers from 74 worldwide locations and 9 different countries.  Portions of the video are accompanied by audio, 3D meshes of the environment, eye gaze, stereo, and/or synchronized videos from multiple egocentric cameras at the same event.  The approach to data collection was designed to uphold rigorous privacy and ethics standards with consenting participants and robust de-identification procedures where relevant.


## Getting Started
- To **access** the data, please refer to the Documentation's [Getting Started](https://ego4d-data.org/docs/start-here/) page.
- To **download** the data, refer to the [CLI README](ego4d/cli/README.md)
- **Explore** the dataset here (you'll need a license): [Ego4D Visualizer](https://visualize.ego4d-data.org/)

## Visualization and Demo
- For a demo notebook: [Annotation Notebook](notebooks/annotation_visualization.ipynb)
- For the visualization engine: [Viz README](viz/narrations/README.md)

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

# License

Ego4D is released under the [MIT License](LICENSE).
