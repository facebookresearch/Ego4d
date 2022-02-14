# Ego4D Features

This sub-directory contains code to extract features from the Ego4D dataset.
The code allows you to use a set of models and can be used for audio, video or
image features.

For scheduling on a cluster with SLURM support see `slurm.py`. Scheduling
occurs via a greedy bin packing algorithm. SLURM arrays for job submission are
used.

## Requirements

Please see requirements.txt at the base repository directory.

submitit is the only requirement that is not required if you are wanting to
*not* schedule on a SLURM cluster.

### conda

```sh
conda create --name ego4d_public
conda activate ego4d_public
pip install -r requirements.txt
```

## Usage

### Profiling/Testing
Running a test extraction to ensure you have everything setup right:

```sh
python3 ego4d/features/profile.py --config-name slowfast_r101_8x8 schedule_config.run_locally=1
```

This will benchmark the code to allow you to estimate/configure the scheduling
parameters. Don't provide `schedule_config.run_locally=1` if you want to
schedule it on the cluster.

### Schedule The Extraction

```sh
python3 ego4d/features/slurm.py --config-name slowfast_r101_8x8
```

### As an API

Refer to `ego4d/features/extract_features.py` and the functions:
- `extract_features`
- `perform_feature_extraction`


## Configuring

Hydra is used for configuration. You can override configuration options through
CLI arguments or by modifying the yaml files in the directory

Pre-configured YAML files are in the subdirectory `ego4d/features/configs/`.

There exists the following model configurations:
1. SlowFast 8x8 ResNet101 pre-trained on Kinetics 400 (see [`slowfast_r101_8x8.yaml`](ego4d/features/configs/slowfast_r101_8x8.yaml))

#### How to Run with a different config (model)

Provide `--config-name <name>`

Where name is the name of the configuration file without the `.yaml` extension.

#### Run on a subset of videos

Provide `io.uid_list` in the YAML (`InputOutputConfig.uid_list`) or as a list of arguments on the CLI.

Example:

```bash
python3 ego4d/features/slurm.py --config-name slowfast_r101_8x8 io.uid_list="[000a3525-6c98-4650-aaab-be7d2c7b9402]"
```

## Adding a Model

I'd recommend just copy-pasting an existing model python file.

1. Add a new python file to `ego4d/features/models`
2. Ensure you have the following:
    - ModelConfig, which must inherit from `ego4d.features.model.base_model_config.BaseModelConfig`
        - Additional configuration for your model
    - get_transform(config: ModelConfig)
    - load_model(config: ModelConfig)
