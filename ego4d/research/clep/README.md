# CLEP: Contrastive Language Egocentric-video Pre-training

NOTE: this is very experimental code. You may have to read and modify the code
in order to run it on your machine. Dependent on interest, this may change.

This repository contains the code for contrastive learning with Ego4D's
narrations. This serves as an example of using the features for
training/prototyping ideas. It includes:

- A dataset for supervised-learning with the features. Refer to `LabelledFeatureDataset` on the [research/README.md](../README.md)
- Pre-processing scripts to convert the features to HDF5, which is a more efficient format for training.
- Zero-shot on Kinetics, Ego-Charades

## CVPR Presentation

The code in this repository was presented at CVPR 2022 in New Orleans. You can
find the presentation code in: `notebooks/` under the root directory of this
repository of this repository.

## Preprocessing Data

To pre-process the data please use the script under
`ego4d/research/clep/run_preprocess.py`. Provide `pre_config.mode` to change what
is being pre-processed.


```
python3 ego4d/research/clep/run_preprocess.py --config-name omnivore_features pre_config.mode="k400"
python3 ego4d/research/clep/run_preprocess.py --config-name omnivore_features pre_config.mode="ego_charade"
python3 ego4d/research/clep/run_preprocess.py --config-name omnivore_features pre_config.mode="ego4d_narr"
python3 ego4d/research/clep/run_preprocess.py --config-name omnivore_features pre_config.mode="ego_features"
python3 ego4d/research/clep/run_preprocess.py --config-name omnivore_features pre_config.mode="cc"
```

Also add `pre_config.slurm_config.run_locally=1` to run the preprocessing locally. 

Valid preprocessing options are:
- `ego4d_narr`
    - Extracts narration embeddings and saves via `torch.save`
- `ego4d_features`
    - Converts the features to HDF5
- `k400`
    - Extracts features from Kinetics and 
- `ego_charade`
    - Extracts features from Ego-Charades
- `cc`
    - Extracts features from Conceptual Captions
    - Requires you to download with `open_clip` prior

You will likely have to modify the paths in `configs/omnivore_features`

## Datasets

Please refer to the code `ego4d/research/clep/dataset.py`.

- `create_ego_charades_dset`: creates the Ego-Charades dataset (previously
  pre-processed)
- `create_kinetics`: creates the K400 dataset (previously pre-processed)
- `Ego4DCLEP`: The dataset for narrations/video Ego4D data 
- `CCDset`: A Conceptual Captions dataset for Ego4D

## Training

Run training with `ego4d/research/clep/train.py`.

Pass `run_locally=1` if you want to run the training process not on the cluster.
