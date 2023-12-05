# EgoExo Dowloader

## Setup

Be sure you are in the root directory of the repository. Setup a conda environment:
```
conda create -n ego4d python=3.11 # create or use an existing env
conda use ego4d
pip install .
```

## Usage

```
egoexo -o <out-dir>
```
By default, without specifying --parts you will be downloading the following parts: metadata captures takes trajectory annotations

To only download annotations:
```
egoexo -o <out-dir> --parts annotations
```
Feel free to include any other part you wish, you can include multiple, e.g.
```
egoexo -o <out-dir> --parts annotations metadata
```

Use --help for more information
```
egoexo --help
```
