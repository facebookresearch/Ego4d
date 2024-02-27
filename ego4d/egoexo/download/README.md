# EgoExo Dowloader

>
> [!IMPORTANT]
> Please ensure you have installed the Ego4D python package. Follow the
> [instructions at the root
> README](https://github.com/facebookresearch/Ego4d/tree/main?tab=readme-ov-file#setup)
> for details.

## Pre-Read

Please see the [documentation](https://docs.ego-exo4d-data.org/download/) for
an overview of how to download the data and how/why it is partitioned.

## Usage

To use the downloader, please run `egoexo`. Typing

```bash
egoexo --help
```

Will show you a summary of the available options. You will need to supply an
output directory for where you wish to download the data to. You can do so by
supplying an argument to `-o`, for example:

>[!WARNING] 
>**If confirmed:** this will attempt to **download 14TiB** which is the
>*recommended set*. Please see the section on **[Filtering](#filtering) to
>reduce the download** size to obtain only what you care to download.

```bash
egoexo -o <out-dir>
```

By default, this will download the recommended set of data. This is equivalent
to providing `--parts metadata annotations takes captures take_trajectory`. This is quite large
(~14TiB), and as such the rest of this document will describe how to filter down
this set or include parts that are not in the "recommended" set.

### Basic Examples

To only download annotations:
```bash
egoexo -o <out-dir> --parts annotations
```
Feel free to include any other part you wish, you can include multiple, e.g.
```bash
egoexo -o <out-dir> --parts annotations metadata
```

Use --help for more information
```bash
egoexo --help
```

### Filtering

The following flags are used for filtering:

- `--benchmarks <b1> [b2] ...`: only include data from a specific benchmark. If a provided dataset `--part` includes data that is not relevant to a benchmark (i.e. general data):  it will be downloaded.
- `--splits <split1> [split2] ...`: only include data from the train, val (validation) or test set. If a provided dataset `--part` includes data that is not relevant to a split (i.e. general data): it will be downloaded.
- `--views <view1> [view2] ...`: include data only from the provided views. If data
  is general and not specific to any view: it will be downloaded (similar to `--splits`, `--benchmarks`). Provided arguments must be one or more of `ego` or `exo`.
- `-u <u1> [u2] ...`, `--universities`: filter data that comes from specific universities
- `--uids <uid1> [uid2] ...`: filter by a specific `take_uid` or `capture_uid`

### Advanced options

- `-y`, `--yes`: don't prompt a yes/no confirmation to download
- `-d`, `--delete`: delete any auxiliary files from your file system that are not included
  in this download
- `--num_workers <int>`: supply the number of workers (threads) to perform the
  download. Default is 15.
- `--release <release_id>`: download a specific version of the dataset
- `--force`: force a download of all the files. Please see "A Note On Dataset
  Updates" below for more information.

### A Note On Dataset Updates

When an update occurs: if existing files you have downloaded update then the
downloader tool will heuristically check if the file has changed by checking
if there is a delta in file size. This is not as robust as a checksum, thus you
may supply `--force` to force a download of all files.
