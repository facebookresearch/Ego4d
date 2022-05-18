**Please note VQ test annotations (for the challenge) were recently released. If needed, please download the annotations dataset again, e.g. `python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets annotations`**

# Ego4D Dataset Download CLI

This tool provides a command line interface to the publicly available Ego4D datasets.

## Prerequisites
The datasets are hosted on Amazon S3 and require credentials to access. This tool uses
the credentials stored in the home directory file: `~/.aws/credentials`. If you already
have credentials configured then you can skip this step. If not, then:

1. Install the AWS CLI from: https://aws.amazon.com/cli/
1. Open a command line and type `aws configure` (Or `aws configure --profile ego4d` if you'd prefer to use a profile and not the default credentials, in which case you'll need to specify `--aws_profile_name` below.)
1. Leave the default region blank, and enter your AWS access id and secret key when 
   prompted.

The CLI requires python >= 3.8.  Please install the prerequisites via `python setup.py install` (easyinstall) at the repo root, or via `pip install -r requirements.txt`.  (Note that the notebooks and viz engine have separate requirements - please install them by following the relevant README.)
   
## Getting Started

### Basic Usage
```
python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets full_scale annotations --metadata
```

This will download all the full scale Ego4D v1 video files and annotations to a directory on
your local computer at `~/ego4d_data/v1/full_scale` and `~/ego4d_data/v1/annotations`, as well the master metadata file at `~/ego4d_data/v1/ego4d.json`. 

Note that if you want to use the AWS credentials stored in a different [named profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html) than "ego4d", or the system default (default), you can change the `aws_profile_name` flag to the name of the profile that you want to use.

### Detailed Flags

| Flag Name   | Description |
| ---------------- | ----------- |
| `--dataset` |  [Required] A list of identifiers to download: [annotations, full_scale, clips]  Each dataset will be stored in folders in the output directory with the name of the dataset (e.g. output_dir/v1/full_scale/) and manifest. |
| `--output_directory`  | [Required]A local path where the downloaded files and metadata will be stored |
| `--metadata`  |  [Optional] Download the primary `ego4d.json` metadata at the top level (Default: True) |
| `--benchmarks`  |  [Optional] A list of benchmarks to filter dataset downloads by - e.g. Narrations/EM/FHO/AV |
| `-y` `--yes` | [Optional] If this flag is set, then the CLI will not show a prompt asking the user to confirm the download. This is so that the tool can be used as part of shell scripts. |
| `--aws_profile_name` | [Optional] Defaults to “default”. Specifies the AWS profile name from ~/.aws/credentials to use for the download |
| `--video_uids` | [Optional] List of video or clip UIDs to be downloaded. If not specified, all relevant UIDs will be downloaded. |
| `--video_uid_file` | [Optional] Path to a whitespace delimited file that contains a list of UIDs. Mutually exclusive with the `video_uids` flag. |
| `--universities` | [Optional] List of university IDs. If specified, only UIDs from the S3 buckets belonging to the listed universities will be downloaded. |
| `--version`  |  [Optional] A version identifier - e.g. “v1” |
| `--no-metadata`  |  [Optional] Bypass the `ego4d.json` metadata download |
| `--config` | [Optional] Local path to a config JSON file. If specified, the flags will be read from this file instead of the command line |

### Datasets

The following datasets are available (not exhaustive):

| Dataset | Description |
| --- | --- |
| annotations | The full set of annotations for the majority of benchmarks. | 
| full_scale | The full scale version of all videos.  (Provide `benchmarks` or `video_uids` filters to reduce the 5TB download size.) |
| clips | Clips available for benchmark training tasks.  (Provide `benchmarks` or `video_uids` filters to reduce the download size.) |
| viz | The data and thumbnails required to use the visualization package.  | 
| 3d | Annotations for the 3D VQ benchmark. |
| 3d_scans | 3D location scans for the 3D VQ benchmark. |
| 3d_scan_keypoints | 3D location scan keypoints for the 3D VQ benchmark. |
| imu | IMU data for the subset of videos available |
| slowfast8x8_r101_k400 | Precomputed [action features](https://ego4d-data.org/docs/data/features/) for Slowfast 8x8 (R101) model |
| fut_loc | Images and annotations for the future locomotion benchmark. |
| av_models | Model checkpoints for the AV/Social benchmark. |
| lta_models | Model checkpoints for the Long Term Anticipation benchmark. |
| moments_models | Model checkpoints for the Moments benchmark. |
| nlq_models | Model checkpoints for the NLQ benchmark. |
| sta_models | Model checkpoints for the Short Term Anticipation benchmark. |
| vq2d_models | Model checkpoints for the 2D VQ benchmark. |


### Manifests

Each dataset contains a manifest.csv file that lists it's contents as well as additional metadata that's available.  In particular, for `full_scale` there is metadata for each video available.  While the top level metadata `ego4d.json` is generally easier to consume and contains more information, you can consume most simple metadata from the manifest itself for each dataset.

### Universities
The following university IDs can be specified:

| University |
| --- |
| bristol |
| cmu |
| cmu_africa |
| frl_track_1_public |
| georgiatech |
| iiith |
| indiana |
| kaust |
| minnesota |
| nus |
| uniandes |
| unict |
| utokyo |



