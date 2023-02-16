**Please note VQ test annotations (for the challenge) were recently released. If needed, please download the annotations dataset again, e.g. `ego4d --output_directory="~/ego4d_data" --datasets annotations`**

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

### Installation
Install via pip (conda support coming):
```
pip install ego4d
```

### Basic Usage
In your python environment, use the `ego4d` command line directly:
```
ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations --metadata
```

(Alternatively, use traditional python module syntax: `python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets full_scale annotations --metadata --version v2`)

This will download all the full scale Ego4D v2 video files and annotations to a directory on
your local computer at `~/ego4d_data/v2/full_scale` and `~/ego4d_data/v2/annotations`, as well the master metadata file at `~/ego4d_data/v2/ego4d.json`. 

Note that if you want to use the AWS credentials stored in a different [named profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html) than "ego4d", or the system default (default), you can change the `aws_profile_name` flag to the name of the profile that you want to use.

### Detailed Flags

| Flag Name   | Description |
| ---------------- | ----------- |
| `--dataset` |  [Required] A list of identifiers to download: [annotations, full_scale, clips]  Each dataset will be stored in folders in the output directory with the name of the dataset (e.g. output_dir/v2/full_scale/) and manifest. |
| `--output_directory`  | [Required]A local path where the downloaded files and metadata will be stored |
| `--metadata`  |  [Optional] Download the primary `ego4d.json` metadata at the top level (Default: True) |
| `--benchmarks`  |  [Optional] A list of benchmarks to filter dataset downloads by - e.g. Narrations/EM/FHO/AV |
| `-y` `--yes` | [Optional] If this flag is set, then the CLI will not show a prompt asking the user to confirm the download. This is so that the tool can be used as part of shell scripts. |
| `--aws_profile_name` | [Optional] Defaults to “default”. Specifies the AWS profile name from ~/.aws/credentials to use for the download |
| `--video_uids` | [Optional] List of video or clip UIDs to be downloaded. If not specified, all relevant UIDs will be downloaded. |
| `--video_uid_file` | [Optional] Path to a whitespace delimited file that contains a list of UIDs. Mutually exclusive with the `video_uids` flag. |
| `--universities` | [Optional] List of university IDs. If specified, only UIDs from the S3 buckets belonging to the listed universities will be downloaded. |
| `--version`  |  [Optional] A version identifier - e.g. “v1” or "v2" (default) |
| `--no-metadata`  |  [Optional] Bypass the `ego4d.json` metadata download |
| `--config` | [Optional] Local path to a config JSON file. If specified, the flags will be read from this file instead of the command line |

### Datasets

The following datasets are available (not exhaustive):

| Dataset | Description |
| --- | --- |
| annotations | The full set of annotations for the majority of benchmarks. | 
| full_scale | The full scale version of all videos.  (Provide `benchmarks` or `video_uids` filters to reduce the 5TB download size.) |
| clips | Clips available for benchmark training tasks.  (Provide `benchmarks` or `video_uids` filters to reduce the download size.) |
| video_540ss | The downscaled version of all videos - rescaled to 540px on the short side.  (Provide `benchmarks` or `video_uids` filters to reduce the 5TB download size.) |
| annotations_540ss | The annotations corresponding to the downscaled `video_540ss` videos - primarily differing only in spatial annotations (e.g. bounding boxes). |
| 3d | Annotations for the 3D VQ benchmark. |
| 3d_scans | 3D location scans for the 3D VQ benchmark. |
| 3d_scan_keypoints | 3D location scan keypoints for the 3D VQ benchmark. |
| imu | IMU data for the subset of videos available |
| slowfast8x8_r101_k400 | Precomputed [action features](https://ego4d-data.org/docs/data/features/) for the Slowfast 8x8 (R101) model |
| omnivore_video_swinl | Precomputed [action features](https://ego4d-data.org/docs/data/features/) for the Omnivore Video model |
| omnivore_image_swinl | Precomputed [action features](https://ego4d-data.org/docs/data/features/) for the Omnivore Image model |
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


## Frequently Asked Questions (and their answers)
1. I get `ClientError: An error occurred (403) when calling the HeadObject operation: Forbidden`
   1. Are your datasets spelled correctly? If not, you'll see warnings at the top of your cli output.
   2. Has it been more than 14 days since you got your license? License keys expire after 14 days, please re-sign the license with the same email at ego4ddataset.com to reactivate it.
