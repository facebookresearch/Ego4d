# Ego4D Dataset Download CLI

This tool provides a command line interface to the publicly available Ego4D datasets.

## Prerequisites
The datasets are hosted on Amazon S3 and require credentials to access. This tool uses
the credentials stored in the home directory file: `~/.aws/credentials`. If you already
have credentials configured then you can skip this step. If not, then:

1. Install the AWS CLI from: https://aws.amazon.com/cli/
1. Open a command line and type `aws configure --profile ego4d`  (Or default if not other AWS profiles.)
1. Leave the default region blank, and enter your AWS access id and secret key when 
   prompted.
   
## Getting Started

### Basic Usage
```
python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets full_scale annotations
```

This will download all the full scale Ego4D v1 video files and annotations to a directory on
your local computer at `~/ego4d_data/full_scale`. Note that if you want to use the
AWS credentials stored in a different [named profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
than "ego4d", or the system default (default), you can change the `aws_profile_name` flag to the name of the profile that you want to use.

### Detailed Flags
| Flag Name   | Description |
| ---------------- | ----------- |
| `--dataset` |  [Required] A list of identifiers to download: [annotations, full-scale]  Each dataset will be stored in folders in the output directory with the name of the dataset (e.g. output_dir/full-scale/) and manifest. |
| `--output_directory`  | [Required]A local path where the downloaded files and metadata will be stored |
| `--version`  |  [Optional] A version identifier - e.g. “v1” |
| `--benchmarks`  |  [Optional] A list of benchmarks to filter dataset downloads by - e.g. Narrations/EM/FHO/AV |
| `--aws_profile_name` | [Optional] Defaults to “default”. Specifies the AWS profile name from ~/.aws/credentials to use for the download |
| `--video_uids` | [Optional] List of video UIDs to be downloaded. If not specified, all relevant UIDs will be downloaded. |
| `--video_uid_file` | [Optional] Path to a whitespace delimited file that contains a list of UIDs. Mutually exclusive with the `video_uids` flag. |
| `--universities` | [Optional] List of university IDs. If specified, only UIDs from the S3 buckets belonging to the listed universities will be downloaded. |
| `-y` `--yes` | [Optional] If this flag is set, then the CLI will not show a prompt asking the user to confirm the download. This is so that the tool can be used as part of shell scripts. |
| `--config` | [Optional] Local path to a config JSON file. If specified, the flags will be read from this file instead of the command line |

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



