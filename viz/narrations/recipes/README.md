# Recipes for running a narrations visualization server

Get the prerequisites set up:

- [Ego4D CLI](https://github.com/facebookresearch/Ego4D/tree/master/ego4d/cli)
- [Mephisto CLI](https://github.com/facebookresearch/mephisto/blob/main/docs/quickstart.md)

_Optional:_

- [`jq`](https://stedolan.github.io/jq/download/) - We provide a precomputed input file which makes this dependency unnecessary. If you'd like to create custom input files from the `recipes/` folder, however, you'll need this utility.

Then from the `recipes/` folder, you can do:

```console
$ ./1_gather_ids.sh 5 | ./2_dl_videos.sh | ./3_prepare_input.sh | ./4_review.sh
```

This will download and run the first 5 narrations and view in the Mephisto review interface.

--

If you already have all of the files saved down somewhere, you can just do:

```shell
$ VID_ROOT=~/ego4d ./3_prepare_input.sh ALL | ./4_review.sh
```

If you also have a precomputed input file, you can use it as such:

```shell
$ VID_ROOT=~/ego4d cat precomputed.json | ./4_review.sh
```

#### Cross-script Variables (with defaults)

```bash
INPUT_FILE=./narrations_v2_7-27-21.json
VID_ROOT=~/ego4d
```

These variables can also be updated and exported into your current terminal session by running `./0_config.sh`.

- By default, `./2_dl_videos.sh` will download files to `~/ego4d` and the review server launched in `./4_review.sh` will also serve them from there. This can be configured via the `$VID_ROOT` ENV var, also exported from `./0_config.sh`
- By default, the narrations file that will be used is `./narrations_v2_7-27-21.json` (used by `./1_gather_ids.sh` and `./3_prepare_input.sh`). This can be configured via the `$INPUT_FILE` ENV var, also exported from `./0_config.sh`

# Details

```console
$ ./1_gather_ids.sh 5
000786a7-3f9d-4fe6-bfb3-045b368f7d44 000a3525-6c98-4650-aaab-be7d2c7b9402 000cd456-ff8d-499b-b0c1-4acead128a8b 001e3e4e-2743-47fc-8564-d5efd11f9e90 00277df3-9107-4592-ba85-b8d054149551

$ ./1_gather_ids.sh
000786a7-3f9d-4fe6-bfb3-045b368f7d44 000a3525-6c98-4650-aaab-be7d2c7b9402

$ ./1_gather_ids.sh ALL
000786a7-3f9d-4fe6-bfb3-045b368f7d44 ...

# Note: Naive and slow implementation at the moment.
#
# Feel free to modify with your own logic as long as the output is
# a whitespace delimited list of video_uids
#
# Right now it just grabs the first X uids from the narrations.json file (default: 2).
```

```console
$ echo 000786a7-3f9d-4fe6-bfb3-045b368f7d44 | ./2_dl_videos.sh
000786a7-3f9d-4fe6-bfb3-045b368f7d44

$ ./1_gather_ids.sh | ./2_dl_videos.sh
000786a7-3f9d-4fe6-bfb3-045b368f7d44 000a3525-6c98-4650-aaab-be7d2c7b9402

$ echo 000786a7-3f9d-4fe6-bfb3-045b368f7d44 | ./2_dl_videos.sh LOG

# Note: Downloads the list of video_uids passed in via stdin.
# Uses the ego4d cli under the hood to do so.

# Passing in LOG as an arg will output the ego4d CLI output for debug purpposes.
# NOTE: This will break piping functionality, e.g. you won't be able to do this then:
#
# echo 000786a7-3f9d-4fe6-bfb3-045b368f7d44 | ./2_dl_videos.sh | ./3_prepare_input.sh
```

```console
$ echo 000786a7-3f9d-4fe6-bfb3-045b368f7d44 ... | ./3_prepare_input.sh
{ "info": ... }
{ "info": ... }
{ "info": ... }

$ ./3_prepare_input.sh ALL
# Prints out an output file with all of the video annotations in the input narrations file
```

```console
# Uses the Mephisto Review CLI under the hood

$ ./1_gather_ids.sh 5 | ./3_prepare_input.sh | ./4_review.sh
# Will launch the review server with the first 5 files in the narrations file. Note that the videos will not load
# unless saved down before, as step 2 isn't run here.

$ ./1_gather_ids.sh 5 | ./2_dl_videos.sh | ./3_prepare_input.sh | ./4_review.sh
# Now the review server will run with the videos as well, as opposed to the variation above
```
