# COLMAP

Run `preprocess.py`

## `preprocess.py` documentation:

Please note, as hydra is used, you should use absolute paths.

This script will generate a bash script which will execute COLMAP on the frames
produced by the `preprocess.py` script. It will additionally download and cache
data (you can force a download, see below).

Please refer to the section below for more granular docs. Please place your
YAML configuration files in `ego4d/internal/colmap/configs`

### Examples

```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example
```

specific frames for each file:
```
python3 ego4d/internal/colmap/preprocess.py --config-name frame_example
```

force download:
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example force_download=True
```

colmap bin
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example colmap_bin="path_to_colmap"
```

sync exo views (requires syncing to be complete)
```
python3 ego4d/internal/colmap/preprocess.py --config-name cmu_example sync_exo_views=True
```

### Parameter Documentation
input:
- data parameters:
    - uni_name
    - take_id
    - data_dir = "./data/"
    - colmap_dir = "./colmap"
- reconstruction params:
    - name (str, optional)
        - if no name provided, one will be generated for you based off the
          reconstruction parameters (see code below)
    - use_gpu (bool)
    - sync_exo_views (bool)
    - include_aria (bool)
    - rot_mode (int):
        - 0 => rotate none
        - 1 => rotate aria (if available)
        - 2 => rotate exo and mobile to aria
- frame selection parameters
    - frame_rate (float) (default = 1/2)
        frame rate to sample mobile, aria and exo/gopro videos from
    - exo_from_frame (int) (default = null)
        start frame for exo video
    - exo_to_frame (int) (default = null)
        end frame for exo video
    - exo_fps (float) (default = null)
        the fps for the exo video, used for sampling frame. Not used if
        supplying frames

        if null this is obtained from pyav
    - aria_fps (float) (default = 30)
        the fps for the aria video, used for sampling frame. Not used if
        supplying frames
    - mobile_fps (float) (default = null)
        the fps for the mobile (or walkthrough) video, used for sampling frame.
        Not used if supplying frames

        if null this is obtained from pyav
    - exo1_frames (list of int) (default = null)
        frames to use for the first exo video
        if non-null, `exo_to_frame` / `exo_from_frame` must be null and
        the exo2, exo3, exo4_frames must be non null
    - exo2_frames (list of int) (default = null)
        same as exo1_frames but for 2nd exo video
    - exo3_frames (list of int) (default = null)
        same as exo1_frames but for 3rd exo video
    - exo4_frames  (list of int) (default = null)
        same as exo1_frames but for 4th exo video
    - mobile_frames (list of int) (default = null)
        frames to use for the mobile video. If null, all frames to make
        `frame_rate` will be used (skip every N frames according to `mobile_fps`)
    - aria_frames (list of int) (default = null)
        frames to use for the aria video. If null, all frames to make
        `frame_rate` will be used (skip every M frames according to `aria_fps`)
output:
- in `data_dir`:
    <uni_name>_<take_id>:
        aria01 (vrs file)
        cam01 (mp4 file)
        cam02 (mp4 file)
        cam03 (mp4 file)
        cam04 (mp4 file)
        mobile (mp4 file)
        timesync.csv
        metadata.json
- in `<data_dir>/colmap/<uni_name>_<take_id>/<name>`
    - config.json
    - frames/
        contains frames
    - run_colmap.sh
        this will default to use COLMAP_BIN but can be overridden to a
        custom COLMAP_BIN

Once you run ./run_colmap.sh
