This folder consists of:

1. a review interface in the `review/` directory
2. various recipes in the `recipes/` directory

## Start Script

The easiest way to run the interface is to use the run_viz script.
Once you have Mephisto installed ([`pip install mephisto`](https://github.com/facebookresearch/mephisto/blob/main/docs/quickstart.md)) and the Ego4D CLI installed, you can run:

```
./run_viz.sh
```

from this repo's root directory.

This will install the ego4d `viz` dataset and launch the interface. You can use `-h` or examine the script to modify configurations options.

In particular, `VID_ROOT` will default to `~\ego4d_data` and should be updated if you've already used the CLI to download the videos (and/or the `viz` dataset, which is used here).

## Review Interface

The `review/` folder was created via a create-react-app template.

```bash
$ npx create-react-app review --template mephisto-review
```

Most of the custom code added to the template will be found in either the `review/src/custom/`folder or `review/src/index.js`.

Therefore, if you'd like to update the version of the template to integrate upstream changes, you can easily do so by invoke `create-react-app` again as shown above and copying over the file and folder mentioned above.

## Recipes

See the [README](recipes/README.md) in the directory for more information of installation and prerequisites.
