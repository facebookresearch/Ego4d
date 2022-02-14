// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

const config = {
  /*
    The port that useMephistoReview() in the browser will connect to the
    server on.

    Tip: This is useful when developing on the review interface locally so UI changes
    don't require you to kill and relaunch the server each time.
    You can launch `$ mephisto review <some_placeholder_dir> --port 9000` once to launch
    the review server fed in with the appropriate data source, and then instead of using
    the interface at <some_placeholder_dir>, you can run `npm start` to get a
    live-reloading developer build server.
    */
  // port: 3030,
};

export default config;
