// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import config from "./config";

function getHostname() {
  return config.port
    ? `${window.location.protocol}//${window.location.hostname}:${config.port}`
    : window.location.origin;
}

export { getHostname };
