import config from "./config";

function getHostname() {
  return config.port
    ? `${window.location.protocol}//${window.location.hostname}:${config.port}`
    : window.location.origin;
}

export { getHostname };
