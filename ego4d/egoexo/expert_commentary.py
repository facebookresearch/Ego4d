from typing import List


def get_paths_for_commentary_time(comm: dict, t_sec: float) -> List[dict]:
    t = t_sec * 1000  # convert to ms

    paths = []
    comm_start_t = comm["start_global_time"]
    for event in comm["events"]:
        assert event["type"] == "path"
        event_t_rel = event["global_time"] - comm_start_t
        for path in event["paths"]:
            if event["action"] == "clear" and event_t_rel < t:
                # NOTE: this could be implemented more efficiently
                paths = []  # clear out the paths

            path_t_rel = path["to"]["t"] - comm["start_global_time"]
            if path_t_rel > t:
                break
            else:
                paths.append(
                    {
                        "from": {
                            "x": path["from"]["x"],
                            "y": path["from"]["y"],
                            "t": (path["from"]["t"] - event["global_time"]) / 1000.0,
                        },
                        "to": {
                            "x": path["to"]["x"],
                            "y": path["to"]["y"],
                            "t": (path["to"]["t"] - event["global_time"]) / 1000.0,
                        },
                    }
                )
    return paths
