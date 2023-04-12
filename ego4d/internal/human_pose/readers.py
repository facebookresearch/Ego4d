import av


def get_video_meta(path):
    with av.open(path) as cont:
        n_frames = cont.streams[0].frames
        codec = cont.streams[0].codec.name
        tb = cont.streams[0].time_base

        all_pts = []
        for x in cont.demux(video=0):
            if x.pts is None:
                continue
            all_pts.append(x.pts)

            if len(all_pts) >= 2:
                assert all_pts[-1] > all_pts[-2]

        assert len(all_pts) == n_frames
        return {
            "all_pts": all_pts,
            "codec": codec,
            "tb": tb,
        }


def read_frame_idx_set(path, frame_indices, stream_id):
    meta = get_video_meta(path)
    with av.open(path) as cont:
        initial_pts = meta["all_pts"][frame_indices[0]]
        last_pts = meta["all_pts"][frame_indices[-1]]
        pts_to_idx = {meta["all_pts"][idx]: idx for idx in frame_indices}
        cont.seek(initial_pts, stream=cont.streams.video[stream_id], any_frame=False)
        seen = 0
        for f in cont.decode(video=stream_id):
            if f.pts > last_pts:
                break
            if f.pts not in pts_to_idx:
                # print("Skipping", f.pts)
                continue

            idx = pts_to_idx[f.pts]
            seen += 1
            yield idx, f.to_ndarray(format="rgb24")

        assert seen == len(pts_to_idx)
