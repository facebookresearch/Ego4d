import sys, os
import json


def is_overlapping(x, y):
    if y[0] < x[1] and y[1] >= x[0]:
        return True
    return False


def merge_transcription(trn_list):
    N = len(trn_list)
    if N < 1:
        return "", []
    trn_list= sorted(trn_list, key=lambda x: x[0])
    final_list = []
    final_times_list = []
    is_used = [0] * N
    for k, x in enumerate(trn_list):
        merged_list= []
        if x[3] is None :
            continue
        if is_used[k]:
            continue
        is_used[k] = True
        for j in range(k+1, N):
            if is_overlapping(x, trn_list[j]):
                merged_list.append(trn_list[j])
                is_used[j] = 1
            else:
                break
        overlap_txt = " ".join([z[3].strip() for z in merged_list if z[3] is not None])
        if overlap_txt.strip() != "" and x[3] != overlap_txt:
            if x[3].strip() != '':
                final_str = "{} {} / @ / {} {}".format("{", x[3], overlap_txt, "}")
                final_list.append(final_str)
                final_times_list.append(
                    (
                        x[0], x[1], x[2], final_str
                    )
                )
            else:
                final_list.append(overlap_txt)
                final_times_list.append((x[0], x[1], x[2], overlap_txt))
        else:
            if x[3].strip != '':
                final_list.append(x[3])
                final_times_list.append(x)

    final_str = " ".join([z.strip() for z in final_list if z is not None]).strip()
    return final_str, final_times_list


def main(args):
    clip2trn = {}

    with open(args.subset_json, 'r') as f:
        av_json = json.load(f)

        for video in av_json["videos"]:
            video_uid = video["video_uid"]
            for clip in video["clips"]:
                clip_uid = clip["clip_uid"]
                trn_list = []
                for trn in clip["transcriptions"]:
                    if trn["transcription"].strip() != "":
                        trn_list.append(
                            (
                                float(trn["start_time_sec"]),
                                float(trn["end_time_sec"]),
                                trn["person_id"],
                                trn["transcription"].strip()
                            )
                        )

                final_txt, final_trn_list = merge_transcription(trn_list)
                clip2trn[clip_uid] = final_txt

    with open(args.out_csv, 'w') as f:
        with open(args.out_trn, 'w') as trn_f:
            for clip in sorted(clip2trn):
                f.write("{},{}\n".format(clip, clip2trn[clip]))
                trn_f.write("{} ({}_1_0_30000)\n".format(clip2trn[clip], clip))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subset_json",
        type=str,
        help="Input JSON file that contains train/val/test set info"
    )
    parser.add_argument(
        "out_csv",
        type=str,
        help="Output CSV file that contains clip to transcription mapping"
    )
    parser.add_argument(
        "out_trn",
        type=str,
        help="Output TRN file that has transcriptions in SCLITE trn format"
    )

    args = parser.parse_args()
    main(args)
