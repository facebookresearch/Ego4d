import sys

def get_list_of_clips(split_file):
    clip_list = []
    with open(split_file, 'r')  as f:
        for line in f:
            try:
                clip = line.strip().split(',', 1)[0]
                clip_list.append(clip)
            except:
                print("Input format must be a CSV")
                exit(1)

    return clip_list
