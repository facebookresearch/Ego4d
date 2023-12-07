import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import spacy

from ego4d.egoexo.expert_commentary.data import (
    load_uniq_commentaries,
    RAW_EXTRACTED_COMM_ROOT,
)
from tqdm.auto import tqdm

comms = load_uniq_commentaries(raw_extracted_dir=RAW_EXTRACTED_COMM_ROOT)
takes = json.load(
    open("/large_experiments/egoexo/dataset/takes_dropped.json")
) + json.load(open("/large_experiments/egoexo/dataset/takes.json"))
takes_by_name = {t["root_dir"]: t for t in takes}

skipped = 0
all_transc = []
all_data = []
for comm_dir in tqdm(comms):
    transc_path = os.path.join(comm_dir, "transcriptions.json")
    if not os.path.exists(transc_path):
        skipped += 1
        continue
    transc = json.load(open(transc_path))
    data_path = os.path.join(comm_dir, "data.json")
    data = json.load(open(data_path))
    all_data.append(data)
    all_transc.extend(
        [
            {
                "take": data["video_name"],
                "commentary": os.path.basename(comm_dir),
                "recording": k,
                "text": v.get("text"),
                "error": v["error"],
                "error_desc": v["error_desc"],
            }
            for k, v in transc.items()
        ]
    )

errors = [x for x in all_transc if x["error"]]
all_transc_succ = [x for x in all_transc if not x["error"]]
len(all_data), len(all_transc), len(all_transc_succ), len(errors), skipped

nlp = spacy.load("en_core_web_md")
stats = {
    "num_nouns": [],
    "num_verbs": [],
    "num_sents": [],
    "num_words": [],
    "words_per_sentence": [],
}
noun_counts = defaultdict(int)
verb_counts = defaultdict(int)
for x in tqdm(all_transc_succ):
    doc = nlp(x["text"])
    num_sents = len(list(doc.sents))
    num_words = len(doc)
    words_per_sentence = num_words / num_sents if num_sents > 0 else None
    toks_by_class = defaultdict(list)
    for tok in doc:
        toks_by_class[tok.pos_].append(tok)
    num_nouns = len(toks_by_class["NOUN"]) + len(toks_by_class["PROPN"])
    num_verbs = len(toks_by_class["VERBS"])
    for tok in toks_by_class["NOUN"]:
        noun_counts[tok.text] += 1
    for tok in toks_by_class["PROPN"]:
        noun_counts[tok.text] += 1
    for tok in toks_by_class["VERB"]:
        if tok.text == "'s":
            continue
        verb_counts[tok.text] += 1

    stats["num_nouns"].append(num_nouns)
    stats["num_verbs"].append(num_verbs)
    stats["num_sents"].append(num_sents)
    stats["num_words"].append(num_words)
    stats["words_per_sentence"].append(words_per_sentence)

noun_counts_sorted = sorted(noun_counts.items(), key=lambda x: -x[1])
verb_counts_sorted = sorted(verb_counts.items(), key=lambda x: -x[1])

stats_df = pd.DataFrame(stats)

num_anns = len({x["commentary"] for x in all_transc_succ})
num_takes = len({x["take"] for x in all_transc_succ})

comms_per_ann = defaultdict(list)
for x in all_transc_succ:
    comms_per_ann[x["commentary"]].append(x)

comms_per_ann_arr = np.array([len(xs) for xs in comms_per_ann.values()])

comm_per_min = []
for comm, xs in comms_per_ann.items():
    tn = xs[0]["take"]
    take = takes_by_name[tn]
    take_min = take["duration_sec"] / 60
    num_comms = len(xs)
    comm_per_min.append(num_comms / take_min)

comms_per_min = np.array(comm_per_min)
# comm_durs = np.array(comm_durs)

print(
    f"""
# Annotations = {num_anns}
# Takes Annotated = {num_takes}
# Commentaries = {len(all_transc_succ)}
Avg Commentaries per Annotation = {comms_per_ann_arr.mean():.3f} (std dev = {comms_per_ann_arr.std():.3f})
# Sentences = {stats_df.num_sents.sum()}
Avg Sentences per Commentary = {stats_df.num_sents.mean():.3f} (std dev = {stats_df.num_sents.std():.3f})
# Words = {stats_df.num_words.sum()}
Avg Words per Sentence = {stats_df.words_per_sentence.mean():.3f} (std dev = {stats_df.words_per_sentence.std():.3f})
# Unique Nouns = {len(noun_counts_sorted)}
# Unique Verbs = {len(verb_counts_sorted)}
Average Commentaries per Minute = {comms_per_min.mean():.3f}
"""
)

for x, _ in noun_counts_sorted[0:150]:
    print(x)
for _, count in noun_counts_sorted[0:150]:
    print(count)

for x, _ in verb_counts_sorted[0:150]:
    print(x)
for _, count in verb_counts_sorted[0:150]:
    print(count)

prof_counts = defaultdict(int)
# prof_sents = []
for x in all_data:
    rating, text = x["proficiency"]["rating"], x["proficiency"]["why"]
    prof_counts[rating] += 1
    # prof_sents.append(text)
