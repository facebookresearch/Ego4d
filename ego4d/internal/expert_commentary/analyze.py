import os
import json

import spacy
import pandas as pd
from spacy.lang.en.examples import sentences 
from tqdm.auto import tqdm
from collections import defaultdict

from ego4d.internal.expert_commentary.data import (
    RAW_EXTRACTED_COMM_ROOT,
    load_raw_commentaries,
)

comms = load_raw_commentaries(RAW_EXTRACTED_COMM_ROOT)
comms[0]

all_transc = []
for comm_dir in comms:
    transc_path = os.path.join(comm_dir, "transcriptions.json")
    assert os.path.exists(transc_path)
    transc = json.load(open(transc_path))
    all_transc.extend([
        {
            "commentary": os.path.basename(comm_dir),
            "recording": k,
            "text": v.get("text"),
            "error": v["error"],
            "error_desc": v["error_desc"]
        }
        for k, v in transc.items()
    ])

errors = [x for x in all_transc if x["error"]]
len(errors)
len(all_transc)
all_transc_succ = [x for x in all_transc if not x["error"]]


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
        verb_counts[tok.text] += 1

    stats["num_nouns"].append(num_nouns)
    stats["num_verbs"].append(num_verbs)
    stats["num_sents"].append(num_sents)
    stats["num_words"].append(num_words)
    stats["words_per_sentence"].append(words_per_sentence)

noun_counts_sorted = sorted(noun_counts.items(), key=lambda x: -x[1])
verb_counts_sorted = sorted(verb_counts.items(), key=lambda x: -x[1])

print("# Nouns", len(noun_counts_sorted))
for x, count in noun_counts_sorted[0:150]:
    print(x, count)
print()
print()
print()
print("# Verbs", len(verb_counts_sorted))
for x, count in verb_counts_sorted[0:150]:
    print(x, count)

stats_df = pd.DataFrame(stats)
stats_df
