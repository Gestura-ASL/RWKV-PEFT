#!/usr/bin/env python3
"""
Convert ParaNMT-50M into RWKV keyword→sentence dataset.

Each row in the ParaNMT file:
  sentence1<TAB>sentence2<TAB>score
Produces two JSONL entries:
  {"text": "User: <keywords>\n\nAssistant: <original sentence>"}
"""

import json
import re
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

DATA_FILE = "/home/karthikssalian/work/RWKV-PEFT/revision/para-nmt-50m.txt"
TARGET_FILE = "/home/karthikssalian/work/RWKV-PEFT/revision/converted.jsonl"


def simplify_to_keywords(text):
    """
    Convert sentence into a keyword list with inline modifiers.
    - Removes stopwords
    - Keeps content words
    - Inserts 'question' or 'exclamation' where '?' or '!' appear inline
    """
    text = text.strip()
    tokens = re.findall(r"[A-Za-z']+|[!?]", text)
    keywords = []

    for token in tokens:
        t = token.lower()
        if t in STOP:
            continue
        if t == "?":
            keywords.append("question")
        elif t == "!":
            keywords.append("exclamation")
        else:
            keywords.append(t)

    if not keywords:
        # fallback if all words removed
        keywords = [t.lower() for t in re.findall(r"[A-Za-z']+", text.lower())[:3]]

    return " ".join(keywords)


# clear the output file first
with open(TARGET_FILE, "w", encoding="utf-8"):
    pass

total = 0
with open(TARGET_FILE, "a", encoding="utf-8") as write_file:
    with open(DATA_FILE, encoding="utf-8") as read_file:
        for line in read_file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            s1, s2 = parts[0], parts[1]

            for sent in (s1, s2):
                kw = simplify_to_keywords(sent)
                text = f"User: {kw}\n\nAssistant: {sent}"
                write_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

                total += 1
                if total % 5000 == 0:
                    print(f"Processed {total:,} datapoints", end="\r")

print(f"\n✅ Created {total:,} datapoints → {TARGET_FILE}")
