import random
from collections import Counter
import pandas as pd
import spacy
from tqdm import tqdm
import os

# -----------------------------
# CONFIG
# -----------------------------
random.seed(42)
nlp = spacy.load("de_core_news_sm")

# Path to your input file (adjust as needed)
INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "news_data_text_10000.jsonl")   # replace with your full dataset path
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "news-data-train-v4.jsonl")

# Target distribution (fractions of total sentences)
TARGET_DISTRIBUTION = {
    "none": 0.15,
    "homophone_swap": 0.08,
    "article_swap": 0.08,
    "article_removal": 0.08,
    "preposition_swap": 0.08,
    "noun_decapitalization": 0.08,
    "umlaut_removal": 0.08,
    "punctuation_error": 0.08,
    "adjective_ending_error": 0.08,
    "verb_conjugation_error": 0.08,
    "position_error": 0.08,
}

# Maximum number of error attempts per sentence
MAX_ERRORS_PER_SENTENCE = 3

# -----------------------------
# Utilities & vocab
# -----------------------------
articles = {"der", "die", "das", "dem", "den", "des", "ein", "eine", "einen", "einem"}
prepositions = {"an", "auf", "in", "bei", "mit", "zu", "nach", "von", "über"}
homophones = {"dass": "das", "das": "dass", "seit": "seid", "seid": "seit", "wieder": "wider", "wider": "wieder"}
umlaut_map = str.maketrans({'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss'})
ADJ_ENDINGS = ["", "e", "en", "er", "es", "em"]

# -----------------------------
# Error application functions
# Each returns (new_text, applied_bool)
# -----------------------------

def apply_homophone_swap_text(sentence):
    toks = sentence.split()
    applied = False
    out = []
    for tok in toks:
        low = tok.lower()
        if (not applied) and (low in homophones):
            swap = homophones[low]
            swap = swap.capitalize() if tok[0].isupper() else swap
            out.append(swap)
            applied = True
        else:
            out.append(tok)
    return " ".join(out), applied

def apply_article_swap_text(sentence):
    toks = sentence.split()
    applied = False
    out = []
    for tok in toks:
        if (not applied) and (tok.lower() in articles):
            new_article = random.choice(list(articles))
            new_article = new_article.capitalize() if tok[0].isupper() else new_article
            out.append(new_article)
            applied = True
        else:
            out.append(tok)
    return " ".join(out), applied

def apply_article_removal_text(sentence):
    toks = sentence.split()
    applied = False
    out = []
    for tok in toks:
        if (not applied) and (tok.lower() in articles):
            applied = True
            continue
        out.append(tok)
    return " ".join(out), applied

def apply_preposition_swap_text(sentence):
    toks = sentence.split()
    applied = False
    out = []
    for tok in toks:
        if (not applied) and (tok.lower() in prepositions):
            new_p = random.choice(list(prepositions))
            new_p = new_p.capitalize() if tok[0].isupper() else new_p
            out.append(new_p)
            applied = True
        else:
            out.append(tok)
    return " ".join(out).replace(" .", ".").replace(" ,", ","), applied

def apply_noun_decapitalization_text(sentence):
    toks = sentence.split()
    for i, tok in enumerate(toks):
        # skip first word if it's start of sentence but still allow (learners can decapitalize)
        if tok[0].isupper() and len(tok) > 1 and tok.isalpha():
            toks[i] = tok.lower()
            return " ".join(toks), True
    return sentence, False

def apply_umlaut_removal_text(sentence):
    # replace first token that has any umlaut or ß
    toks = sentence.split()
    for i, tok in enumerate(toks):
        if any(c in "äöüß" for c in tok):
            toks[i] = tok.translate(umlaut_map)
            return " ".join(toks), True
    return sentence, False

def apply_punctuation_error_text(sentence):
    if "," in sentence:
        return sentence.replace(",", ""), True
    # also occasionally remove periods if present (but be conservative)
    if "." in sentence and random.random() < 0.05:
        return sentence.replace(".", ""), True
    return sentence, False

def apply_adjective_ending_error_text(sentence):
    doc = nlp(sentence)
    applied = False
    out = []
    for token in doc:
        txt = token.text
        if (not applied) and token.pos_ == "ADJ":
            base = token.text.rstrip("eernsm")
            txt = base + random.choice(ADJ_ENDINGS)
            applied = True
        out.append(txt + token.whitespace_)
    return "".join(out), applied

def apply_conjugation_error_text(sentence):
    doc = nlp(sentence)
    out = [t.text + t.whitespace_ for t in doc]
    applied = False
    for i, token in enumerate(doc):
        if applied:
            break
        if token.pos_ in ("VERB", "AUX") and token.morph.get("VerbForm") == ["Fin"]:
            lemma = token.lemma_
            # replace with infinitive sometimes
            if random.random() < 0.4:
                out[i] = lemma + token.whitespace_
            else:
                wrong = random.choice(["e", "st", "t", "en"])
                out[i] = lemma + wrong + token.whitespace_
            applied = True
    return "".join(out), applied

def apply_position_error_text(sentence):
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    indices = [i for i, t in enumerate(doc) if not t.is_punct]
    if len(indices) < 2:
        return sentence, False
    i1, i2 = random.sample(indices, 2)
    tokens[i1], tokens[i2] = tokens[i2], tokens[i1]
    return " ".join(tokens).replace(" .", ".").replace(" ,", ","), True

# Map names to functions
ERROR_FUNCS = {
    "homophone_swap": apply_homophone_swap_text,
    "article_swap": apply_article_swap_text,
    "article_removal": apply_article_removal_text,
    "preposition_swap": apply_preposition_swap_text,
    "noun_decapitalization": apply_noun_decapitalization_text,
    "umlaut_removal": apply_umlaut_removal_text,
    "punctuation_error": apply_punctuation_error_text,
    "adjective_ending_error": apply_adjective_ending_error_text,
    "verb_conjugation_error": apply_conjugation_error_text,
    "position_error": apply_position_error_text,
}

# -----------------------------
# Feasibility checks
# -----------------------------
def feasible_errors_for_sentence(sentence):
    """Return a set/list of error type keys that can be applied to this sentence (quick checks)."""
    doc = nlp(sentence)
    toks = [t.text for t in doc]
    lowers = [t.lower() for t in toks]
    feasible = set()

    # homophone: only if any token is in homophones
    if any(l in homophones for l in lowers):
        feasible.add("homophone_swap")
    # articles
    if any(l in articles for l in lowers):
        feasible.add("article_swap")
        feasible.add("article_removal")
    # prepositions
    if any(l in prepositions for l in lowers):
        feasible.add("preposition_swap")
    # noun decap
    if any(t.text[0].isupper() and len(t.text) > 1 and t.text.isalpha() for t in doc):
        feasible.add("noun_decapitalization")
    # umlaut removal
    if any(any(c in "äöüß" for c in t.text) for t in doc):
        feasible.add("umlaut_removal")
    # punctuation error
    if "," in sentence or "." in sentence:
        feasible.add("punctuation_error")
    # adjective error
    if any(t.pos_ == "ADJ" for t in doc):
        feasible.add("adjective_ending_error")
    # conjugation error
    if any((t.pos_ in ("VERB", "AUX") and t.morph.get("VerbForm") == ["Fin"]) for t in doc):
        feasible.add("verb_conjugation_error")
    # position error (need >=2 non-punct tokens)
    non_punct = [t for t in doc if not t.is_punct]
    if len(non_punct) >= 2:
        feasible.add("position_error")

    # allow none
    if random.random() < 0.4:
        feasible.add("none")
    return feasible

# -----------------------------
# Dynamic balanced sampler
# -----------------------------
def build_targets(total_sentences, target_dist):
    targets = {}
    for k, v in target_dist.items():
        targets[k] = int(round(total_sentences * v))
    # adjust rounding so sum equals total_sentences (put residual in 'none')
    sum_targets = sum(targets.values())
    diff = total_sentences - sum_targets
    targets["none"] += diff
    return targets

def choose_errors_for_sentence(feasible_set, current_counts, targets):
    """
    Choose up to MAX_ERRORS_PER_SENTENCE error types for a sentence.
    Uses weighted bias toward underrepresented feasible errors.
    Returns list of chosen error keys (or ['none']).
    """
    # If only 'none' feasible, return it
    feasible = [f for f in feasible_set if f in TARGET_DISTRIBUTION]
    if not feasible:
        print("Warning: no feasible errors found, returning 'none'")
        return ["none"]

    # Compute "need" = target - current (positive means underrepresented)
    need = {}
    for f in feasible:
        need_val = targets[f] - current_counts.get(f, 0)
        # small floor to avoid zero; if already above target, we set small positive to keep sample diversity
        need[f] = max(0.0, need_val)

    # If all needs are zero or negative, allow some slack by converting to small positive weights
    if all(v <= 0 for v in need.values()):
        weights = {f: 1.0 for f in feasible}
    else:
        # bias weights by need, but add epsilon to ones with zero
        weights = {f: (need[f] + 1e-3) for f in feasible}

    # Decide how many errors to attempt (0 means 'none')
    # Slight bias: prefer 1 error, sometimes 2, rarely 3
    n = random.choices([0,1,2,3], weights=[0.15,0.55,0.25,0.05], k=1)[0]
    if n == 0:
        return ["none"]

    # We will sample without replacement among feasible types, weighted by weights.
    chosen = []
    avail = list(feasible)
    avail_weights = [weights[a] for a in avail]

    for _ in range(min(n, len(avail))):
        # normalize weights
        total_w = sum(avail_weights)
        if total_w <= 0:
            break
        pick = random.choices(avail, weights=avail_weights, k=1)[0]
        idx = avail.index(pick)
        chosen.append(pick)
        # remove chosen
        del avail[idx]
        del avail_weights[idx]

    if not chosen:
        print("Warning: no errors chosen, returning 'none'")
        return ["none"]
    return chosen

# -----------------------------
# Pipeline: main processing loop
# -----------------------------
def generate_corrupted_dataset(clean_sentences, targets):
    current = Counter()
    results = []
    total = len(clean_sentences)

    for sent in tqdm(clean_sentences, desc="Generating"):
        feasible = feasible_errors_for_sentence(sent)

        # pick candidate errors biased by need
        chosen = choose_errors_for_sentence(feasible, current, targets)

        if chosen == ["none"]:
            current["none"] += 1
            results.append((sent, sent))
            continue

        corrupted = sent
        applied_any = False
        applied_errors_for_this_sent = set()

        # Try to apply each chosen error; if it fails, try fallback feasible types
        for err in chosen:
            if err == "none":
                continue
            func = ERROR_FUNCS[err]
            new_text, applied = func(corrupted)
            if applied:
                corrupted = new_text
                applied_any = True
                applied_errors_for_this_sent.add(err)
                current[err] += 1

        # If none of the chosen errors applied, attempt fallback: try feasible errors in order of need
        if not applied_any:
            feasible_list = list(feasible - {"none"})
            # sort by need descending
            feasible_list.sort(key=lambda x: targets[x] - current.get(x,0), reverse=True)
            for fallback in feasible_list:
                if fallback in applied_errors_for_this_sent:
                    continue
                new_text, applied = ERROR_FUNCS[fallback](corrupted)
                if applied:
                    corrupted = new_text
                    applied_any = True
                    current[fallback] += 1
                    applied_errors_for_this_sent.add(fallback)
                    break

        if not applied_any:
            # truly no error could be applied
            current["none"] += 1
            results.append((sent, sent))
        else:
            # ensure we still count 'none' if everything cancelled out
            results.append((sent, corrupted))

    # final sanity: sum counts should equal total
    # assert sum(current.values()) == total, f"count mismatch {sum(current.values())} vs {total}"
    return results, current

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # load data
    data = pd.read_json(INPUT_PATH, lines=True)
    clean_sentences = data["text"].tolist()#[:1000]
    total = len(clean_sentences)
    print(f"Total sentences: {total}")

    # build integer targets
    targets = build_targets(total, TARGET_DISTRIBUTION)
    print("Targets (counts):")
    for k in sorted(targets.keys()):
        print(f"  {k}: {targets[k]}")

    # generate
    corrupted_pairs, counts = generate_corrupted_dataset(clean_sentences, targets)

    # write output as jsonl
    out_df = pd.DataFrame([{"de_correct": a, "de_corrupted": b} for a,b in corrupted_pairs])
    out_df.to_json(OUTPUT_PATH, lines=True, orient="records", force_ascii=False)
    print(f"Wrote {len(out_df)} records to {OUTPUT_PATH}\n")

    # print final counts vs targets
    print("Final counts vs targets:")
    for k in sorted(targets.keys()):
        actual = counts.get(k, 0)
        print(f"  {k:22s}  actual={actual:7d}   target={targets[k]:7d}   diff={actual - targets[k]:+6d}")

    # quick percentage table
    print("\nPercentages:")
    for k in sorted(targets.keys()):
        pct = counts.get(k,0) / total * 100
        print(f"  {k:22s}  {pct:5.2f}%")
