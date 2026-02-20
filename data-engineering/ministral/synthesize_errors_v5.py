import random
from collections import Counter
import pandas as pd
import spacy
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
            prog='synthesize_errors_v5.py',
            description='Adds realistic grammatical errors to clean sentences, with a dynamic sampling strategy to achieve a target distribution of error types. Optimized for German and anti-laziness.',
            usage='python synthesize_errors_v5.py --input <input_path> --correct_column <column_name> --corrupt_column <column_name> --output <output_path>',
        )

parser.add_argument('--input', type=str, default="data/news_data_train.jsonl", help='Path to input JSONL file containing clean sentences.')
parser.add_argument('--correct_column', type=str, default="de_correct", help='Name of the column containing clean sentences.')
parser.add_argument('--corrupt_column', type=str, default="de_corrupt", help='Name of the column to store sentences with errors.')
parser.add_argument('--output', type=str, default="data/news_data_train-errors.jsonl", help='Path to output JSONL file for sentences with errors.')
args = parser.parse_args()

# -----------------------------
# CONFIG
# -----------------------------
random.seed(42)
nlp = spacy.load("de_core_news_sm")

# Path to your input file (adjust as needed)
INPUT_PATH = os.path.join(os.getcwd(), args.input)
OUTPUT_PATH = os.path.join(os.getcwd(), args.output)
CORRECT_COLUMN = args.correct_column
CORRUPT_COLUMN = args.corrupt_column

# OPTIMIZED FOR "ANTI-LAZINESS" (German)
TARGET_DISTRIBUTION = {
    "none": 0.05,                   # REDUCED: Don't let it relax yet.
    
    # --- THE "LAZINESS" KILLERS (60%) ---
    # These force the model to analyze every noun phrase.
    "article_swap": 0.20,           # DOUBLED: Force it to check Der/Die/Das/Dem/Den
    "article_removal": 0.15,        # BOOSTED: Force insertion of missing words
    "adjective_ending_error": 0.15, # BOOSTED: Force checking "großem" vs "großen"
    "noun_decapitalization": 0.10,  # BOOSTED: Force checking Nouns
    
    # --- THE STRUCTURAL CHECKS (25%) ---
    "verb_conjugation_error": 0.10,
    "position_error": 0.10,         # V2 rule violations are critical for German
    "preposition_swap": 0.05,
    
    # --- THE NOISE (10%) ---
    # Keep these low; they are easy to learn and don't help with deep grammar.
    "homophone_swap": 0.05,        
    "umlaut_removal": 0.03,        
    "punctuation_error": 0.02,     
}

# Maximum number of error attempts per sentence
MAX_ERRORS_PER_SENTENCE = 3

# -----------------------------
# Utilities & vocab
# -----------------------------
article_swaps = {
            "der": ["die", "das", "den", "dem"],
            "die": ["der", "das"], 
            "das": ["der", "die"],
            "den": ["dem", "der"],
            "dem": ["den", "der"]
        }
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
        low = tok.lower()
        # Check if we have a smart swap for this specific article
        if (not applied) and (low in article_swaps):
            # Pick from the specific confusion set (e.g. dem -> den)
            new_article = random.choice(article_swaps[low])
            
            # Match capitalization
            new_article = new_article.capitalize() if tok[0].isupper() else new_article
            out.append(new_article)
            applied = True
        elif (not applied) and (low in articles):
            # Fallback for articles not in your swap dict (like 'ein')
            new_article = random.choice(list(articles - {low}))
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
        low = tok.lower()
        if (not applied) and (low in prepositions):
            # Exclude the current preposition from choices
            new_p = random.choice(list(prepositions - {low}))
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
    # Remove only the first comma, not all commas
    if "," in sentence:
        return sentence.replace(",", "", 1), True
    # also occasionally remove periods if present (but be conservative)
    if "." in sentence and random.random() < 0.05:
        return sentence.replace(".", "", 1), True
    return sentence, False

def apply_adjective_ending_error_text(sentence):
    doc = nlp(sentence)
    applied = False
    out = []
    for token in doc:
        txt = token.text
        if (not applied) and token.pos_ == "ADJ" and len(token.text) > 2:
            # Try to find the base by removing known endings
            word = token.text
            base = word
            for ending in ["em", "en", "er", "es", "e"]:
                if word.endswith(ending) and len(word) > len(ending) + 1:
                    base = word[:-len(ending)]
                    break
            # Pick a different ending than the current one
            current_ending = word[len(base):] if len(base) < len(word) else ""
            available_endings = [e for e in ADJ_ENDINGS if e != current_ending]
            txt = base + random.choice(available_endings)
            applied = True
        out.append(txt + token.whitespace_)
    return "".join(out), applied

def apply_tense_safe_conjugation_error(sentence):
    doc = nlp(sentence)
    out = [t.text + t.whitespace_ for t in doc]
    applied = False
    
    # 1. Define sets of endings to swap WITHIN the same tense
    # We strip these endings and replace them with others from the same group
    present_endings = ["e", "st", "t", "en"]
    # Common weak past endings (te, test, ten, tet)
    past_endings = ["te", "test", "ten", "tet"] 
    
    for i, token in enumerate(doc):
        if applied:
            break
            
        # 2. TARGET ONLY FINITE VERBS
        if token.pos_ in ("VERB", "AUX") and token.morph.get("VerbForm") == ["Fin"]:
            
            # 3. SAFETY CHECK: THE "SIE" TRAP
            # Find the subject of this verb
            subjects = [child for child in token.children if child.dep_ == "sb"]
            if subjects:
                subj_text = subjects[0].text.lower()
                # If subject is "sie" (she/they) or "es" (it), ambiguity is high. SKIP.
                # "Wir/Ihr/Ich/Du" are safer because the pronoun defines the verb strictly.
                if subj_text in ["sie", "es"]:
                    continue

            text = token.text
            
            # STRATEGY A: WEAK VERBS (Regular) - Swap Suffixes
            # Check if it looks like a Past Tense word (ends in -te, -ten, etc.)
            if any(text.endswith(end) for end in past_endings):
                # It's likely Past Tense. 
                # Pick a WRONG ending from the PAST group.
                current_ending = next(end for end in past_endings if text.endswith(end))
                stem = text[:-len(current_ending)]
                
                wrong_ending = random.choice([e for e in past_endings if e != current_ending])
                out[i] = stem + wrong_ending + token.whitespace_
                applied = True
                
            # STRATEGY B: PRESENT / STRONG VERBS - Swap Suffixes
            elif any(text.endswith(end) for end in present_endings):
                # It's likely Present.
                current_ending = next(end for end in present_endings if text.endswith(end))
                # Be careful not to strip too much from short verbs (e.g. "tu")
                if len(text) > len(current_ending) + 1:
                    stem = text[:-len(current_ending)]
                    wrong_ending = random.choice([e for e in present_endings if e != current_ending])
                    out[i] = stem + wrong_ending + token.whitespace_
                    applied = True

            # STRATEGY C: FALLBACK (If strict logic failed, just Append)
            # Only do this if we haven't applied a change yet
            if not applied:
                 # Just adding an 'st' or 'n' often breaks grammar without changing tense
                 # e.g. "ging" -> "gingst" (Still looks past tense)
                 wrong_ending = random.choice(present_endings)
                 out[i] = text + wrong_ending + token.whitespace_
                 applied = True

    return "".join(out), applied

def apply_position_error_text(sentence):
    doc = nlp(sentence)
    # Filter for non-punctuation indices that have a non-punct neighbor
    indices = [i for i, t in enumerate(doc) if not t.is_punct]
    
    # Find valid swap pairs (both non-punct and adjacent)
    valid_pairs = []
    for i in range(len(indices) - 1):
        idx1, idx2 = indices[i], indices[i + 1]
        # Only swap if they are actually adjacent in the original doc
        if idx2 == idx1 + 1:
            valid_pairs.append((idx1, idx2))
    
    if not valid_pairs:
        return sentence, False
    
    idx1, idx2 = random.choice(valid_pairs)
    
    # Rebuild with swapped tokens, preserving whitespace
    out = []
    for i, token in enumerate(doc):
        if i == idx1:
            out.append(doc[idx2].text + token.whitespace_)
        elif i == idx2:
            out.append(doc[idx1].text + token.whitespace_)
        else:
            out.append(token.text + token.whitespace_)
    return "".join(out), True

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
    "verb_conjugation_error": apply_tense_safe_conjugation_error,
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

    # Note: "none" is handled dynamically in choose_errors_for_sentence based on need
    return feasible

# -----------------------------
# Dynamic balanced sampler
# -----------------------------
def build_targets(total_sentences, target_dist):
    targets = {}
    for k, v in target_dist.items():
        targets[k] = int(round(total_sentences * v))
    # Note: Due to multiple errors per sentence, sum of targets may exceed total_sentences.
    # This is expected behavior - targets represent error type counts, not sentence counts.
    return targets

def choose_errors_for_sentence(feasible_set, current_counts, targets):
    """
    Choose up to MAX_ERRORS_PER_SENTENCE error types for a sentence.
    Uses weighted bias toward underrepresented feasible errors.
    Returns list of chosen error keys (or ['none']).
    """
    # Get feasible error types (excluding 'none' for now)
    feasible = [f for f in feasible_set if f in TARGET_DISTRIBUTION and f != "none"]
    
    # Compute "need" for 'none' to decide if we should return no errors
    none_need = targets.get("none", 0) - current_counts.get("none", 0)
    total_remaining = sum(targets.values()) - sum(current_counts.values())
    
    # Calculate probability of choosing 'none' based on how much we still need
    if total_remaining > 0 and none_need > 0:
        none_prob = none_need / total_remaining
        if random.random() < none_prob:
            return ["none"]
    
    # If no feasible errors, return 'none'
    if not feasible:
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

    # Decide how many errors to attempt
    # Slight bias: prefer 1 error, sometimes 2, rarely 3
    n = random.choices([1,2,3], weights=[0.65,0.30,0.05], k=1)[0]

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
    print(f"Loading data from {INPUT_PATH}...")
    data = pd.read_json(INPUT_PATH, lines=True)
    clean_sentences = data[CORRECT_COLUMN].tolist() #[:10]
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
    out_df = pd.DataFrame([{CORRUPT_COLUMN: b, CORRECT_COLUMN: a} for a,b in corrupted_pairs])
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
