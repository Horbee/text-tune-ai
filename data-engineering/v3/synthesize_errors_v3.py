import random
import pandas as pd
import os
import spacy
from tqdm import tqdm

random.seed(42)

error_counts = {
    "none": 0,
    "homophone_swap": 0,
    "article_swap": 0,
    "article_removal": 0,
    "preposition_swap": 0,
    "noun_decapitalization": 0,
    "umlaut_removal": 0,
    "punctuation_error": 0,
    "adjective_ending_error": 0,
    "verb_conjugation_error": 0,
    "position_error": 0
}

nlp = spacy.load("de_core_news_sm")
ADJ_ENDINGS = ["", "e", "en", "er", "es", "em"]

def introduce_errors(sentence, error_rate=0.3):
    """
    Corrupts a German sentence with various error types.
    error_rate: Probability that a specific token gets corrupted.
    """
    tokens = sentence.split()
    new_tokens = []
    
    # Mappings for specific errors
    articles = ["der", "die", "das", "dem", "den", "des", "ein", "eine", "einen", "einem"]
    prepositions = ["an", "auf", "in", "bei", "mit", "zu", "nach", "von", "über"]
    homophones = {
        "dass": "das", "das": "dass",
        "seit": "seid", "seid": "seit",
        "wieder": "wider", "wider": "wieder"
    }
    umlaut_map = str.maketrans({'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss'})

    for token in tokens:
        roll = random.random()
        
        # Skip corruption based on error rate
        if roll > error_rate:
            new_tokens.append(token)
            continue

        # 1. Homophone Swap (Specific words)
        if token.lower() in homophones:
            new_token = homophones[token.lower()]
            # Maintain capitalization
            if token[0].isupper(): new_token = new_token.capitalize()
            new_tokens.append(new_token)
            error_counts["homophone_swap"] += 1
        
        # 2. Article Swap (Grammar/Case error)
        elif token.lower() in articles:
            if random.random() < 0.5:
                # Article Removal
                error_counts["article_removal"] += 1
                continue
            else:
                # Article Swap
                new_article = random.choice(articles)
                if token[0].isupper(): new_article = new_article.capitalize()
                new_tokens.append(new_article)
                error_counts["article_swap"] += 1
                
        # 3. Preposition Swap
        elif token.lower() in prepositions:
            new_prep = random.choice(prepositions)
            if token[0].isupper(): new_prep = new_prep.capitalize()
            new_tokens.append(new_prep)
            error_counts["preposition_swap"] += 1

        # 4. Noun De-capitalization (Orthography)
        elif token[0].isupper() and len(token) > 1:
            new_tokens.append(token.lower())
            error_counts["noun_decapitalization"] += 1

        # 5. Umlaut Removal (Typos)
        elif any(char in "äöüß" for char in token):
            new_tokens.append(token.translate(umlaut_map))
            error_counts["umlaut_removal"] += 1

        # 6. No error applicable, keep original
        else:
            new_tokens.append(token)

    # Reconstruct sentence
    corrupted_sentence = " ".join(new_tokens)
    
    # Optional: Remove commas (Punctuation error)
    if random.random() < 0.3:
        corrupted_sentence = corrupted_sentence.replace(",", "")
        error_counts["punctuation_error"] += 1

    if corrupted_sentence == sentence:
        error_counts["none"] += 1

    return corrupted_sentence

def introduce_adjective_error(sentence):
    doc = nlp(sentence)
    tokens = []

    for token in doc:
        text = token.text

        if token.pos_ == "ADJ":
            error_counts["adjective_ending_error"] += 1
            error_type = random.choice(["wrong_ending", "strip_ending"])
            
            if error_type == "wrong_ending":
                base = token.text.rstrip("eernsm")  # rough stem
                new_ending = random.choice(ADJ_ENDINGS)
                text = base + new_ending

            elif error_type == "strip_ending":
                text = token.text.rstrip("eernsm")
                
        tokens.append(text + token.whitespace_)

    return "".join(tokens) 

def introduce_conjugation_error(sentence):
    doc = nlp(sentence)
    tokens = [t.text + t.whitespace_ for t in doc]
    # print(tokens)

    for i, token in enumerate(doc):
        # print(token.text, token.pos_, token.morph)
        # Identify finite verbs (best target for conjugation errors)
        if token.pos_ in ["VERB", "AUX"] and token.morph.get("VerbForm") == ["Fin"]:
            error_counts["verb_conjugation_error"] += 1

            original = token.text
            lemma = token.lemma_

            # Simple error strategies:
            # 1) Replace with infinitive
            if random.random() < 0.4:
                tokens[i] = lemma + token.whitespace_  # infinitive form
                continue

            # 2) Replace with incorrect suffix
            if original.endswith("e") and random.random() < 0.5:
                tokens[i] = original[:-1] + "t" + token.whitespace_
                continue

            if original.endswith("t") and random.random() < 0.5:
                tokens[i] = original[:-1] + "en" + token.whitespace_
                continue

            # 3) Add wrong person ending
            wrong_endings = ["e", "st", "t", "en"]
            wrong = random.choice(wrong_endings)
            tokens[i] = lemma + wrong + token.whitespace_

    return "".join(tokens)

def introduce_position_error(sentence):
    doc = nlp(sentence)
    tokens = [t.text + t.whitespace_ for t in doc]

    # Get indices of all tokens (excluding punctuation)
    word_indices = [i for i, t in enumerate(doc) if not t.is_punct]

    # Swap two random words if there are at least 2
    if len(word_indices) >= 2:
        i1, i2 = random.sample(word_indices, 2)
        tokens[i1], tokens[i2] = tokens[i2], tokens[i1]
        error_counts["position_error"] += 1

    return "".join(tokens)

def error_introduction_pipeline(sentence):
    if random.random() < 0.4:
        corruptions = [
            introduce_conjugation_error,
            introduce_adjective_error,
            introduce_position_error
        ]

        return random.choice(corruptions)(sentence)
    else:
        return introduce_errors(sentence)


data = pd.read_json(f'{os.path.join(os.path.dirname(__file__), "data", "news_data_text_10000.jsonl")}', lines=True)
clean_sentences = data['text'].tolist()

corrupted_data = pd.DataFrame({
    'de_correct': clean_sentences,
    'de_corrupted': [error_introduction_pipeline(s) for s in tqdm(clean_sentences)]
})


corrupted_data.to_json(f'{os.path.join(os.path.dirname(__file__), "data", "news_data_train_10000_v3.jsonl")}', lines=True, orient='records', force_ascii=False)

print("Error Counts:")
for error_type, count in error_counts.items():
    print(f"{error_type}: {count}")

#Error Counts:
# none: 66
# homophone_swap: 15
# article_swap: 9
# article_removal: 12
# preposition_swap: 26
# noun_decapitalization: 142
# umlaut_removal: 29
# punctuation_error: 57
# adjective_ending_error: 15
# verb_conjugation_error: 66
# position_error: 51