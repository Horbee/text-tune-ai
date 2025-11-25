import random
import pandas as pd
import os

error_counts = {
    "none": 0,
    "homophone_swap": 0,
    "article_swap": 0,
    "preposition_swap": 0,
    "noun_decapitalization": 0,
    "umlaut_removal": 0,
    "punctuation_error": 0
}

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

# --- Test it ---
# clean_sentences = [
#     "Ich weiß, dass das Wetter heute schön ist.",
#     "Der Mann gibt der Frau einen Apfel.",
#     "Wir freuen uns auf den Urlaub in Spanien."
# ]

data = pd.read_json(f'{os.path.join(os.path.dirname(__file__), "data", "news_data_text_5000.jsonl")}', lines=True)
clean_sentences = data['text'].tolist()

corrupted_data = pd.DataFrame({
    'de_correct': clean_sentences,
    'de_corrupted': [introduce_errors(s) for s in clean_sentences]
})

corrupted_data.to_json(f'{os.path.join(os.path.dirname(__file__), "data", "news_data_text_5000_corrupted.jsonl")}', lines=True, orient='records', force_ascii=False)

print("Error Counts:")
for error_type, count in error_counts.items():
    print(f"{error_type}: {count}")