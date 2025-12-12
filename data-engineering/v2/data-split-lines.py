import pandas as pd
# import spacy
from tqdm import tqdm
import re
import nltk
import os

# nltk.download('punkt_tab')

# nlp = spacy.load("de_core_news_sm")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../v1/data', 'news_data.csv'))
df_de = df["text"].dropna().tolist()

def clean(text: str) -> str:
    text = re.sub(r'[„“"""\n]', '', text)
    text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
    text = ' '.join(text.split())
    return text


def chunk_sentences(sentences, chunk_size=3):
    for i in range(0, len(sentences), chunk_size):
        yield ' '.join(sentences[i:i + chunk_size])


df_de_chunked = []
for text in tqdm(df_de[:30000], desc="Processing German texts"):
    sentences = [clean(sent.strip()) for sent in nltk.tokenize.sent_tokenize(text, language='german')]
    # chunks = list(chunk_sentences(sentences, chunk_size=3))
    df_de_chunked.extend(sentences)

# sentences = [clean(sent.text.strip()) for sent in nlp('Die Ukraine ist als Corona-Risikogebiet eingestuft. „Wir halten uns an die Auflagen und überwachen das strengsten. Die Spieler tragen Masken im Hotel, viele Sitzungen machen wir in kleinen Gruppen“, so Löw. „Wir tun unser Möglichstes und wollen das Spiel dort auch durchführen. Es ist wichtig, die nächsten Spiele siegreich zu gestalten.“",Julien Wolff,1602109285,"Fußball, Unentschieden gegen die Türkei",welt').sents]
# chunks = list(chunk_sentences(sentences, chunk_size=3))
# df_de_chunked.extend(chunks)

print("Final length:", len(df_de_chunked))
# print(df_de_chunked)

df_output = pd.DataFrame({ "id": range(len(df_de_chunked)), "text": df_de_chunked})
df_output.to_json(
    os.path.join(os.path.dirname(__file__), 'data', 'news_data_text_30000.jsonl'), orient='records', lines=True, force_ascii=False
)

print("Saved to news_data_text_30000.jsonl")