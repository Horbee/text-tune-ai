import numpy as np
from tqdm import tqdm
import ollama

EMBEDDING_MODEL = "embeddinggemma" #"bge-m3" #"nomic-embed-text"


def get_embeddings(sentences: list[str], model=EMBEDDING_MODEL, show_progress=True) -> np.ndarray:
    """
    Convert a list of sentences into vector embeddings using Ollama.
    """
    embeddings = []
    for sentence in tqdm(sentences, desc="Getting embeddings", disable=not show_progress):
        response = ollama.embed(model=model, input=sentence)
        embeddings.append(response["embeddings"][0])
    return np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between corresponding rows."""
    # Normalize vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    # Compute dot product for each pair
    return np.sum(a_norm * b_norm, axis=1)


def get_similarity_score(text1: list[str], text2: list[str], model=EMBEDDING_MODEL, show_progress=False) -> list[float]:
    """
    Get similarity score between two texts using Ollama embeddings.
    """
    emb1 = get_embeddings(text1, model=model, show_progress=show_progress)
    emb2 = get_embeddings(text2, model=model, show_progress=show_progress)
    sim = cosine_similarity(emb1, emb2)
    return sim.tolist()



if __name__ == "__main__":
    print(f"Using Ollama embedding model: {EMBEDDING_MODEL}")

    texts1 = ["Das ist ein Beispieltext.", "Das ist ein rotes Auto.", "Ich liebe Programmierung.", "Das Wetter ist heute schön."]
    texts2 = ["Das ist ein Beispieltext.", "Dies ist mein rotes Auto.", "Ich hasse Programmierung.", "Das Wetter ist heute sehr schlecht."]

    scores = get_similarity_score(texts1, texts2, model=EMBEDDING_MODEL)  
    print(f"Similarity Score Example: {scores}")
