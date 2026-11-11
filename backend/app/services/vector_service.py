"""Multilingual sentence embeddings projected into the existing pgvector schema."""
from functools import lru_cache
import re
import numpy as np

VECTOR_DIM = 128
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_VERSION = "minilm-multilingual-128-v1"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")

@lru_cache(maxsize=4)
def projection(source_dim: int, target_dim: int) -> np.ndarray:
    # Fixed projection keeps queries and catalogue in the same vector space.
    matrix = np.random.default_rng(42).normal(size=(source_dim, target_dim))
    return np.linalg.qr(matrix)[0].astype(np.float32)

def generate_text_embedding(text: str, dim: int = VECTOR_DIM) -> list[float]:
    normalized = " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))
    if not normalized:
        raise ValueError("Search text must contain letters or numbers")
    vector = get_embedding_model().encode(normalized, convert_to_numpy=True)
    if not 1 <= dim <= len(vector):
        raise ValueError("Invalid embedding dimension")
    if dim < len(vector):
        vector = vector @ projection(len(vector), dim)
    return (vector / np.linalg.norm(vector)).tolist()
