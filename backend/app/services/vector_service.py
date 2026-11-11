"""Native 384-dimensional multilingual sentence embeddings for PostgreSQL pgvector."""
from functools import lru_cache
import re
from typing import List
import numpy as np

VECTOR_DIM = 384
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_VERSION = "minilm-multilingual-384-v2"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")


def generate_text_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """
    Генерирует честный 384-мерный нормализованный вектор SentenceTransformer MiniLM.
    Без потерь размерности и искажений случайными проекциями.
    """
    normalized = " ".join(re.findall(r"\w+", (text or "").casefold().replace("ё", "е")))
    if not normalized:
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        return v.tolist()

    vector = get_embedding_model().encode(normalized, convert_to_numpy=True, normalize_embeddings=True)
    if len(vector) != dim:
        # Резервный срез/дополнение на случай нестандартной размерности
        if len(vector) > dim:
            vector = vector[:dim]
        else:
            vector = np.pad(vector, (0, dim - len(vector)))
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    return [float(x) for x in vector]
