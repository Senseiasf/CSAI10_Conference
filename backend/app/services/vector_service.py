import hashlib
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

VECTOR_DIM = 128


def generate_text_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """
    Генерирует детерминированный семантико-хэшевый эмбеддинг фиксированной размерности (128).
    Работает без тяжелых зависимостей и без обязательного внешнего API-ключа,
    нормализует вектор к единичной длине (L2 norm) для точного косинусного сходства в pgvector.
    """
    words = text.lower().split()
    vector = np.zeros(dim, dtype=np.float32)
    
    if not words:
        vector[0] = 1.0
        return vector.tolist()
        
    for word in words:
        # Проекция каждого слова через хэширование n-грамм
        h = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(min(len(h), dim)):
            # Псевдослучайный весовой знак из битов хэша
            sign = 1.0 if (h[i] % 2 == 0) else -1.0
            idx = (int(h[i]) + i * 7) % dim
            vector[idx] += sign * (float(h[i]) / 255.0)
            
    # L2-нормализация
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    else:
        vector[0] = 1.0
        
    return [float(x) for x in vector]
