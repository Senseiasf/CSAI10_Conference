import hashlib
import re
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

VECTOR_DIM = 128

# Семантические якоря предметной области (первые 12 компонент вектора)
SEMANTIC_CLUSTERS = [
    ("море", "морск", "пляж", "побереж", "купат", "загар", "океан", "бухт", "риф", "снорклинг", "дайвинг"),
    ("гор", "вершин", "ущел", "лыж", "сноуборд", "канатк", "треккинг", "поход", "альп", "кавказ", "алтай"),
    ("семь", "дет", "ребен", "ребён", "малыш", "аквапарк", "анимац", "коляск", "семейн"),
    ("люкс", "премиум", "роскош", "спа", "вилла", "бассейн", "пятизвезд", "5 звезд", "vip", "яхт"),
    ("бюджет", "дешев", "недорог", "эконом", "выгодн", "хостел", "доступн", "студенч"),
    ("питом", "собак", "кошк", "животн", "пес", "пёс", "pet"),
    ("истор", "музе", "архитектур", "дворц", "экскурс", "культур", "древн", "храм", "собор"),
    ("природ", "озер", "озёр", "лес", "тайг", "водопад", "воздух", "тишин", "уедин", "эко"),
    ("тусовк", "ночн", "клуб", "бар", "шопинг", "мегаполис", "небоскреб", "современ"),
    ("тепл", "жарк", "солнечн", "тропик", "экзотик", "пальм", "остров", "зимовк"),
    ("гастро", "вин", "кухн", "ресторан", "вкусн", "хачапур", "морепродукт", "все включено", "all inclusive"),
    ("романтик", "вдвоем", "медов", "закат", "панорам", "уютн"),
]


def generate_text_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """
    Генерирует семантически согласованный 128-мерный вектор:
    - Компоненты [0..len(SEMANTIC_CLUSTERS)-1]: активация предметных концептов (устойчива к словоформам).
    - Компоненты [len(SEMANTIC_CLUSTERS)..dim-1]: проекция символьных 3-грамм и 4-грамм очищенных токенов.
    Нормализуется по L2 для косинусного оператора `<=>` в pgvector.
    """
    clean_text = re.sub(r"[^а-яёa-z0-9\s]", " ", (text or "").lower().replace("ё", "е"))
    tokens = [t for t in clean_text.split() if len(t) >= 2]
    vector = np.zeros(dim, dtype=np.float32)

    if not tokens:
        vector[0] = 1.0
        return vector.tolist()

    joined = " ".join(tokens)
    offset = len(SEMANTIC_CLUSTERS)

    # 1. Семантические концепт-якоря (вес 2.5 для сильного совпадения по смыслу)
    for idx, stems in enumerate(SEMANTIC_CLUSTERS):
        matches = sum(1 for stem in stems if stem in joined)
        if matches > 0:
            vector[idx] = 2.5 * min(3.0, float(matches))

    # 2. Символьные n-граммы (n=3, 4) для морфологической близости однокоренных слов
    ngram_dim = dim - offset
    for token in tokens:
        padded = f"<{token}>"
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                ngram = padded[i : i + n]
                digest = hashlib.md5(ngram.encode("utf-8")).digest()
                bucket = offset + (int.from_bytes(digest[:2], "little") % ngram_dim)
                sign = 1.0 if (digest[2] % 2 == 0) else -1.0
                vector[bucket] += sign * 0.35

    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    else:
        vector[0] = 1.0

    return [round(float(x), 6) for x in vector]
