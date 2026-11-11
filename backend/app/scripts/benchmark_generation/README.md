# Контур генерации и фильтрации бенчмарка CSAI 2026

Папка содержит модули создания и очистки эталонной выборки из 150 пользовательских сценариев:

1. `generate_raw_queries.py`:
   - Генератор первичного пула запросов по 5 архетипам (`budget`, `pet`, `family`, `luxury`, `solo`) и 9 направлениям.
   - Задает параметры: `flight_time_pref` (morning/day/evening), `prefer_direct_flight`, бюджет, длительность, дети, животные.

2. `filter_diversity.py`:
   - Модуль очистки по N-граммам и фильтрации дубликатов по матрице попарного TF-IDF косинусного сходства.
   - Подготовлен каркас функций `clean_text_ngrams`, `compute_tfidf_similarity_matrix`, `filter_by_diversity_threshold` и `audit_catalog_feasibility` для адаптации готового файла от вашей другой статьи.

3. Итоговый экспортируемый артефакт:
   - `backend/app/data/csai_150_benchmark_queries.json`

