# CSAI 2026: Hybrid Neuro-Symbolic Architecture for Multi-Constraint Travel Planning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector%20(384D)-336791.svg)](https://github.com/pgvector/pgvector)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.21%20Community%20(Strict%20Graph)-008cc1.svg)](https://neo4j.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10%20(LTR)-ffcc00.svg)](https://catboost.ai/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM--L12--v2%20(384D)-orange.svg)](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose%20v2-2496ed.svg)](https://www.docker.com/)

> **Active Branch:** `feat/exp-minilm-and-graph-validation`  
> **Key Innovations:** Native 384D SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`), strict Neo4j catalog topology validation, calibrated semantic vibes without lexical attractors, and optimal 200-tree CatBoost LTR.  

Official experimental benchmark stand and codebase for the research paper:
> **"Hybrid Neuro-Symbolic Routing and Learning-to-Rank for Complex Multi-Constraint Travel Synthesis"**  
> *Submitted to CSAI 2026 (10th International Conference on Computer Science and Artificial Intelligence)*

---

## 1. Project Transformation & Repository Scope

This repository originally originated from a conversational travel booking prototype and has been rigorously **refactored into an academic research stand** for controlled CSAI 2026 evaluation:

### What was removed (Production & Hackathon Overhead)
* **Web UI & Static Assets:** Deleted Jinja2 templates (`backend/app/templates/`) and static frontend assets (`backend/app/static/`).
* **Telegram Bot:** Deleted polling and webhook handlers (`backend/app/bot/`).
* **Interactive Dialogue Services:** Deleted stateful conversation tracking, multi-turn state machines, and booking stubs (`agent_service.py`, `dialogue_parser.py`, `trip_controller.py`, `preference_memory.py`, `booking_stub.py`, `calendar_service.py`).
* **Product-Level UI Tests:** Removed 17 product and dialogue test files, retaining pure ML and LTR regression tests.

### What is preserved (The Core Scientific Pipeline)
* **Symbolic Knowledge Graph:** Neo4j logistics graph validating routes, transfers, and layover windows.
* **Vector Semantic Retrieval:** PostgreSQL with `pgvector` performing 384-d cosine similarity search (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
* **Learning-to-Rank Engine:** CatBoost LTR model with 12 normalized value-for-money features.
* **Evaluation Suite:** Deterministic 200-query benchmark harness (`run_csai_benchmark.py`), publication figure plotting (`plot_csai_figures.py`), and regression tests.

---

## 2. Pipeline Architecture & Workflow

Modern LLMs struggle with hard temporal, topological, and budgetary constraints, frequently hallucinating unfeasible flight connections or violating luggage/pet policies. Our architecture decouples **soft natural language understanding** and **multi-criteria ranking** from **strict symbolic validation**:

```
                 [Raw User Query (Natural Language)]
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 1: Neural Intent Extraction (NLU / LLM)         │
      │  - Extracts: archetype, budget ceiling, traveler count │
      │  - Detects: preferred vibes, pet/kid flags, origin     │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 2: Dense Semantic Retrieval (pgvector)          │
      │  - 384-dimensional destination embeddings (MiniLM)     │
      │  - Cosine similarity ranking over catalog regions      │
      │  - Selects Top-2 candidate destinations                │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 3: Symbolic Feasibility Check (Neo4j Graph)     │
      │  - Flight schedule matching & layover threshold ≥ 60m  │
      │  - Ground transfer feasibility & nocturnal boat limits │
      │  - Hard pruning of physically impossible itineraries   │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │  Stage 4: Learning-to-Rank Selection (CatBoost LTR)    │
      │  - 12 normalized value-for-money features              │
      │  - Shrunk 200-tree PairLogit ranking model             │
      │  - Optimal candidate package selection                │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
             [Optimal Feasible & Personalized Itinerary]
```

---

## 3. Dense Retrieval Evolution: 128D Projection vs. Native 384D MiniLM

### The 128D Projection & "Lexical Attractor" Collapse
In initial experiments with neural embeddings, a random orthogonal QR projection was tested to compress 384-dimensional `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` down to the legacy 128D pgvector schema:
$$\mathbf{v}_{128} = \text{QR}(\mathbf{W}_{384 \times 128})^\top \mathbf{v}_{384}$$

While mathematically preserving pairwise distances in expectation, this compression caused an acute **semantic collapse**:
1. **Dimension Loss:** Discarding 66.7% of dimensions collapsed fine-grained cultural nuances into coarse lexical clusters.
2. **The "Turkey Attractor" Bug:** In the catalog, Turkey's description was heavily overloaded with generic travel buzzwords (*"все включено"*, *"отель"*, *"пляж"*, *"семейный отдых"*, *"аквапарк"*, *"лагуны"*, *"горы"*). Under 128D projection, Turkey acted as a semantic gravitational attractor, scoring top-1 in **172 out of 200 queries**.
3. **Catastrophic CSR Collapse (40.8%):** Because Turkey's catalog packages are predominantly premium/deluxe (hotels: 42,000–128,000 RUB/night + 44,500 RUB roundtrip flight), budget queries (50,000–120,000 RUB) assigned to Turkey could not satisfy user budget constraints, driving Constraint Satisfaction Rate (CSR@3) down to 40.8%.

### The Native 384D Solution with Calibrated Vibes
On this branch (`feat/exp-minilm-and-graph-validation`), the pipeline was upgraded to native 384D:
1. **Schema Upgrade:** PostgreSQL column migrated natively to `vector(384)`. pgvector performs sub-millisecond exact cosine distance computations ($<=>$) without dimensionality reduction.
2. **Lexical De-biasing:** All generic attractor words (*"отель"*, *"пляж"*, *"отдых"*, *"все включено"*) were stripped across all 9 catalog regions.
3. **Distinctive Semantic Markers:** Destination descriptions were calibrated with culturally distinct, natural semantic markers:
   - *Turkey:* Ottoman architecture, Bosphorus straits, Istanbul mosques, Hagia Sophia, Cappadocia hot air balloons.
   - *Egypt:* Red Sea coral reefs, marine fish, snorkeling, scuba diving, Giza pyramids, Sphinx.
   - *Sochi:* Black Sea coast, Caucasus mountains, Sirius promenade, Rosa Khutor hiking (domestic, no foreign passport).
   - *Altai:* Katun river, Chuysky tract, coniferous taiga, Siberian cedar, secluded chalets, quiet nature.
   - *St. Petersburg:* Neva granite embankments, drawbridges, Hermitage museum, Mariinsky theatre, white nights.
   - *Georgia:* Old Tbilisi carved balconies, sulfur baths, Georgian wine, khinkali, Gudauri freeride, Kazbek peak.
   - *Thailand:* Andaman Sea islands, Buddhist temples, tropical fruits, elephants, wintering, coworiking.
   - *Maldives:* Indian Ocean atolls, overwater villas on stilts, seaplane transfers, manta rays and whale sharks.
   - *UAE:* Arabian desert dunes, Burj Khalifa, futuristic skyline, luxury mega-malls, desert safari.

**Result:** Recall@2 surged to **80.0%** (up from 74.0%), while CSR@3 jumped to an all-time high of **90.3%** (up from 40.8% and exceeding baseline 77.5%).

---

## 4. CatBoost LTR Model & Training Specifications

### Motivation for Retraining
The legacy baseline CatBoost model suffered from:
1. **Tree Saturation:** Split borders capped at 237,500 RUB caused luxury atoll packages (e.g. Maldives resorts at 400,000–800,000 RUB) to receive identical tree leaves.
2. **Unconstrained Score Ties:** On open-ended queries without explicit constraints, identical scores led to arbitrary ranking.
3. **Overfitting Beyond 200 Trees:** Training on synthetic pools past 200 iterations degraded test-set pairwise accuracy from 89.5% down to 79.7%.

### Feature Schema (12 Features)
The model was retrained on **69,311 pairs** (`part_4060ti.jsonl` + `part_4080s.jsonl`) with engineered normalized features:

| Feature Name | Type | Description / Normalization Formula |
| :--- | :---: | :--- |
| `archetype` | Categorical | Target user persona (`family`, `budget`, `luxury`, `pet`, `solo`) |
| `cities` | Categorical | Canonical routing path (e.g. `Сочи -> Красная Поляна -> Сириус`) |
| `days` | Numerical | Length of stay in nights ($\ge 1$) |
| `log_price` | Numerical | $\ln(1 + \text{total\_price\_rub})$ — prevents luxury price saturation |
| `log_price_per_day` | Numerical | $\ln(1 + \frac{\text{total\_price\_rub}}{\text{days}})$ — daily cost rate |
| `transfers` | Numerical | Total segment transfers (connecting flights + ground shuttles) |
| `layover_hours` | Numerical | Total layover / connection buffer time in hours |
| `hotel_stars` | Numerical | Hotel star rating ($1..5$) |
| `dist_center_km` | Numerical | Distance to city / resort center in kilometers |
| `pet_friendly` | Binary | Hotel accepts pets ($1$ / $0$) |
| `breakfast` | Binary | Breakfast included in booking ($1$ / $0$) |
| `stars_per_10k` | Numerical | $\frac{\text{hotel\_stars}}{\max(0.1, \text{price\_per\_day} / 10\,000)}$ — value-for-money metric |

### Hyperparameters & Model Checkpoint
* **Loss Function:** `PairLogit` (Pairwise ranking)
* **Tree Depth:** 5
* **Border Count:** 254
* **Optimal Tree Shrinkage:** 200 trees (checkpoint: `backend/app/data/catboost_travel_ranker.cbm`, size: 278 KB)
* **Native Latency:** **0.91 ms** per candidate batch (no GPU required).

---

## 5. Benchmark Dataset and Configurations

The frozen catalogue benchmark contains 200 synthetic Russian-language queries,
40 per traveler archetype. The query file retains its historical name
`csai_150_benchmark_queries.json`. The generation pipeline cleans and
deduplicates texts, then records catalogue feasibility without changing budgets.
The three corrected reference budgets agree with the explicit sums in their text.

| Config | Retrieval | Candidate source and filtering | Ranking |
| --- | --- | --- | --- |
| C1 | Lexical names and aliases | Connected catalogue packages | Price |
| C2 | Explicit names, otherwise MiniLM cosine | Connected catalogue packages | Price |
| C3 | Same as C2 | Live Neo4j paths and hard predicates | Analytical heuristic |
| C4 | Same as C2 | Same as C3 | CatBoost |

Candidates from both retrieved destinations are considered. C1 returns an empty
result if it finds no explicit match. C3 and C4 return an empty result when all
candidates fail. Flight and transfer endpoints must match in every configuration.
The budget model uses catalogue round-trip estimates per traveler, hotel cost per
room-night, pet fees where present and the listed transfer cost per package.

## 6. Metrics and Run Types

- **Destination Hit@2** uses any match with a reference destination.
- **LFR@3** and **CSR@3** average per-query fractions of returned packages passing
  logistics and user constraints respectively. Empty responses contribute zero.
- **Success@3** counts queries with at least one returned package that satisfies
  relevance, logistics and all evaluated constraints simultaneously.
- **Coverage** counts queries with any returned package.
- **Pairwise Accuracy** and **NDCG@2** use the same pairs of admissible candidates
  for all rankers. Both pair positions are recorded before labeling. Exact score
  and price ties contribute half credit to accuracy and have their own counts.
- **Ranking latency** measures only ordering the candidates, in milliseconds.

`--slots oracle` evaluates downstream components on reference slots. `--slots live`
uses extracted slots and evaluates the resulting offers against the references.
`--judge pareto` creates diagnostic labels only where one admissible tour dominates
the other on the defined features. This test favors the price baseline by design
and does not measure preference on trade-offs. `--judge live` obtains A/B/tie
labels from the configured model with the actual departure time shown for each tour.
Live modes require valid provider settings and fail on unavailable dependencies.
Historical headline metrics use a different protocol. See the [audit](csai2026_benchmark_fix_audit.md)
for the corrected run and its scope.

## 7. Run and Reproduce

With seeded services in Docker Compose:

```bash
docker compose up -d --build
docker compose exec app python -m app.scripts.run_csai_benchmark --slots oracle --retrieval pgvector --judge pareto
docker compose exec app python -m app.scripts.plot_csai_figures
```

From `backend/` with a seeded Neo4j and local Python dependencies:

```bash
python -m unittest discover -s tests -v
python -m app.scripts.run_csai_benchmark --slots oracle --retrieval numpy --judge pareto
python -m app.scripts.plot_csai_figures
```

Set the database endpoints to the running services. The NumPy mode computes exact
cosine similarity from real 384D MiniLM embeddings. The pgvector mode checks that
stored embeddings match the catalogue snapshot and fails on stale or missing rows.
Use a freshly seeded graph when changing catalogue or branch. C3 and C4
apply explicit soft premiums after hard filtering and diversify the top three
while retaining the first offer. Use `--top3 score` to disable diversification
or `--queries` to run another reference set. See the
[preference evaluation](csai2026_preference_ranking.md) for results and scope.
Output includes
CSV, LaTeX, PNG/PDF and per-query, per-pair and failure diagnostics. Both scripts
accept `--output-dir` so different protocols can be saved independently.
`experiment_outputs/` is ignored by Git.

## 9. Repository Layout

```
.
├── backend/
│   ├── app/
│   │   ├── config.py                 # Pydantic v2 application configuration
│   │   ├── database.py               # Async SQLAlchemy & Neo4j driver pools
│   │   ├── main.py                   # FastAPI benchmark monitoring API
│   │   ├── models.py                 # PostgreSQL ORM models with pgvector Vector(384)
│   │   ├── seed.py                   # Database & graph seeder (schema migration & vibes)
│   │   ├── data/
│   │   │   ├── destinations.json     # 9 destinations, 1327 hotels, 276 flights
│   │   │   ├── csai_150_benchmark_queries.json  # 200 queries (40 per archetype)
│   │   │   └── catboost_travel_ranker.cbm      # Shrunk 200-tree CatBoost LTR model
│   │   ├── domain/
│   │   │   ├── experiment.py         # Benchmark data contracts
│   │   │   ├── route.py              # Flight/transfer route entities
│   │   │   └── trip.py               # Quote and tour version contracts
│   │   ├── scripts/
│   │   │   ├── run_csai_benchmark.py # Main benchmark harness (Method 2)
│   │   │   ├── plot_csai_figures.py  # Publication figure generator
│   │   │   ├── train_ltr.py          # CatBoost LTR training pipeline
│   │   │   └── benchmark_generation/ # 200-query generation & deduplication pipeline
│   │   └── services/
│   │       ├── compliance_service.py # Visa and border control rules
│   │       ├── graph_service.py      # Neo4j Cypher pathfinding & catalog validator
│   │       ├── llm_service.py        # NLU slot extraction client
│   │       ├── ltr_service.py        # CatBoost inference & feature engineering
│   │       └── vector_service.py     # Native 384D MiniLM pgvector search
│   └── tests/
│       ├── test_ltr_edge_cases.py    # Robustness, boundary conditions, fallbacks
│       └── test_ltr_service.py       # Feature extraction & schema unit tests
├── docker-compose.yml                # Multi-container orchestration (db, neo4j)
├── experiment_outputs/               # CSV, LaTeX tables, JSON, PNG/PDF figures
├── Makefile                          # Build targets (up, down, test, benchmark)
└── README.md                         # Benchmark stand documentation
```

---

## 10. License & Citation

This codebase is distributed under the **MIT License**.

```bibtex
@inproceedings{csai2026_neuro_symbolic_travel,
  title     = {Hybrid Neuro-Symbolic Routing and Learning-to-Rank for Complex Multi-Constraint Travel Synthesis},
  booktitle = {Proceedings of the 10th International Conference on Computer Science and Artificial Intelligence (CSAI 2026)},
  year      = {2026},
  pages     = {1--8}
}
```
