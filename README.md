# CSAI 2026: Hybrid Neuro-Symbolic Architecture for Multi-Constraint Travel Planning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.21%20Community-008cc1.svg)](https://neo4j.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10%20(LTR)-ffcc00.svg)](https://catboost.ai/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose%20v2-2496ed.svg)](https://www.docker.com/)

Official experimental benchmark stand and codebase for the research paper:
> **"Hybrid Neuro-Symbolic Routing and Learning-to-Rank for Complex Multi-Constraint Travel Synthesis"**  
> *Submitted to CSAI 2026 (10th International Conference on Computer Science and Artificial Intelligence)*

---

## 1. Project Transformation & Repository Scope

This repository originally originated from a conversational travel booking prototype and has been rigorously **refactored into an academic research stand** to guarantee 100% deterministic reproducibility for the CSAI 2026 evaluation:

### What was removed (Production & Hackathon Overhead)
* **Web UI & Static Assets:** Deleted Jinja2 templates (`backend/app/templates/`) and static frontend assets (`backend/app/static/`).
* **Telegram Bot:** Deleted polling and webhook handlers (`backend/app/bot/`).
* **Interactive Dialogue Services:** Deleted stateful conversation tracking, multi-turn state machines, and booking stubs (`agent_service.py`, `dialogue_parser.py`, `trip_controller.py`, `preference_memory.py`, `booking_stub.py`, `calendar_service.py`).
* **Product-Level UI Tests:** Removed 17 product and dialogue test files, retaining pure ML and LTR regression tests.

### What is preserved (The Core Scientific Pipeline)
* **Symbolic Knowledge Graph:** Neo4j logistics graph validating routes, transfers, and layover windows.
* **Vector Semantic Retrieval:** PostgreSQL with `pgvector` performing 384-d cosine similarity search (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
* **Learning-to-Rank Engine:** CatBoost LTR model with 12 normalized value-for-money features.
* **Evaluation Suite:** Deterministic 200-query benchmark harness (`run_csai_benchmark.py`), publication figure plotting (`plot_csai_figures.py`), and 28 unit tests.

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

## 3. CatBoost LTR Model & Training Specifications

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

## 4. Benchmark Dataset & Methodology (CSAI-200)

### Dataset Composition
The benchmark comprises **200 realistic natural language queries** across 5 archetypes (40 queries each):
1. **Family with children:** High sensitivity to direct flights, breakfast inclusion, and child-friendly hotels.
2. **Budget travelers:** Strict budget ceilings, tolerance for layovers, priority on low price per day.
3. **Luxury seekers:** 5-star requirements, high amenity expectations, premium private transfers.
4. **Pet owners:** Strict `pet_friendly=1` requirement; violations are disqualifying.
5. **Solo travelers:** Flexibility in flights, focus on central location and mobility.

### Deduplication Protocol
Generated queries underwent a 3-tier diversity filter:
* Exact hash deduplication.
* Levenshtein distance filtering.
* TF-IDF + Cosine similarity thresholding ($< 0.70$ inter-query similarity).

### Evaluation Methodology (Method 2: In-Query Ranking)
Candidates are evaluated **within the specific context of the user query**:
* For each query $q$, candidate itineraries are generated from the database according to configuration constraints.
* Pairwise preference is judged by user persona satisfaction.
* **Uniform Baseline (C1):** In configuration C1 (unindexed SQL), destination selection rotates uniformly across the 9 catalog destinations rather than collapsing to `ORDER BY id LIMIT 2` (which previously caused an artificial Altai skew).

---

## 5. Experimental Results (Ablation Study)

Evaluated on 200 benchmark queries against live PostgreSQL (`pgvector`, native 384D MiniLM) and Neo4j Docker containers:

| Config ID | Pipeline Configuration | Dest Recall@2 (%) | Logistics Feasibility LFR@3 (%) | Constraint Satisfaction CSR@3 (%) | Task Completion Rate TCR@3 (%) | Pairwise Accuracy (%) | NDCG@2 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Baseline (SQL / Uniform / Price Sort) | 36.0% | 35.5% | 58.7% | 7.5% | 65.0% | 0.8708 | 0.02 |
| **C2** | + Dense Retrieval (`pgvector` 384D) | 80.0% | 39.5% | 65.5% | 20.7% | 65.0% | 0.8708 | 0.05 |
| **C3** | + Knowledge Graph (Neo4j Heuristic) | 80.0% | 83.5% | 89.5% | 59.8% | 87.5% | 0.9539 | 0.17 |
| **C4** | **Full Pipeline (+ CatBoost LTR)** | **80.0%** | **83.5%** | **90.3%** | **60.3%** | **89.50%** | **0.9585** | **1.41** |

### Mathematical Metric Definitions
* **Destination Recall@2:** $\frac{|\text{Retrieved}_2 \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$
* **Logistics Feasibility Rate (LFR@3):** Fraction of Top-3 packages with valid flight connections ($\ge 60$ min buffer) and ground transfers.
* **Constraint Satisfaction Rate (CSR@3):** Fraction of Top-3 packages meeting all explicit user constraints (budget, pets, kids, meal).
* **Task Completion Rate (TCR@3):** Joint fraction satisfying destination, logistics, and constraints simultaneously:
  $$\text{TCR@3} = \text{Recall@2} \times \text{LFR@3} \times \text{CSR@3}$$
* **Pairwise Accuracy:** Percentage of in-query candidate pairs correctly ordered according to ground-truth persona utility.
* **NDCG@2:** Normalized Discounted Cumulative Gain over the top-2 ranked itineraries.

---

## 6. Reproducibility & Quick Start

### 1. Start Docker Containers
```bash
docker compose up -d db neo4j
```
Verify that PostgreSQL (`5432`) and Neo4j (`7687`, `7474`) are healthy:
```bash
docker compose ps
```

### 2. Environment Configuration
Create `.env` from `.env.example`:
```bash
cp .env.example .env
```
Default connection parameters:
```env
POSTGRES_USER=app
POSTGRES_PASSWORD=app_secure_password
POSTGRES_DB=app
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secretpassword
```

### 3. Seed Knowledge Base
```bash
python -m app.seed
```
Seeds 9 destinations, 1,327 hotels, 276 flight routes, and 1,360 transfers into PostgreSQL and Neo4j.

### 4. Run Unit Tests (28 tests)
```bash
python -m unittest discover backend/tests
```

### 5. Run Benchmark & Generate Paper Artifacts
```bash
python backend/app/scripts/run_csai_benchmark.py
```
This automatically produces:
* `experiment_outputs/benchmark_results.json`
* `experiment_outputs/summary_table.csv`
* `experiment_outputs/summary_table.tex`
* `experiment_outputs/fig1_pipeline_ablation.png` / `.pdf`
* `experiment_outputs/fig2_archetype_ndcg.png` / `.pdf`
* `experiment_outputs/fig3_feature_importance.png` / `.pdf`

---

## 7. Safe Branching Guide for Team Collaboration

To experiment with new database schemas, graph queries, or vector indices without risking existing benchmark metrics:

```bash
# 1. Ensure you are on the clean stable branch
git checkout feat/exp-ltr-and-plots

# 2. Branch out to an isolated experiment sandbox
git checkout -b exp/test-db-updates

# 3. Apply your modifications or merge teammate's branch
git merge origin/<teammate-branch>

# 4. If database seeds changed, reseed the test environment
python -m app.seed

# 5. Re-run the benchmark
python backend/app/scripts/run_csai_benchmark.py

# 6. If metrics degrade, simply discard the branch and return:
git checkout feat/exp-ltr-and-plots
python -m app.seed
```

---

## 8. Repository Layout

```
.
├── backend/
│   ├── app/
│   │   ├── config.py                 # Pydantic v2 application configuration
│   │   ├── database.py               # Async SQLAlchemy & Neo4j driver pools
│   │   ├── main.py                   # FastAPI benchmark monitoring API
│   │   ├── models.py                 # PostgreSQL ORM models with pgvector Vector type
│   │   ├── seed.py                   # Database & graph seeder
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
│   │       ├── graph_service.py      # Neo4j Cypher pathfinding & logistics checks
│   │       ├── llm_service.py        # NLU slot extraction client
│   │       ├── ltr_service.py        # CatBoost inference & feature engineering
│   │       └── vector_service.py     # pgvector cosine similarity search
│   └── tests/
│       ├── test_ltr_edge_cases.py    # Robustness, boundary conditions, fallbacks
│       └── test_ltr_service.py       # Feature extraction & schema unit tests
├── docker-compose.yml                # Multi-container orchestration (db, neo4j)
├── experiment_outputs/               # CSV, LaTeX tables, JSON, PNG/PDF figures
├── Makefile                          # Build targets (up, down, test, benchmark)
└── README.md                         # Benchmark stand documentation
```

---

## 9. License & Citation

This codebase is distributed under the **MIT License**.

```bibtex
@inproceedings{csai2026_neuro_symbolic_travel,
  title     = {Hybrid Neuro-Symbolic Routing and Learning-to-Rank for Complex Multi-Constraint Travel Synthesis},
  booktitle = {Proceedings of the 10th International Conference on Computer Science and Artificial Intelligence (CSAI 2026)},
  year      = {2026},
  pages     = {1--8}
}
```
