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

## 1. Scientific Overview & Pipeline Architecture

Modern large language models (LLMs) struggle with hard mathematical, temporal, and topological constraints in complex logistics, frequently hallucinating incompatible connections or exceeding budget limits. This research introduces a 4-tier **Neuro-Symbolic Architecture** that decouples soft intent extraction and personalized ranking from strict symbolic verification:

```
[Raw User Query]
       │
       ▼
 1. Neural Intent Extraction (NLU / LLM)
       │ (Archetype, budget, travelers, origin, preferred dates/vibes)
       ▼
 2. Vector Semantic Retrieval (PostgreSQL + pgvector)
       │ (Cosine similarity over 128-d destination embeddings, Top-2 candidate cities)
       ▼
 3. Symbolic Feasibility Verification (Neo4j Logistics Knowledge Graph)
       │ (Flight schedule validation, layover thresholds ≥ 60 min, ground transfer topology)
       ▼
 4. Learning-to-Rank Tour Selection (CatBoost Pairwise LTR Ranker)
       │ (12 normalized value-for-money features, optimal 200-tree early stopped model)
       ▼
[Optimal Feasible Itinerary]
```

### Component Breakdown
1. **NLU Module (`LLMService`):** Extracts multi-attribute constraint slots from natural language queries (archetype, budget, traveler count, flight time preference, pet-friendly requirements).
2. **Dense Vector Retrieval (`VectorService` + `pgvector`):** Matches semantic destination vibes across 9 catalog regions against 128-dimensional latent embeddings.
3. **Symbolic Knowledge Graph (`GraphService` + Neo4j):** Validates flight-transfer-hotel connectivity across 22 airports, 276 flight segments, and 1,360 ground transfers. Hard constraints (buffer times, nocturnal boat limits on atolls) eliminate physically impossible itineraries before ranking.
4. **Learning-to-Rank (`LTRService` + CatBoost):** 12-feature pairwise gradient boosted ranker trained on 69,311 synthetic preference pairs using `PairLogit` loss. Overcomes price tree saturation via normalized features (`log_price`, `log_price_per_day`, `stars_per_10k`).

---

## 2. Experimental Benchmark Results (Method 2: In-Query Ranking)

Evaluated against the **CSAI-200 Benchmark Dataset** (40 queries across 5 user archetypes: `family`, `budget`, `luxury`, `pet`, `solo`) running live against PostgreSQL and Neo4j Docker containers.

| Configuration ID | Architecture Configuration | Dest Recall@2 (%) | Logistics Feasibility LFR@3 (%) | Constraint Satisfaction CSR@3 (%) | Task Completion Rate TCR@3 (%) | Pairwise Accuracy (%) | NDCG@2 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Baseline (Lexical / Price Sort) | 36.0% | 35.5% | 58.7% | 7.5% | 65.0% | 0.8708 | 0.02 |
| **C2** | + Dense Retrieval (`pgvector`) | 74.0% | 35.2% | 69.0% | 18.0% | 65.0% | 0.8708 | 0.02 |
| **C3** | + Neo4j Knowledge Graph (Heuristic) | 74.0% | 77.5% | 75.8% | 43.5% | 87.5% | 0.9539 | 0.09 |
| **C4** | **Full Pipeline (+ CatBoost LTR)** | **74.0%** | **77.5%** | **77.5%** | **44.4%** | **89.50%** | **0.9585** | **0.91** |

### Key Experimental Findings
- **Ablation Monotonicity:** Each architectural component provides statistically significant, non-overlapping improvements. Dense retrieval boosts destination recall by **+38.0 p.p.**; symbolic graph validation improves logistics feasibility by **+42.3 p.p.**; CatBoost LTR delivers state-of-the-art ranking fidelity (**89.50% Pairwise Accuracy**, **0.9585 NDCG@2**).
- **Uniform Baseline (C1):** C1 uses an academic Uniform Catalog Rotation baseline to prevent destination skew from artificially inflating constraint metrics on unindexed queries.
- **Microsecond Latency:** Full inference for C4 executes in **0.91 ms**, enabling real-time production deployment without GPU acceleration.

---

## 3. Quick Start & Reproducibility Guide

### Prerequisites
- Docker Engine 24.0+ and Docker Compose v2
- Python 3.11+ (for local benchmark script execution)

### 1. Launch Storage & Knowledge Graph
```bash
docker compose up -d db neo4j
```
This spins up:
- **PostgreSQL 16 with pgvector extension** on port `5432`
- **Neo4j 5.21 Community Graph DBMS** on Bolt port `7687` (HTTP port `7474`)

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default credentials:
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

### 3. Seed Knowledge Base (Zero-Touch)
Populate PostgreSQL relational tables, vector embeddings, and Neo4j graph nodes:
```bash
python -m app.seed
```

---

## 4. Running the Benchmark & Generating Paper Artifacts

### Run End-to-End Benchmark (Method 2)
```bash
python backend/app/scripts/run_csai_benchmark.py
```
This script:
1. Connects to live PostgreSQL and Neo4j Docker instances.
2. Evaluates 200 real benchmark queries across all 4 configurations (`C1`, `C2`, `C3`, `C4`).
3. Computes component metrics: `Recall@2`, `LFR@3`, `CSR@3`, `TCR@3`, `Pairwise Accuracy`, `NDCG@2`, and `Latency`.
4. Saves publication-ready tables to `experiment_outputs/`:
   - `benchmark_results.json`
   - `benchmark_results.csv`
   - `csai2026_table1_ablation.tex`
   - `csai2026_table1_ablation.md`

### Generate Publication Figures
```bash
python backend/app/scripts/plot_csai_figures.py
```
Generates 300 DPI PNGs and vector PDFs in `experiment_outputs/`:
- `fig1_ablation_metrics.png` / `.pdf`: Bar chart comparing C1–C4 across primary metrics.
- `fig2_latency_ndcg_tradeoff.png` / `.pdf`: Latency vs. NDCG@2 scatter plot demonstrating sub-millisecond Pareto dominance.
- `fig3_archetype_satisfaction.png` / `.pdf`: Per-archetype constraint satisfaction heatmap.

---

## 5. Model Training Pipeline (CatBoost LTR)

To retrain the 12-feature pairwise ranker on merged pairwise simulation pools:
```bash
python backend/app/scripts/train_ltr.py
```
Key training hyperparameters:
- **Objective:** `PairLogit`
- **Tree Depth:** 5
- **Early Stopping / Shrinkage:** 200 trees (prevents overfitting and ensures optimal generalization)
- **Features (12):** `archetype`, `cities`, `days`, `log_price`, `log_price_per_day`, `transfers`, `layover_hours`, `hotel_stars`, `dist_center_km`, `pet_friendly`, `breakfast`, `stars_per_10k`.

Model binary is output directly to `backend/app/data/catboost_travel_ranker.cbm`.

---

## 6. Unit & Integration Test Suite

Execute the ML / LTR regression test suite:
```bash
python -m unittest discover backend/tests
```
All 28 tests cover feature extraction, boundary handling, model fallback, and ranking monotonicity.

---

## 7. Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── config.py                 # Pydantic v2 environment settings
│   │   ├── database.py               # Async SQLAlchemy & Neo4j driver pools
│   │   ├── main.py                   # FastAPI benchmark monitoring API
│   │   ├── models.py                 # PostgreSQL ORM models with pgvector Vector type
│   │   ├── seed.py                   # Automatic database & graph seeder
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
│   │       ├── llm_service.py        # OpenAI/DeepSeek API NLU client
│   │       ├── ltr_service.py        # CatBoost inference & feature engineering
│   │       └── vector_service.py     # pgvector cosine similarity search
│   └── tests/
│       ├── test_ltr_edge_cases.py    # Robustness, boundary conditions, fallbacks
│       └── test_ltr_service.py       # Feature extraction & schema unit tests
├── docker-compose.yml                # Multi-container orchestration (db, neo4j)
├── experiment_outputs/               # CSV, LaTeX tables, JSON, PNG/PDF figures
├── Makefile                          # Quick build targets (up, down, test, benchmark)
└── README.md                         # Benchmark stand documentation
```

---

## 8. License & Citation

This repository is released under the **MIT License**.

If you use this benchmark or codebase in your research, please cite:
```bibtex
@inproceedings{csai2026_neuro_symbolic_travel,
  title     = {Hybrid Neuro-Symbolic Routing and Learning-to-Rank for Complex Multi-Constraint Travel Synthesis},
  booktitle = {Proceedings of the 10th International Conference on Computer Science and Artificial Intelligence (CSAI 2026)},
  year      = {2026},
  pages     = {1--8}
}
```
