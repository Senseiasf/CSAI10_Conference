from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    use_vector_search: bool = True
    use_strict_graph: bool = True
    ranking_mode: Literal["price", "heuristic", "catboost"] = "catboost"


EXPERIMENT_CONFIGS = {
    "C1_Baseline_SQL": PipelineConfig(
        name="C1: Baseline (SQL/Service)",
        use_vector_search=False,
        use_strict_graph=False,
        ranking_mode="price",
    ),
    "C2_Vector_RAG": PipelineConfig(
        name="C2: + pgvector Search",
        use_vector_search=True,
        use_strict_graph=False,
        ranking_mode="price",
    ),
    "C3_Graph_Heuristic": PipelineConfig(
        name="C3: + Neo4j Graph (Heuristic)",
        use_vector_search=True,
        use_strict_graph=True,
        ranking_mode="heuristic",
    ),
    "C4_Full_CatBoost": PipelineConfig(
        name="C4: Full Pipeline (+ CatBoost LTR)",
        use_vector_search=True,
        use_strict_graph=True,
        ranking_mode="catboost",
    ),
}

