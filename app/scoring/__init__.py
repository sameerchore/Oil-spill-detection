"""
Attribution Scoring Subsystem for TideTrace.
Provides multi-factor attribution scoring, explainability generation, and suspect ranking.
"""

from app.scoring.engine import (
    AttributionScoringEngine,
    ScoredSuspect,
    VESSEL_TYPE_SCORES,
    calculate_proximity_score,
    calculate_trajectory_score,
    compute_attribution_score,
    map_vessel_type_to_score,
    rank_suspects,
)
from app.scoring.explain import (
    format_forensic_summary,
    generate_reason_bullets,
    generate_reasons,
)

__all__ = [
    "AttributionScoringEngine",
    "ScoredSuspect",
    "VESSEL_TYPE_SCORES",
    "calculate_proximity_score",
    "calculate_trajectory_score",
    "compute_attribution_score",
    "map_vessel_type_to_score",
    "rank_suspects",
    "generate_reason_bullets",
    "generate_reasons",
    "format_forensic_summary",
]
