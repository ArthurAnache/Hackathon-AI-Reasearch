"""Research feedback framework: turns experiment results into the next experiment.

    Space       the tunable knobs of an optimizer (CONFIG + optional SEARCH_SPACE)
    Knowledge   what the results so far say (knob effects, strategy track record, trust region, failures)
    strategies  pluggable proposal generators (coordinate, surrogate, combine, explore, evidence, refine)
    FeedbackAgent  the ResearchAgent of the contract: picks a strategy by its track record
"""
from .agent import FeedbackAgent
from .knowledge import Knowledge
from .space import INIT_KEY, Knob, Space
from .strategies import (Candidate, CombineWins, CoordinateStep, EvidenceHints, Refine, Surrogate,
                         TrustRegion, default_strategies)

__all__ = ["FeedbackAgent", "Knowledge", "Space", "Knob", "INIT_KEY", "Candidate", "CoordinateStep",
           "Surrogate", "TrustRegion", "CombineWins", "EvidenceHints", "Refine", "default_strategies"]
