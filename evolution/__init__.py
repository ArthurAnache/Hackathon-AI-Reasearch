"""L3 RESEARCH + L4 ORCHESTRATION: an evolutionary ("survival of the fittest") research loop.

    genes.py         Gene (config | formulation | operator), GenePool (Beta beliefs, validated, graveyard)
    operators.py     Thompson crossover, mutation (nudge / add / drop / LLM edit), knock-out plan, tournament
    llm.py           optional LLMEditor (OpenAI or Claude, $-capped): genome -> prompt -> new optimizer file -> operator gene
    evaluate.py      Spec -> parallel multi-seed evaluation, trusted re-verification, real_margin
    arms.py          hill_sampling (control), gene_evolution, formulation
    islands.py       independent lineages; migration of validated genes + best certificates via the stores
    orchestrator.py  meta-bandit over arms, rounds, records into the shared store
    tools.py         Exa (literature) + past.dev (gene verdicts shared between agents), both optional

Run: python -m evolution --problem 1a --optimizer optimizers/1a_tuned.py --islands 2 --rounds 4
"""
from .evaluate import Evaluation, Spec, evaluate_batch, real_margin
from .genes import Gene, GenePool, compatible, express, seed_genes
from .islands import Island, migrate
from .orchestrator import MetaBandit, Orchestrator
from .tools import ResearchTools

__all__ = ["Gene", "GenePool", "compatible", "express", "seed_genes", "Spec", "Evaluation", "evaluate_batch",
           "real_margin", "Island", "migrate", "MetaBandit", "Orchestrator", "ResearchTools"]
