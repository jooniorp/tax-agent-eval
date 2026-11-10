"""Entry point for the folder layout in docs/plan.md; the code lives in tax_agent.evaluation.runner.

Normally run as `uv run eval`; `uv run python evals/run_eval.py` does the same.
"""
from tax_agent.evaluation.runner import main

if __name__ == "__main__":
    main()
