"""List the LLM providers usable now and, when OPENAI_API_KEY is available, the OpenAI model ids.

    python -m autoloop.models [--env-file PATH]

The env file is loaded at run time for the LLM keys only; keys are never printed.
"""
import argparse
import os
import shutil

from .loop import DEFAULT_ENV
from .policy import list_openai_models, load_env

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--env-file", default=DEFAULT_ENV)
    a = ap.parse_args()
    load_env(a.env_file)
    print(f"OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'not set'}; "
          f"OPENAI_MODEL: {os.environ.get('OPENAI_MODEL') or 'not set'}; "
          f"ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}; "
          f"claude CLI: {'found' if shutil.which('claude') else 'missing'}")
    if os.environ.get("OPENAI_API_KEY"):
        ids = list_openai_models()
        print(f"{len(ids)} OpenAI models:")
        print("\n".join(ids))
