"""RQ1 utilities for tracing the circuit behind a refusal representation.

This package is intentionally independent of ``external/refusal_direction`` at
runtime.  The small pieces of methodology it adapts are acknowledged in
``docs/RQ1_SETUP.md``.
"""

DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"

__all__ = ["DEFAULT_MODEL_NAME"]
