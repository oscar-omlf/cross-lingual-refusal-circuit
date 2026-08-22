"""RQ1 utilities for tracing the circuit behind a refusal representation.

This package is intentionally independent of ``external/refusal_direction`` at
runtime. Method and provenance details are retained in saved result metadata
and the repository archive.
"""

DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"

__all__ = ["DEFAULT_MODEL_NAME"]
