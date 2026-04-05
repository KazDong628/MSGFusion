"""Vendored OpenAI CLIP implementation (subset used by MSGFusion)."""

from .clip import available_models, load, tokenize

__all__ = ["available_models", "load", "tokenize"]
