"""Model loader for inference.

This module reuses the shared training model loader so future checkpoints can
be swapped by updating only the model path in config.yaml.
"""

from __future__ import annotations

from typing import Any, Dict

from ..model_loader import Encoder, load_model as _load_shared_model


def load_model(config: Dict[str, Any]) -> Encoder:
    """Load the embedding model for inference."""
    return _load_shared_model(config)