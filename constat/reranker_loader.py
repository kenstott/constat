# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Background cross-encoder reranker model loader with thread-safe access.

Provides a singleton pattern for loading a CrossEncoder model
in a background thread, with thread-safe access that waits if needed.

Usage:
    # Start loading early (e.g., during vector store init)
    loader = RerankerModelLoader.get_instance()
    loader.start_loading("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Later, when you need the model (will wait if still loading)
    model = loader.get_model()
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerModelLoader:
    """Thread-safe singleton for background cross-encoder model loading."""

    _instance: Optional["RerankerModelLoader"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None
        self._loading = False
        self._loaded = threading.Event()
        self._load_error: Optional[Exception] = None
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "RerankerModelLoader":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_loading(self, model_name: str | None = None) -> None:
        """Start loading the model in a background thread.

        Safe to call multiple times - will only start loading once.

        Args:
            model_name: CrossEncoder model name. Uses DEFAULT_RERANKER_MODEL if None.
        """
        with self._lock:
            if self._loading or self._model is not None:
                return
            self._loading = True
            self._model_name = model_name or DEFAULT_RERANKER_MODEL

        self._thread = threading.Thread(target=self._load_model, daemon=True)
        self._thread.start()
        logger.debug(f"Started background reranker model loading: {self._model_name}")

    def _load_model(self) -> None:
        """Load the model (runs in background thread)."""
        try:
            logger.info(f"Loading reranker model: {self._model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self._load_error = e
        finally:
            self._loaded.set()

    def get_model(self, timeout: Optional[float] = None):
        """Get the loaded model, waiting if necessary.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The loaded CrossEncoder model.

        Raises:
            RuntimeError: If loading failed or timed out.
        """
        if not self._loading and self._model is None:
            self.start_loading()

        if not self._loaded.wait(timeout=timeout):
            raise RuntimeError(f"Reranker model loading timed out after {timeout}s")

        if self._load_error is not None:
            raise RuntimeError(f"Reranker model loading failed: {self._load_error}")

        return self._model

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._loaded.set()  # Unblock any waiters
                cls._instance = None
