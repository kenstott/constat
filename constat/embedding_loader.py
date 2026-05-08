# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Background embedding model loader with thread-safe access.

Provides a singleton pattern for loading the SentenceTransformer model
in a background thread, with thread-safe access that waits if needed.

Usage:
    # Start loading early (e.g., during session init)
    loader = EmbeddingModelLoader.get_instance()
    loader.start_loading()

    # Later, when you need the model (will wait if still loading)
    model = loader.get_model()
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


class EmbeddingModelLoader:
    """Thread-safe singleton for background embedding model loading.

    The model is loaded in a background thread to avoid blocking the main
    thread during initialization. When the model is needed, callers can
    use get_model() which will wait if loading is still in progress.
    """

    _instance: Optional["EmbeddingModelLoader"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._loading = False
        self._loaded = threading.Event()
        self._load_error: Optional[Exception] = None
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "EmbeddingModelLoader":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_loading(self) -> None:
        """Start loading the model in a background thread.

        Safe to call multiple times - will only start loading once.
        """
        with self._lock:
            if self._loading or self._model is not None:
                return
            self._loading = True

        self._thread = threading.Thread(target=self._load_model, daemon=True)
        self._thread.start()
        logger.debug("Started background embedding model loading")

    def _load_model(self) -> None:
        """Load the model (runs in background thread)."""
        try:
            logger.info(f"Starting to download/load embedding model: {EMBEDDING_MODEL}")
            logger.info("This may take a few minutes on first run...")

            from sentence_transformers import SentenceTransformer

            # Try loading with different strategies to handle meta tensor issues
            # that can occur with accelerate/device_map on MPS
            strategies = [
                # Strategy 1: Load to CPU first (most reliable)
                {"device": "cpu"},
                # Strategy 2: Explicit CPU with model_kwargs to disable low_cpu_mem_usage
                {"device": "cpu", "model_kwargs": {"low_cpu_mem_usage": False}},
            ]

            last_error = None
            for i, kwargs in enumerate(strategies):
                try:
                    logger.info(f"Trying loading strategy {i + 1}/{len(strategies)}: {kwargs}")
                    self._model = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
                    logger.info(f"Embedding model loaded successfully!")
                    return
                except Exception as e:
                    last_error = e
                    logger.warning(f"Strategy {i + 1} failed: {e}")
                    continue

            # All strategies failed
            raise last_error

        except Exception as e:
            logger.error(f"FATAL: Failed to load embedding model after trying all strategies: {e}")
            logger.exception("Full error:")
            self._load_error = e
        finally:
            self._loaded.set()

    def get_model(self, timeout: Optional[float] = None):
        """Get the loaded model, waiting if necessary.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            RuntimeError: If loading failed or timed out.
        """
        # Start loading if not already started
        if not self._loading and self._model is None:
            self.start_loading()

        # Wait for loading to complete
        if not self._loaded.wait(timeout=timeout):
            raise RuntimeError(f"Embedding model loading timed out after {timeout}s")

        if self._load_error is not None:
            raise RuntimeError(f"Embedding model loading failed: {self._load_error}")

        return self._model

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def is_loading(self) -> bool:
        """Check if loading is in progress."""
        return self._loading and not self._loaded.is_set()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._loaded.set()  # Unblock any waiters
                cls._instance = None
