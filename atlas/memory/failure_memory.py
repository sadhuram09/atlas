"""
atlas/memory/failure_memory.py

FailureMemory — FAISS-powered long-term memory of past failures and fixes.

This is what makes ATLAS genuinely learn over time.

How it works:
  1. A subtask fails → CoderAgent retries → tests pass
  2. We store the (error, broken_code, fixed_code) triple in FAISS
  3. Next time ANY task has a similar error, we search the index
  4. The top-K most similar past failures are returned
  5. The PromptEnhancer injects "here's how we fixed this before" into the prompt
  6. CoderAgent gets a head start — it's not solving from scratch

Why FAISS?
  FAISS (Facebook AI Similarity Search) is the industry standard for
  fast approximate nearest-neighbour search on embedding vectors.
  We embed error messages using sentence-transformers (runs locally, free).
  Search across 10,000 stored failures takes <10ms.

The embedding model:
  all-MiniLM-L6-v2 — 384 dimensions, 80MB, runs on CPU.
  Good enough for semantic similarity of Python error messages.
  Fast enough to not add latency to the retry loop.

Phase 3: In-memory FAISS index, persisted to disk as numpy arrays.
Phase 5: Swap for a managed vector DB (Pinecone, Weaviate) in production.

Fallback:
  If sentence-transformers isn't installed (e.g. CI environment),
  FailureMemory gracefully degrades to keyword matching.
  The rest of the system doesn't know or care.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from atlas.contracts_v3 import FailurePattern, MemorySearchResult
from atlas.logging import get_logger

log = get_logger(__name__)

# Where we persist the FAISS index and metadata between restarts
MEMORY_DIR = Path("atlas_memory")
INDEX_PATH = MEMORY_DIR / "failures.index"
META_PATH = MEMORY_DIR / "failures_meta.json"

# How many similar patterns to return
TOP_K = 3

# Minimum similarity to include (0.0-1.0)
# Below this threshold the pattern is too different to be useful
MIN_SIMILARITY = 0.3


class FailureMemory:
    """
    Semantic search over past failure patterns using FAISS.

    Usage:
        memory = FailureMemory()
        await memory.initialize()

        # Store a failure+fix pair
        await memory.store(pattern)

        # Search for similar failures before retrying
        results = await memory.search(error_output, top_k=3)
    """

    def __init__(self) -> None:
        self._index = None          # FAISS index
        self._embedder = None       # sentence-transformers model
        self._patterns: list[FailurePattern] = []
        self._embeddings: list[list[float]] = []
        self._available = False     # False if deps not installed
        self._use_faiss = False

    def initialize(self) -> None:
        """
        Load the embedding model and FAISS index.

        Called once at startup. Gracefully handles missing dependencies.
        """
        MEMORY_DIR.mkdir(exist_ok=True)

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("memory_embedder_loaded", model="all-MiniLM-L6-v2")
        except ImportError:
            log.warning(
                "memory_embedder_unavailable",
                message="sentence-transformers not installed — using keyword fallback",
            )
            self._available = True  # Keyword fallback still available
            self._load_metadata()
            return

        # Try to load FAISS
        try:
            import faiss
            import numpy as np

            if INDEX_PATH.exists() and META_PATH.exists():
                self._index = faiss.read_index(str(INDEX_PATH))
                self._load_metadata()
                log.info(
                    "memory_index_loaded",
                    patterns=len(self._patterns),
                    index_size=self._index.ntotal,
                )
            else:
                # Create a new flat L2 index (384 = MiniLM embedding size)
                self._index = faiss.IndexFlatIP(384)  # Inner product = cosine similarity
                log.info("memory_index_created", type="IndexFlatIP", dims=384)

            self._use_faiss = True
        except ImportError:
            log.warning(
                "memory_faiss_unavailable",
                message="faiss-cpu not installed — using keyword fallback",
            )

        self._available = True

    def _load_metadata(self) -> None:
        """Load pattern metadata from disk."""
        if META_PATH.exists():
            try:
                data = json.loads(META_PATH.read_text())
                self._patterns = [FailurePattern(**p) for p in data]
                log.info("memory_metadata_loaded", count=len(self._patterns))
            except Exception as e:
                log.warning("memory_metadata_load_failed", error=str(e))
                self._patterns = []

    def _save_metadata(self) -> None:
        """Persist pattern metadata to disk."""
        try:
            META_PATH.write_text(
                json.dumps([p.model_dump(mode="json") for p in self._patterns], indent=2)
            )
        except Exception as e:
            log.warning("memory_metadata_save_failed", error=str(e))

    def store(
        self,
        task_id: str,
        subtask_title: str,
        language: str,
        error_output: str,
        failed_code: str,
        fixed_code: str,
        test_count: int = 0,
    ) -> FailurePattern | None:
        """
        Store a failure+fix pair in the memory index.

        Called by the Orchestrator when a subtask passes after ≥1 retry.
        The error_output is the actual pytest output from the failed attempt.

        Returns the stored pattern, or None if storage failed.
        """
        if not self._available:
            return None

        # Extract error type from pytest output
        error_type = self._extract_error_type(error_output)
        error_summary = error_output[:500]

        # Generate fix description
        fix_description = self._describe_fix(failed_code, fixed_code)

        pattern = FailurePattern(
            id=str(uuid.uuid4()),
            task_id=task_id,
            subtask_title=subtask_title,
            language=language,
            error_type=error_type,
            error_summary=error_summary,
            failed_code=failed_code,
            fixed_code=fixed_code,
            fix_description=fix_description,
            test_count=test_count,
        )

        self._patterns.append(pattern)

        # Add to FAISS index if available
        if self._use_faiss and self._embedder is not None:
            try:
                import numpy as np
                import faiss

                embedding = self._embedder.encode([error_summary])[0]
                # Normalise for cosine similarity via inner product
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                self._index.add(np.array([embedding], dtype=np.float32))
                self._embeddings.append(embedding.tolist())

                # Persist index
                faiss.write_index(self._index, str(INDEX_PATH))

            except Exception as e:
                log.warning("memory_faiss_store_failed", error=str(e))

        self._save_metadata()

        log.info(
            "memory_pattern_stored",
            pattern_id=pattern.id,
            error_type=error_type,
            total_patterns=len(self._patterns),
        )

        return pattern

    def search(
        self,
        error_output: str,
        language: str = "python",
        top_k: int = TOP_K,
    ) -> list[MemorySearchResult]:
        """
        Find the most similar past failures to the current error.

        Returns top_k results sorted by similarity (highest first).
        Returns empty list if no relevant patterns found.

        Called before every retry — gives CoderAgent past fix context.
        """
        if not self._available or not self._patterns:
            return []

        error_summary = error_output[:500]

        if self._use_faiss and self._embedder is not None:
            results = self._faiss_search(error_summary, language, top_k)
        else:
            results = self._keyword_search(error_summary, language, top_k)

        log.info(
            "memory_search_complete",
            results=len(results),
            top_score=results[0].similarity_score if results else 0,
        )

        return results

    def _faiss_search(
        self, error_summary: str, language: str, top_k: int
    ) -> list[MemorySearchResult]:
        """FAISS semantic similarity search."""
        try:
            import numpy as np

            query_embedding = self._embedder.encode([error_summary])[0]
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            query = np.array([query_embedding], dtype=np.float32)
            k = min(top_k * 2, self._index.ntotal)  # Fetch extra, filter by language
            if k == 0:
                return []

            scores, indices = self._index.search(query, k)

            results = []
            rank = 1
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self._patterns):
                    continue
                pattern = self._patterns[idx]
                similarity = float(score)

                if similarity < MIN_SIMILARITY:
                    continue
                if language and pattern.language != language:
                    continue

                results.append(MemorySearchResult(
                    pattern=pattern,
                    similarity_score=round(similarity, 4),
                    rank=rank,
                ))
                rank += 1
                if rank > top_k:
                    break

            return results

        except Exception as e:
            log.warning("memory_faiss_search_failed", error=str(e))
            return self._keyword_search(error_summary, language, top_k)

    def _keyword_search(
        self, error_summary: str, language: str, top_k: int
    ) -> list[MemorySearchResult]:
        """
        Fallback keyword-based similarity when FAISS isn't available.

        Uses Jaccard similarity on word tokens — not semantic but better than nothing.
        """
        query_words = set(error_summary.lower().split())
        if not query_words:
            return []

        scored: list[tuple[float, FailurePattern]] = []

        for pattern in self._patterns:
            if language and pattern.language != language:
                continue

            pattern_words = set(pattern.error_summary.lower().split())
            if not pattern_words:
                continue

            # Jaccard similarity
            intersection = len(query_words & pattern_words)
            union = len(query_words | pattern_words)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= MIN_SIMILARITY:
                scored.append((similarity, pattern))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            MemorySearchResult(
                pattern=pattern,
                similarity_score=round(score, 4),
                rank=rank + 1,
            )
            for rank, (score, pattern) in enumerate(scored[:top_k])
        ]

    @staticmethod
    def _extract_error_type(error_output: str) -> str:
        """Extract the Python exception type from pytest output."""
        import re
        # Look for "ExceptionType: message" pattern
        match = re.search(r"([\w]+Error|[\w]+Exception|[\w]+Warning):", error_output)
        if match:
            return match.group(1)
        if "FAILED" in error_output:
            return "TestFailure"
        return "UnknownError"

    @staticmethod
    def _describe_fix(failed_code: str, fixed_code: str) -> str:
        """Generate a one-line description of what changed between attempts."""
        failed_lines = set(failed_code.splitlines())
        fixed_lines = set(fixed_code.splitlines())

        added = fixed_lines - failed_lines
        removed = failed_lines - fixed_lines

        if not added and not removed:
            return "Minor formatting or whitespace change"

        parts = []
        if removed:
            sample = next(iter(removed)).strip()[:60]
            parts.append(f"Removed: {sample}")
        if added:
            sample = next(iter(added)).strip()[:60]
            parts.append(f"Added: {sample}")

        return " | ".join(parts) if parts else "Code restructured"

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def is_available(self) -> bool:
        return self._available


# Singleton — shared across the entire application
failure_memory = FailureMemory()
