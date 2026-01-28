from __future__ import annotations

import logging
from typing import List, Set, Dict
import numpy as np
import re
from sentence_transformers import SentenceTransformer

log = logging.getLogger("retriever.similarity_expander")


class SimilarityExpander:
    """
    Expands query terms by finding similar terms from catalog vocabulary
    using semantic embeddings instead of hard-coded dictionaries.
    """
    def __init__(
        self,
        embedder_model: SentenceTransformer,
        similarity_threshold: float = 0.5,
        max_expansions: int = 3,
    ):
        self.model = embedder_model
        self.similarity_threshold = similarity_threshold
        self.max_expansions = max_expansions

        # Vocabulary built from catalog
        self.vocabulary: List[str] = []
        self.vocab_embeddings: np.ndarray | None = None

    def build_vocabulary_from_catalog(self, docs: List[Dict]) -> None:
        """
        Extract unique terms from catalog documents and embed them.
        """
        vocab_set: Set[str] = set()

        for doc in docs:
            # Extract from key semantic fields
            tasks = doc.get("tasks", []) or []
            anatomy = doc.get("anatomy", []) or []
            modality = doc.get("modality", []) or []
            keywords = doc.get("keywords", []) or []

            for term in tasks + anatomy + modality + keywords:
                if not term or not isinstance(term, str):
                    continue
                term_clean = term.strip().lower()
                if term_clean and len(term_clean) > 2:
                    vocab_set.add(term_clean)

        self.vocabulary = sorted(vocab_set)
        log.info(f"Built vocabulary with {len(self.vocabulary)} unique terms")

        if not self.vocabulary:
            self.vocab_embeddings = None
            return

        # Embed vocabulary (batch for efficiency)
        log.info("Embedding vocabulary terms...")
        self.vocab_embeddings = self.model.encode(
            self.vocabulary,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        log.info("Vocabulary embedding complete")

    def expand_query(self, query: str) -> str:
        """
        Expand query by finding similar terms from catalog vocabulary.
        """
        if not self.vocabulary or self.vocab_embeddings is None:
            log.warning("Vocabulary not built, returning original query")
            return query

        # Tokenize query (simple word splitting)
        query_lower = query.lower()
        query_terms = [
            t for t in re.findall(r'\b[a-z0-9]+\b', query_lower)
            if len(t) > 2
        ]

        if not query_terms:
            return query

        # Find similar terms for each query term
        expansions: Set[str] = set()

        for term in query_terms:
            similar = self._find_similar_terms(term)
            expansions.update(similar)

        # Build expanded query
        if expansions:
            # Remove terms already in original query to avoid redundancy
            new_terms = [t for t in expansions if t not in query_lower]
            if new_terms:
                expansion_str = " ".join(sorted(new_terms)[:10])  # Cap at 10 to avoid bloat
                return f"{query} {expansion_str}"

        return query

    def _find_similar_terms(self, term: str) -> List[str]:
        """
        Find vocabulary terms similar to the given term.
        """
        if not self.vocabulary or self.vocab_embeddings is None:
            return []

        # Exact match already in vocabulary
        if term in self.vocabulary:
            term_idx = self.vocabulary.index(term)
        else:
            # Embed the term
            term_emb = self.model.encode(
                [term],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")

            # Find most similar terms
            similarities = np.dot(self.vocab_embeddings, term_emb.T).flatten()
            term_idx = None
            # Use similarities directly
            scores = similarities
        
        # If exact match exists, use its embedding
        if term_idx is not None:
            scores = np.dot(self.vocab_embeddings, self.vocab_embeddings[term_idx])
        else:
            pass

        # Get top matches above threshold
        candidates = []
        for idx, score in enumerate(scores):
            if score >= self.similarity_threshold:
                vocab_term = self.vocabulary[idx]
                if vocab_term != term:  # Exclude exact match
                    candidates.append((vocab_term, float(score)))

        # Sort by score descending and take top K
        candidates.sort(key=lambda x: -x[1])
        return [term for term, _ in candidates[:self.max_expansions]]

    def suggest_alternative_queries(
        self,
        original_query: str,
        num_alternatives: int = 2,
    ) -> List[str]:
        """
        Generate alternative query phrasings by replacing terms with similar ones.
        """
        if not self.vocabulary or self.vocab_embeddings is None:
            return []

        query_lower = original_query.lower()
        query_terms = [
            t for t in re.findall(r'\b[a-z0-9]+\b', query_lower)
            if len(t) > 2
        ]

        if not query_terms:
            return []

        alternatives = []

        # Strategy 1: Replace key terms with most similar neighbor
        for i in range(min(num_alternatives, len(query_terms))):
            if i >= len(query_terms):
                break

            term = query_terms[i]
            similar = self._find_similar_terms(term)

            if similar:
                # Replace term with top similar term
                alt_query = query_lower
                alt_query = alt_query.replace(term, similar[0])
                if alt_query != query_lower:
                    alternatives.append(alt_query)

        # Strategy 2: Broaden query by using more general terms
        # Look for more general terms (shorter, higher frequency in catalog)
        if len(alternatives) < num_alternatives:
            # Use first half of most similar terms (likely more general)
            general_terms = set()
            for term in query_terms:
                similar = self._find_similar_terms(term)
                if similar:
                    general_terms.add(similar[0])

            if general_terms:
                alt_query = " ".join(general_terms)
                if alt_query not in alternatives:
                    alternatives.append(alt_query)

        return alternatives[:num_alternatives]