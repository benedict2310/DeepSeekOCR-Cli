# hybrid_search.py
"""Text indexing and hybrid search functionality."""
from __future__ import annotations

import json
from pathlib import Path

import hnswlib
import numpy as np
from filelock import FileLock


def load_st_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a sentence transformer model for text embeddings.

    Args:
        name: Model name or path

    Returns:
        SentenceTransformer model instance
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class TextIndex:
    """HNSW index for text embeddings with document metadata."""

    def __init__(self, space="cosine", dim=None):
        """
        Initialize text index.

        Args:
            space: Distance metric (cosine, l2, ip)
            dim: Embedding dimension (auto-detected on build)
        """
        self.space = space
        self.dim = dim
        self.index = None
        self.docs = []  # [{path, name, doc_id, page_no}]

    def build(self, X: np.ndarray, docs: list, M=32, efC=200):
        """
        Build index from embeddings and document metadata.

        Args:
            X: Embeddings matrix [N, D]
            docs: List of document metadata dictionaries
            M: HNSW M parameter (connections per node)
            efC: HNSW ef_construction parameter
        """
        self.dim = X.shape[1]
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=len(X), ef_construction=efC, M=M)
        self.index.add_items(X, ids=np.arange(len(X)))
        self.index.set_ef(64)
        self.docs = docs

    def resize(self, new_max):
        """
        Resize index to accommodate more elements.

        Args:
            new_max: New maximum number of elements
        """
        self.index.resize_index(new_max)

    def add(self, X: np.ndarray, docs: list):
        """
        Add new embeddings to existing index.

        Args:
            X: Embeddings matrix [N, D]
            docs: List of document metadata dictionaries
        """
        start = len(self.docs)
        ids = np.arange(start, start + len(docs))
        self.index.add_items(X, ids=ids)
        self.docs.extend(docs)

    def save(self, folder: Path):
        """
        Save index and metadata to disk with file locking.

        Args:
            folder: Directory to save index files
        """
        folder.mkdir(parents=True, exist_ok=True)
        lock_file = folder / ".index.lock"

        with FileLock(str(lock_file), timeout=60):
            (folder / "dim.txt").write_text(str(self.dim), encoding="utf-8")
            self.index.save_index(str(folder / "hnsw.bin"))
            (folder / "docs.json").write_text(
                json.dumps(self.docs, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def load(self, folder: Path):
        """
        Load index and metadata from disk with file locking.

        Args:
            folder: Directory containing index files
        """
        lock_file = folder / ".index.lock"

        with FileLock(str(lock_file), timeout=60):
            self.dim = int((folder / "dim.txt").read_text().strip())
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(str(folder / "hnsw.bin"))
            self.index.set_ef(64)
            self.docs = json.loads((folder / "docs.json").read_text(encoding="utf-8"))

    def query(self, vec: np.ndarray, k=5):
        """
        Query index for similar documents.

        Args:
            vec: Query vector
            k: Number of results to return

        Returns:
            List of (document_metadata, similarity_score) tuples
        """
        lbl, dist = self.index.knn_query(vec.reshape(1, -1).astype(np.float32), k=k)
        sims = 1.0 - dist[0]
        return [(self.docs[int(i)], float(s)) for i, s in zip(lbl[0], sims)]


def hybrid_search(
    query: str,
    text_index: TextIndex,
    visual_index=None,
    query_image=None,
    text_model=None,
    visual_embedder=None,
    alpha=0.6,
    k=5,
):
    """
    Perform hybrid search combining text and visual similarity.

    Args:
        query: Text query string
        text_index: TextIndex instance
        visual_index: Optional VisualIndex instance
        query_image: Optional PIL image for visual search
        text_model: SentenceTransformer model for text embeddings
        visual_embedder: Optional pre-loaded DeepSeekVisionEmbedder (for reuse)
        alpha: Weight for text scores (1-alpha for visual scores)
        k: Number of results to return

    Returns:
        List of (metadata, combined_score) tuples
    """
    results = {}

    # Text search
    if query and text_model:
        qv = text_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        # Guard against small indexes: clamp k*2 to actual index size
        text_k = min(k * 2, len(text_index.docs)) if text_index.docs else k
        text_results = text_index.query(qv, k=text_k)
        for doc, score in text_results:
            key = doc.get("name", doc.get("path", "unknown"))
            results[key] = {"doc": doc, "text_score": score, "visual_score": 0.0}

    # Visual search
    if query_image and visual_index:
        # Use pre-loaded embedder if provided, otherwise create one
        # WARNING: Creating embedder here will reload model on every query
        if visual_embedder is None:
            from visual_index import DeepSeekVisionEmbedder

            visual_embedder = DeepSeekVisionEmbedder()

        qv = visual_embedder.embed_image(query_image)
        # Guard against small indexes: clamp k*2 to actual index size
        visual_k = min(k * 2, len(visual_index.meta)) if visual_index.meta else k
        visual_results = visual_index.query(qv, topk=visual_k)
        for meta, score in visual_results:
            key = meta.get("display", "unknown")
            if key in results:
                results[key]["visual_score"] = score
            else:
                results[key] = {"doc": meta, "text_score": 0.0, "visual_score": score}

    # Combine scores
    combined = []
    for _key, data in results.items():
        combined_score = alpha * data["text_score"] + (1 - alpha) * data["visual_score"]
        combined.append((data["doc"], combined_score))

    # Sort by combined score and return top k
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]
