# visual_index.py
"""Visual embeddings extraction and HNSW indexing for DeepSeek-OCR."""
from __future__ import annotations

import json
from pathlib import Path

import hnswlib
import numpy as np
import torch
from filelock import FileLock


class DeepSeekVisionEmbedder:
    """Extracts vision embeddings from DeepSeek-OCR model."""

    def __init__(
        self,
        model_id="deepseek-ai/DeepSeek-OCR",
        device=None,
        dtype=torch.float32,
        model=None,
        tokenizer=None,
    ):
        """
        Initialize the vision embedder.

        NOTE: Since CLIP is now used for embeddings (not DeepSeek vision encoder),
        the model and tokenizer are only loaded if explicitly provided.
        For new instances, CLIP is lazily loaded in embed_image() instead.

        Args:
            model_id: HuggingFace model identifier (used if model is None) [DEPRECATED]
            device: Device to use (mps, cuda, or cpu)
            dtype: Data type for model weights (only used when loading new model)
            model: Pre-loaded model instance (optional, for reuse with legacy code)
            tokenizer: Pre-loaded tokenizer instance (optional, for reuse with legacy code)
        """
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_id = model_id
        self.dtype = dtype

        # Only load DeepSeek model if explicitly provided (for backward compatibility)
        if model is not None and tokenizer is not None:
            self.model = model
            self.tok = tokenizer
            self.proc = None
        else:
            # Don't load DeepSeek model - CLIP will be loaded lazily in embed_image()
            self.model = None
            self.tok = None
            self.proc = None

    def _preprocess(self, pil_image):
        """
        Preprocess PIL image for model input.

        Args:
            pil_image: PIL Image object

        Returns:
            Dictionary with preprocessed inputs
        """
        import numpy as np

        img = pil_image.convert("RGB")
        w, h = img.size
        s = 1024 / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0) / 255.0
        ids = self.tok("<image>\n", return_tensors="pt").input_ids
        return {"images": t.to(self.device), "input_ids": ids.to(self.device)}

    def embed_image(self, pil_image) -> np.ndarray:
        """
        Extract vision embedding from PIL image.

        Args:
            pil_image: PIL Image object

        Returns:
            Normalized embedding vector as numpy array
        """
        # DeepSeek-OCR has a very custom vision architecture that's difficult to extract embeddings from.
        # Instead, use CLIP from sentence-transformers for reliable visual embeddings.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers required for visual embeddings. Install with: pip install sentence-transformers"
            ) from None

        # Use CLIP for vision embeddings (more reliable than custom DeepSeek vision model)
        if not hasattr(self, "_clip_model"):
            self._clip_model = SentenceTransformer("clip-ViT-B-32")

        # Convert PIL image to CLIP embedding (normalized for cosine similarity)
        # Note: encode() expects a list/sequence, so wrap single image in a list
        embedding = self._clip_model.encode(
            [pil_image], convert_to_numpy=True, normalize_embeddings=True
        )[
            0
        ]  # Take first (and only) embedding
        return embedding.astype(np.float32)


class VisualIndex:
    """HNSW index for visual embeddings with metadata storage."""

    def __init__(self, space="cosine", dim=None):
        """
        Initialize visual index.

        Args:
            space: Distance metric (cosine, l2, ip)
            dim: Embedding dimension (auto-detected on build)
        """
        self.space = space
        self.dim = dim
        self.index = None
        self.meta = []  # list of dicts: {id, doc_id, page_id, display}

    def build(self, X: np.ndarray, meta: list, M=32, efC=200):
        """
        Build index from embeddings and metadata.

        Args:
            X: Embeddings matrix [N, D]
            meta: List of metadata dictionaries
            M: HNSW M parameter (connections per node)
            efC: HNSW ef_construction parameter
        """
        self.dim = X.shape[1]
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=len(X), ef_construction=efC, M=M)
        self.index.add_items(X, ids=np.arange(len(X)))
        self.index.set_ef(64)
        self.meta = meta

    def resize(self, new_max):
        """
        Resize index to accommodate more elements.

        Args:
            new_max: New maximum number of elements
        """
        self.index.resize_index(new_max)

    def add(self, X: np.ndarray, meta: list):
        """
        Add new embeddings to existing index.

        Args:
            X: Embeddings matrix [N, D]
            meta: List of metadata dictionaries
        """
        start = len(self.meta)
        ids = np.arange(start, start + len(meta))
        self.index.add_items(X, ids=ids)
        self.meta.extend(meta)

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
            (folder / "meta.json").write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
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
            self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))

    def query(self, vec: np.ndarray, topk=5):
        """
        Query index for similar vectors.

        Args:
            vec: Query vector
            topk: Number of results to return

        Returns:
            List of (metadata, similarity_score) tuples
        """
        lbl, dist = self.index.knn_query(vec.reshape(1, -1).astype(np.float32), k=topk)
        sims = 1.0 - dist[0]
        return [(self.meta[int(i)], float(s)) for i, s in zip(lbl[0], sims)]
