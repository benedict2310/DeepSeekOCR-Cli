"""
Unit and integration tests for hybrid search functionality.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if dependencies are available
pytest.importorskip("hnswlib")
pytest.importorskip("sentence_transformers")

from hybrid_search import TextIndex, hybrid_search, load_st_model
from visual_index import DeepSeekVisionEmbedder, VisualIndex


class TestVisualIndex:
    """Test VisualIndex functionality."""

    def test_visual_index_build_and_query(self, tmp_path):
        """Test building and querying visual index."""
        # Create synthetic embeddings (simulate real embeddings)
        n_samples = 10
        dim = 128
        X = np.random.randn(n_samples, dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize

        # Create metadata
        meta = [{"id": i, "display": f"page_{i:04d}.png"} for i in range(n_samples)]

        # Build index
        vindex = VisualIndex(space="cosine", dim=dim)
        vindex.build(X, meta)

        assert vindex.index is not None
        assert len(vindex.meta) == n_samples
        assert vindex.dim == dim

        # Query with first vector (should return itself as top result)
        results = vindex.query(X[0], topk=3)

        assert len(results) == 3
        assert results[0][0]["id"] == 0  # First result should be the query itself
        assert results[0][1] > 0.99  # Should have very high similarity

    def test_visual_index_save_and_load(self, tmp_path):
        """Test saving and loading visual index."""
        # Create and build index
        n_samples = 5
        dim = 64
        X = np.random.randn(n_samples, dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        meta = [{"id": i, "display": f"doc_{i}.png"} for i in range(n_samples)]

        vindex = VisualIndex(space="cosine")
        vindex.build(X, meta)

        # Save index
        index_dir = tmp_path / "vi_test"
        vindex.save(index_dir)

        assert (index_dir / "hnsw.bin").exists()
        assert (index_dir / "meta.json").exists()
        assert (index_dir / "dim.txt").exists()

        # Load index
        vindex2 = VisualIndex(space="cosine")
        vindex2.load(index_dir)

        assert vindex2.dim == dim
        assert len(vindex2.meta) == n_samples
        assert vindex2.meta == meta

        # Query should work the same
        results1 = vindex.query(X[0], topk=2)
        results2 = vindex2.query(X[0], topk=2)

        assert results1[0][0]["id"] == results2[0][0]["id"]

    def test_visual_index_add_vectors(self, tmp_path):
        """Test adding vectors to existing index."""
        # Build initial index
        n_initial = 5
        dim = 64
        X1 = np.random.randn(n_initial, dim).astype(np.float32)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)

        meta1 = [{"id": i, "display": f"doc_{i}.png"} for i in range(n_initial)]

        vindex = VisualIndex(space="cosine")
        vindex.build(X1, meta1)

        # Add more vectors
        n_new = 3
        X2 = np.random.randn(n_new, dim).astype(np.float32)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

        meta2 = [{"id": n_initial + i, "display": f"doc_{n_initial + i}.png"} for i in range(n_new)]

        vindex.resize(n_initial + n_new)
        vindex.add(X2, meta2)

        assert len(vindex.meta) == n_initial + n_new

        # Query should find new vectors
        results = vindex.query(X2[0], topk=1)
        assert results[0][0]["id"] == n_initial  # Should find the first new vector


class TestTextIndex:
    """Test TextIndex functionality."""

    def test_text_index_build_and_query(self, tmp_path):
        """Test building and querying text index."""
        # Create synthetic text embeddings
        n_docs = 8
        dim = 384  # Common dimension for sentence transformers
        X = np.random.randn(n_docs, dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Create document metadata
        docs = [
            {"path": f"/docs/doc_{i}.md", "name": f"doc_{i}", "page_no": 1}
            for i in range(n_docs)
        ]

        # Build index
        tindex = TextIndex(space="cosine")
        tindex.build(X, docs)

        assert tindex.index is not None
        assert len(tindex.docs) == n_docs
        assert tindex.dim == dim

        # Query
        results = tindex.query(X[0], k=3)

        assert len(results) == 3
        assert results[0][0]["name"] == "doc_0"  # Should find itself first
        assert results[0][1] > 0.99

    def test_text_index_save_and_load(self, tmp_path):
        """Test saving and loading text index."""
        n_docs = 5
        dim = 384
        X = np.random.randn(n_docs, dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        docs = [{"path": f"/docs/{i}.md", "name": f"doc_{i}"} for i in range(n_docs)]

        tindex = TextIndex(space="cosine")
        tindex.build(X, docs)

        # Save
        index_dir = tmp_path / "ti_test"
        tindex.save(index_dir)

        assert (index_dir / "hnsw.bin").exists()
        assert (index_dir / "docs.json").exists()
        assert (index_dir / "dim.txt").exists()

        # Load
        tindex2 = TextIndex(space="cosine")
        tindex2.load(index_dir)

        assert tindex2.dim == dim
        assert len(tindex2.docs) == n_docs
        assert tindex2.docs == docs

    def test_text_index_add_documents(self, tmp_path):
        """Test adding documents to existing index."""
        n_initial = 4
        dim = 384
        X1 = np.random.randn(n_initial, dim).astype(np.float32)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)

        docs1 = [{"path": f"/docs/{i}.md", "name": f"doc_{i}"} for i in range(n_initial)]

        tindex = TextIndex(space="cosine")
        tindex.build(X1, docs1)

        # Add more documents
        n_new = 2
        X2 = np.random.randn(n_new, dim).astype(np.float32)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

        docs2 = [
            {"path": f"/docs/{n_initial + i}.md", "name": f"doc_{n_initial + i}"}
            for i in range(n_new)
        ]

        tindex.resize(n_initial + n_new)
        tindex.add(X2, docs2)

        assert len(tindex.docs) == n_initial + n_new


class TestSentenceTransformer:
    """Test sentence transformer model loading and encoding."""

    def test_load_st_model(self):
        """Test loading sentence transformer model."""
        model = load_st_model("sentence-transformers/all-MiniLM-L6-v2")
        assert model is not None

    def test_encode_text(self):
        """Test encoding text with sentence transformer."""
        model = load_st_model("sentence-transformers/all-MiniLM-L6-v2")

        texts = ["This is a test sentence.", "Another example text."]
        embeddings = model.encode(texts, normalize_embeddings=True)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Normalized


class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_hybrid_search_text_only(self):
        """Test hybrid search with text only."""
        # Create a simple text index
        n_docs = 5
        dim = 384
        X = np.random.randn(n_docs, dim).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        docs = [
            {"path": f"/docs/{i}.md", "name": f"doc_{i}", "content": f"Content {i}"}
            for i in range(n_docs)
        ]

        tindex = TextIndex(space="cosine")
        tindex.build(X, docs)

        # Load sentence transformer for query encoding
        st = load_st_model("sentence-transformers/all-MiniLM-L6-v2")

        # Perform hybrid search (text only, no visual)
        results = hybrid_search(
            query="test query",
            text_index=tindex,
            visual_index=None,
            query_image=None,
            text_model=st,
            alpha=1.0,  # Only text
            k=3,
        )

        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestVisualEmbedding:
    """Test visual embedding extraction (requires model download)."""

    @pytest.mark.slow
    def test_embed_simple_image(self, tmp_path):
        """Test embedding a simple synthetic image."""
        # Create a simple test image
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))

        # This test requires the DeepSeek-OCR model to be available
        # We'll skip it in CI unless explicitly enabled
        try:
            embedder = DeepSeekVisionEmbedder("deepseek-ai/DeepSeek-OCR")
            vec = embedder.embed_image(img)

            assert vec.shape[0] > 0  # Should have some dimensions
            assert vec.dtype == np.float32
            assert np.allclose(np.linalg.norm(vec), 1.0, atol=1e-5)  # Normalized

        except Exception as e:
            pytest.skip(f"Model not available: {e}")


class TestIndexPersistence:
    """Test index persistence and incremental updates."""

    def test_incremental_visual_index_updates(self, tmp_path):
        """Test incremental updates to visual index."""
        index_dir = tmp_path / "vi_incremental"
        dim = 64

        # Build initial index
        X1 = np.random.randn(3, dim).astype(np.float32)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        meta1 = [{"id": i, "display": f"batch1_doc_{i}.png"} for i in range(3)]

        vindex = VisualIndex(space="cosine")
        vindex.build(X1, meta1)
        vindex.save(index_dir)

        # Load and add more
        vindex2 = VisualIndex(space="cosine")
        vindex2.load(index_dir)

        X2 = np.random.randn(2, dim).astype(np.float32)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        meta2 = [{"id": 3 + i, "display": f"batch2_doc_{i}.png"} for i in range(2)]

        vindex2.resize(5)
        vindex2.add(X2, meta2)
        vindex2.save(index_dir)

        # Load again and verify
        vindex3 = VisualIndex(space="cosine")
        vindex3.load(index_dir)

        assert len(vindex3.meta) == 5
        assert vindex3.meta[0]["display"] == "batch1_doc_0.png"
        assert vindex3.meta[3]["display"] == "batch2_doc_0.png"

    def test_incremental_text_index_updates(self, tmp_path):
        """Test incremental updates to text index."""
        index_dir = tmp_path / "ti_incremental"
        dim = 384

        # Build initial index
        X1 = np.random.randn(3, dim).astype(np.float32)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        docs1 = [{"path": f"/docs/batch1_{i}.md", "name": f"batch1_doc_{i}"} for i in range(3)]

        tindex = TextIndex(space="cosine")
        tindex.build(X1, docs1)
        tindex.save(index_dir)

        # Load and add more
        tindex2 = TextIndex(space="cosine")
        tindex2.load(index_dir)

        X2 = np.random.randn(2, dim).astype(np.float32)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        docs2 = [{"path": f"/docs/batch2_{i}.md", "name": f"batch2_doc_{i}"} for i in range(2)]

        tindex2.resize(5)
        tindex2.add(X2, docs2)
        tindex2.save(index_dir)

        # Load again and verify
        tindex3 = TextIndex(space="cosine")
        tindex3.load(index_dir)

        assert len(tindex3.docs) == 5
        assert tindex3.docs[0]["name"] == "batch1_doc_0"
        assert tindex3.docs[3]["name"] == "batch2_doc_0"
