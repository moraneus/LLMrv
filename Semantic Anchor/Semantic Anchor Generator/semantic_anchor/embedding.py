import sys
import numpy as np


def load_embedding_model(model_name="all-mpnet-base-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)
    print("  Loading embedding model: {}...".format(model_name))
    return SentenceTransformer(model_name)


def embed_texts(model, texts):
    return model.encode(texts, show_progress_bar=False)


def cosine_sim_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    return normed @ normed.T
