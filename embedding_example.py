"""
Embedding pipeline walkthrough
================================
Step 1 — Load a model that maps text → dense vector
Step 2 — Encode sentences into embeddings
Step 3 — Inspect what a vector looks like
Step 4 — Rank sentences by similarity to a query
Step 5 — Visualize the embedding space in 2D
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text


# ---------------------------------------------------------------------------
# Step 1: Model
# ---------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    """
    Load a bi-encoder model. It was trained with contrastive loss so that
    semantically similar sentences end up close in vector space.
    all-MiniLM-L6-v2 is 22M params, 384-dim output — fast and good enough
    for most retrieval tasks.
    """
    #name = "all-MiniLM-L6-v2"                        # English only

    name = "paraphrase-multilingual-MiniLM-L12-v2"   # 50+ languages
    
    return SentenceTransformer(name)


# ---------------------------------------------------------------------------
# Step 2: Encoding
# ---------------------------------------------------------------------------

def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a (N, D) matrix of float32 vectors.
    Each row is one sentence; D=384 for MiniLM.
    Vectors are L2-normalised by default, so cosine similarity == dot product.
    """
    return model.encode(texts, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Step 3: Inspect
# ---------------------------------------------------------------------------

def inspect_vector(vec: np.ndarray, label: str = "vector") -> None:
    """
    Print the shape, range, and first few values of an embedding.
    No single dimension has a human-readable meaning — the geometry across
    all 384 dimensions together encodes the semantics.
    """
    print(f"\n--- {label} ---")
    print(f"  shape : {vec.shape}   dtype: {vec.dtype}")
    print(f"  range : [{vec.min():.4f}, {vec.max():.4f}]   norm: {np.linalg.norm(vec):.4f}")
    print(f"  first 16 values:\n  {vec[:16]}")


# ---------------------------------------------------------------------------
# Step 4: Similarity search
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already unit-norm, so this is just a dot product.
    return float(np.dot(a, b))


def rank_by_similarity(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    corpus_texts: list[str],
) -> list[tuple[float, str]]:
    """
    Score each corpus sentence against the query and return them sorted
    highest-first. Score range: -1 (opposite) to +1 (identical meaning).
    """
    scores = [
        (cosine_similarity(query_vec, vec), text)
        for vec, text in zip(corpus_vecs, corpus_texts)
    ]
    return sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Step 5: Visualisation
# ---------------------------------------------------------------------------

def plot_embeddings_2d(
    corpus_vecs: np.ndarray,
    corpus_texts: list[str],
    query_vec: np.ndarray,
    query_text: str,
    output_path: str = "embeddings_2d.png",
) -> None:
    """
    Project all vectors to 2D with PCA and scatter-plot them.
    PCA finds the two directions of maximum variance — nearby points are
    semantically similar (though some distance info is lost in compression).
    """
    all_vecs = np.vstack([corpus_vecs, query_vec])
    all_labels = corpus_texts + [f"QUERY: {query_text}"]
    colors = ["steelblue"] * len(corpus_texts) + ["crimson"]

    coords = PCA(n_components=2).fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(11, 7))
    xs, ys = coords[:, 0], coords[:, 1]
    ax.scatter(xs[:-1], ys[:-1], color="steelblue", s=90, zorder=3)
    ax.scatter(xs[-1], ys[-1], color="crimson", s=120, zorder=4, marker="*")

    texts = [
        ax.text(x, y, label, fontsize=8,
                color="crimson" if i == len(all_labels) - 1 else "black",
                fontweight="bold" if i == len(all_labels) - 1 else "normal")
        for i, (x, y, label) in enumerate(zip(xs, ys, all_labels))
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_title("Sentence embeddings projected to 2D (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus = [
        # Animals
        "The cat sat on the mat.",
        "A kitten rested on a rug.",
        "The dog chased the ball across the yard.",
        "A puppy napped under the warm blanket.",
        # Technology / ML
        "Machine learning transforms raw data into predictions.",
        "Neural networks learn by adjusting millions of weights.",
        "Gradient descent minimises the loss function iteratively.",
        # Finance
        "The stock market closed higher today.",
        "Investors bought bonds as interest rates fell.",
        "Inflation eroded the purchasing power of consumers.",
        # Food
        "She baked a sourdough loaf with a crispy crust.",
        "The pasta was tossed in a rich tomato sauce.",
        # Cross-lingual: same topics in other languages
        "Le chaton s'est reposé sur un tapis.",           # FR: A kitten rested on a rug
        "Das Modell lernt durch Gradientenabstieg.",      # DE: The model learns via gradient descent
        "Los inversores compraron bonos hoy.",            # ES: Investors bought bonds today
    ]
    query = "A small feline lounged on a carpet."

    # 1 + 2: load and encode
    model = load_model()
    corpus_vecs = encode(model, corpus)
    query_vec = encode(model, [query])[0]

    # 3: what does a vector look like?
    inspect_vector(query_vec, label=query)

    # 4: ranked similarity
    print(f"\n--- Similarity to query ---")
    print(f"  {'Score':>6}  Sentence")
    print("  " + "-" * 56)
    for score, text in rank_by_similarity(query_vec, corpus_vecs, corpus):
        print(f"  {score:.4f}  {text}")

    # 5: 2D plot
    plot_embeddings_2d(corpus_vecs, corpus, query_vec, query)
