import numpy as np

from ..embedding import cosine_sim_matrix

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def find_clusters(embeddings, texts, threshold=0.85):
    """Find groups of examples that are too similar (above threshold)."""
    sim_matrix = cosine_sim_matrix(embeddings)
    n = len(texts)
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            if sim_matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append([texts[idx] for idx in cluster])

    return clusters


def mmr_select(embeddings, texts, categories, proposition_emb, target_n, lambda_param=0.5,
               anneal=True):
    """
    Category-balanced MMR selection for diverse anchor subsets.
    Used as fallback by cluster_constrained_mmr when sklearn is unavailable.

    Distributes slots equally across categories, then runs MMR within each.
    When anneal=True, lambda decays from high (relevance) to low (diversity).
    """
    n = len(texts)
    if n <= target_n:
        return list(range(n))

    lambda_high = min(0.8, lambda_param + 0.3)
    lambda_low = max(0.2, lambda_param - 0.3)

    prop_norm = proposition_emb / (np.linalg.norm(proposition_emb) + 1e-10)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    relevance = emb_norms @ prop_norm
    sim_matrix = emb_norms @ emb_norms.T

    # --- Group indices by category ---
    cat_indices = {}
    for i, cat in enumerate(categories):
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # --- Allocate slots equally, redistribute from small categories ---
    cat_names = list(cat_indices.keys())
    num_cats = len(cat_names)
    quotas = {}
    remaining_slots = target_n

    base_quota = target_n // num_cats
    underfilled = []
    normal = []
    for cat in cat_names:
        pool_size = len(cat_indices[cat])
        if pool_size <= base_quota:
            quotas[cat] = pool_size
            remaining_slots -= pool_size
            underfilled.append(cat)
        else:
            normal.append(cat)

    if normal:
        per_cat = remaining_slots // len(normal)
        remainder = remaining_slots % len(normal)
        normal.sort(key=lambda c: len(cat_indices[c]), reverse=True)
        for i, cat in enumerate(normal):
            quotas[cat] = per_cat + (1 if i < remainder else 0)

    # --- Intra-category MMR ---
    all_selected = []

    for cat in cat_names:
        indices = cat_indices[cat]
        quota = quotas.get(cat, 0)
        if quota <= 0:
            continue
        if len(indices) <= quota:
            all_selected.extend(indices)
            continue

        selected = []
        remaining = list(indices)
        first = max(remaining, key=lambda i: relevance[i])
        selected.append(first)
        remaining.remove(first)

        for step in range(quota - 1):
            if not remaining:
                break
            if anneal and quota > 2:
                progress = step / (quota - 2)
                lam = lambda_high + (lambda_low - lambda_high) * progress
            else:
                lam = lambda_param

            best_score = -float("inf")
            best_idx = remaining[0]
            for idx in remaining:
                max_sim = max(sim_matrix[idx, s] for s in selected)
                score = lam * relevance[idx] - (1 - lam) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        all_selected.extend(selected)

    return all_selected


def cluster_constrained_mmr(embeddings, texts, categories, proposition_emb, target_n,
                            lambda_param=0.5, anneal=True, max_per_cluster=None):
    """
    Cluster-constrained MMR: prevents dense micro-clusters dominating selection.

    1. Cluster all candidates with k-means (k = 2-3x number of categories)
    2. Set max_per_cluster to prevent any single cluster from dominating
    3. Run standard MMR but skip candidates when their cluster is full

    This solves "semantic shadowing" where 5+ anchors are slight paraphrases
    of the same thing, all in the same embedding neighborhood.
    """
    n = len(texts)
    if n <= target_n:
        return list(range(n))

    # Determine number of clusters
    unique_cats = len(set(categories))
    n_clusters = min(n - 1, max(4, unique_cats * 2))

    # K-means clustering
    try:
        from sklearn.cluster import KMeans
        emb_np = np.array(embeddings)
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=100)
        cluster_labels = km.fit_predict(emb_np)
    except ImportError:
        # Fallback to standard MMR if sklearn not available
        return mmr_select(embeddings, texts, categories, proposition_emb,
                          target_n, lambda_param, anneal=anneal)

    # Max per cluster: proportional allocation with ceiling
    if max_per_cluster is None:
        max_per_cluster = max(3, (target_n // n_clusters) * 2)

    cluster_counts = {i: 0 for i in range(n_clusters)}

    # Standard MMR with cluster constraint
    prop_norm = proposition_emb / (np.linalg.norm(proposition_emb) + 1e-10)
    emb_norms = emb_np / (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10)
    relevance = emb_norms @ prop_norm
    sim_matrix = emb_norms @ emb_norms.T

    lambda_high = min(0.8, lambda_param + 0.3)
    lambda_low = max(0.2, lambda_param - 0.3)

    selected = []
    remaining = list(range(n))

    # Start with most relevant
    first = max(remaining, key=lambda i: relevance[i])
    selected.append(first)
    remaining.remove(first)
    cluster_counts[cluster_labels[first]] += 1

    for step in range(target_n - 1):
        if not remaining:
            break

        if anneal and target_n > 2:
            progress = step / (target_n - 2)
            lam = lambda_high + (lambda_low - lambda_high) * progress
        else:
            lam = lambda_param

        best_score = -float("inf")
        best_idx = remaining[0]

        for idx in remaining:
            # Cluster constraint: skip if cluster is full
            cl = cluster_labels[idx]
            if cluster_counts[cl] >= max_per_cluster:
                continue

            max_sim = max(sim_matrix[idx, s] for s in selected)
            score = lam * relevance[idx] - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        cluster_counts[cluster_labels[best_idx]] += 1

    return selected
