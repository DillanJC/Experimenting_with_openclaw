"""
Core uncertainty computation for LLM outputs.

Adapts geometric/topological methods from PR-max research to token-level
log-probabilities and embedding vectors. Pure numpy/scipy — no torch dependency.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------

def compute_token_margins(top_logprobs: list[dict]) -> np.ndarray:
    """Compute per-token margin (top1 - top2 log-prob).

    Args:
        top_logprobs: List of dicts mapping token -> log-prob for each position.

    Returns:
        Array of margins (>= 0). Positions with fewer than 2 alternatives
        get margin = inf (maximally confident).
    """
    margins = np.full(len(top_logprobs), np.inf)
    for i, lp in enumerate(top_logprobs):
        if lp is None or len(lp) < 2:
            continue
        sorted_vals = sorted(lp.values(), reverse=True)
        # log-probs are negative; top1 is least negative, so margin >= 0
        margins[i] = sorted_vals[0] - sorted_vals[1]
    return margins


def compute_token_entropies(top_logprobs: list[dict]) -> np.ndarray:
    """Compute Shannon entropy at each token position from top-k probs.

    The top-k log-probs are converted to probabilities and renormalized
    so they sum to 1 before computing entropy.
    """
    entropies = np.zeros(len(top_logprobs))
    for i, lp in enumerate(top_logprobs):
        if lp is None or len(lp) == 0:
            continue
        logp = np.array(list(lp.values()), dtype=np.float64)
        probs = np.exp(logp)
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total  # renormalize
        probs = probs[probs > 1e-12]
        entropies[i] = -np.sum(probs * np.log(probs))
    return entropies


def compute_sequence_pr(top_logprobs: list[dict]) -> float:
    """Participation ratio of the token x top-k log-prob matrix.

    Stacks top-k log-probs at each position into an (n_tokens, k) matrix
    and computes the participation ratio via SVD. High PR means the model
    spreads probability mass across many alternatives at many positions
    (= more uncertain). Low PR means a few dominant singular directions
    (= more certain).
    """
    if not top_logprobs:
        return 1.0

    # Determine the common k (max number of alternatives seen)
    k = max((len(lp) for lp in top_logprobs if lp), default=0)
    if k == 0:
        return 1.0

    n = len(top_logprobs)
    mat = np.full((n, k), -30.0)  # fill with very low log-prob
    for i, lp in enumerate(top_logprobs):
        if lp is None:
            continue
        vals = sorted(lp.values(), reverse=True)
        for j, v in enumerate(vals[:k]):
            mat[i, j] = v

    # Center columns
    mat = mat - mat.mean(axis=0, keepdims=True)

    try:
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        S_sq = S ** 2
        denom = np.sum(S_sq ** 2)
        if denom < 1e-15:
            return 1.0
        pr = (S_sq.sum() ** 2) / denom
        return float(pr)
    except np.linalg.LinAlgError:
        return 1.0


def compute_boundary_ratio(margins: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of tokens whose margin is below *threshold*."""
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        return 0.0
    return float(np.mean(finite < threshold))


# ---------------------------------------------------------------------------
# Embedding-level metrics
# ---------------------------------------------------------------------------

def compute_embedding_pr(embeddings: np.ndarray, k: int = 20) -> dict:
    """Geometric analysis of embedding vectors (numpy-only).

    Computes participation ratio, spectral entropy, effective dimensionality,
    and optionally correlation dimension.

    Args:
        embeddings: (N, D) array.
        k: Neighbors for local SVD (capped at N-1).

    Returns:
        Dict with pr_mean, pr_min, se_mean, g_ratio, effective_dim,
        and correlation_dim (if N > 50).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    N, D = embeddings.shape

    if N < 3:
        return {
            "pr_mean": 1.0, "pr_min": 1.0, "se_mean": 0.0,
            "g_ratio": 1.0, "effective_dim": 1.0,
        }

    k_use = min(k, N - 1, D)

    # Pairwise distances for neighbor lookup
    dists = squareform(pdist(embeddings))

    pr_vals = np.zeros(N)
    se_vals = np.zeros(N)

    for i in range(N):
        idx = np.argsort(dists[i])[1:k_use + 1]  # exclude self
        neighbors = embeddings[idx]
        centered = neighbors - neighbors.mean(axis=0)
        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            S_sq = S ** 2
            denom = np.sum(S_sq ** 2)
            pr_vals[i] = (S_sq.sum() ** 2) / (denom + 1e-15)
            p = S_sq / (S_sq.sum() + 1e-15)
            p = p[p > 1e-12]
            se_vals[i] = -np.sum(p * np.log(p))
        except np.linalg.LinAlgError:
            pr_vals[i] = np.nan
            se_vals[i] = np.nan

    pr_valid = pr_vals[~np.isnan(pr_vals)]
    se_valid = se_vals[~np.isnan(se_vals)]
    pr_mean = float(np.mean(pr_valid)) if len(pr_valid) else 1.0
    pr_min = float(np.min(pr_valid)) if len(pr_valid) else 1.0
    se_mean = float(np.mean(se_valid)) if len(se_valid) else 0.0
    g_ratio = pr_min / pr_mean if pr_mean > 0 else 0.0

    # Global SVD for effective dimensionality
    centered_all = embeddings - embeddings.mean(axis=0)
    try:
        _, S_all, _ = np.linalg.svd(centered_all, full_matrices=False)
        S_sq_all = S_all ** 2
        eff_dim = float((S_sq_all.sum() ** 2) / (np.sum(S_sq_all ** 2) + 1e-15))
    except np.linalg.LinAlgError:
        eff_dim = 1.0

    result = {
        "pr_mean": round(pr_mean, 4),
        "pr_min": round(pr_min, 4),
        "se_mean": round(se_mean, 4),
        "g_ratio": round(g_ratio, 4),
        "effective_dim": round(eff_dim, 4),
    }

    # Correlation dimension estimate (Grassberger-Procaccia) for larger sets
    if N > 50:
        result["correlation_dim"] = _estimate_correlation_dim(dists)

    return result


def _estimate_correlation_dim(dist_matrix: np.ndarray) -> float:
    """Grassberger-Procaccia correlation dimension estimate."""
    N = dist_matrix.shape[0]
    # Upper triangle distances
    triu_idx = np.triu_indices(N, k=1)
    all_dists = dist_matrix[triu_idx]
    all_dists = all_dists[all_dists > 0]
    if len(all_dists) < 10:
        return 0.0

    r_min = np.percentile(all_dists, 5)
    r_max = np.percentile(all_dists, 50)
    if r_min <= 0 or r_max <= r_min:
        return 0.0

    radii = np.geomspace(r_min, r_max, 20)
    n_pairs = len(all_dists)
    counts = np.array([np.sum(all_dists < r) / n_pairs for r in radii])
    counts = counts[counts > 0]
    if len(counts) < 5:
        return 0.0

    log_r = np.log(radii[:len(counts)])
    log_c = np.log(counts)
    # Linear fit in log-log space
    coeffs = np.polyfit(log_r, log_c, 1)
    return round(float(coeffs[0]), 4)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence_score(
    margins: np.ndarray | None = None,
    entropies: np.ndarray | None = None,
    boundary_ratio: float | None = None,
) -> float:
    """Weighted combination of uncertainty signals, normalized to [0, 1].

    Higher = more confident.
    """
    components = []
    weights = []

    if margins is not None:
        finite = margins[np.isfinite(margins)]
        if len(finite) > 0:
            # Mean margin: higher margin = more confident.
            # Typical margins in [0, ~5]; sigmoid-like normalization.
            mean_m = float(np.mean(finite))
            margin_score = min(mean_m / 3.0, 1.0)
            components.append(margin_score)
            weights.append(0.4)

    if entropies is not None and len(entropies) > 0:
        mean_e = float(np.mean(entropies))
        # Low entropy = high confidence. Typical range [0, ~3].
        entropy_score = max(1.0 - mean_e / 2.5, 0.0)
        components.append(entropy_score)
        weights.append(0.35)

    if boundary_ratio is not None:
        # Low boundary ratio = high confidence.
        br_score = 1.0 - boundary_ratio
        components.append(br_score)
        weights.append(0.25)

    if not components:
        return 0.5  # no data → neutral

    weights = np.array(weights)
    weights /= weights.sum()
    score = float(np.dot(weights, components))
    return round(np.clip(score, 0.0, 1.0), 4)


def classify_confidence(score: float) -> str:
    """Map a 0-1 confidence score to a human-readable label."""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "moderate"
    if score >= 0.3:
        return "low"
    return "very_low"


def find_uncertain_spans(
    tokens: list[str],
    margins: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    """Identify contiguous spans of low-margin tokens.

    Returns a list of dicts with keys: start, end (token indices),
    char_start, char_end (character offsets into the joined token string),
    tokens, text, and min_margin.
    """
    # Pre-compute character offset of each token in the joined string
    char_offsets: list[int] = []
    pos = 0
    for tok in tokens:
        char_offsets.append(pos)
        pos += len(tok)

    spans: list[dict] = []
    in_span = False
    start = 0
    span_tokens: list[str] = []
    span_min = float("inf")

    for i, (tok, m) in enumerate(zip(tokens, margins)):
        low = np.isfinite(m) and m < threshold
        if low:
            if not in_span:
                start = i
                span_tokens = []
                span_min = float("inf")
                in_span = True
            span_tokens.append(tok)
            span_min = min(span_min, float(m))
        else:
            if in_span:
                text = "".join(span_tokens)
                margin_rounded = round(span_min, 4)
                spans.append({
                    "start": start,
                    "end": i,
                    "char_start": char_offsets[start],
                    "char_end": char_offsets[start] + len(text),
                    "tokens": span_tokens,
                    "text": text,
                    "min_margin": margin_rounded,
                    "display": _span_severity(span_min),
                })
                in_span = False
    # Close trailing span
    if in_span:
        text = "".join(span_tokens)
        margin_rounded = round(span_min, 4)
        spans.append({
            "start": start,
            "end": len(tokens),
            "char_start": char_offsets[start],
            "char_end": char_offsets[start] + len(text),
            "tokens": span_tokens,
            "text": text,
            "min_margin": margin_rounded,
            "display": _span_severity(span_min),
        })
    return spans


def _span_severity(min_margin: float) -> dict:
    """Map a span's minimum margin to a display hint for frontends."""
    if min_margin < 0.1:
        return {"severity": "critical", "color": "#e53e3e"}
    if min_margin < 0.25:
        return {"severity": "high", "color": "#dd6b20"}
    return {"severity": "moderate", "color": "#d69e2e"}


def compute_self_consistency(texts: list[str], n: int = 2) -> dict:
    """Measure agreement between multiple response texts using n-gram overlap.

    Returns pairwise Jaccard similarities and an aggregate agreement score.
    A high agreement score with high confidence = trustworthy.
    High confidence but low agreement = contradictory answers, be wary.
    """
    if len(texts) < 2:
        return {"agreement": 1.0, "pairwise": [], "n": n}

    def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < n:
            return {tuple(words)}
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    sets = [_ngrams(t, n) for t in texts]
    pairwise: list[dict] = []
    scores: list[float] = []

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            if union == 0:
                sim = 1.0
            else:
                sim = len(sets[i] & sets[j]) / union
            sim = round(sim, 4)
            pairwise.append({"i": i, "j": j, "similarity": sim})
            scores.append(sim)

    return {
        "agreement": round(float(np.mean(scores)), 4),
        "pairwise": pairwise,
        "n": n,
    }


def generate_explanation(metrics: dict) -> str:
    """Produce a natural-language explanation of the uncertainty metrics."""
    parts: list[str] = []
    label = metrics.get("confidence_label", "unknown")
    score = metrics.get("confidence_score", None)

    if score is not None:
        parts.append(f"Overall confidence is {label} (score {score:.2f}).")

    br = metrics.get("boundary_ratio")
    if br is not None:
        pct = br * 100
        if pct > 30:
            parts.append(
                f"{pct:.0f}% of tokens are near the decision boundary, "
                "indicating significant uncertainty in word choices."
            )
        elif pct > 10:
            parts.append(
                f"{pct:.0f}% of tokens are near the decision boundary."
            )

    seq_pr = metrics.get("sequence_pr")
    if seq_pr is not None and seq_pr > 3.0:
        parts.append(
            f"The sequence participation ratio is {seq_pr:.1f}, suggesting "
            "the model considered many alternatives across positions."
        )

    n_uncertain = metrics.get("uncertain_span_count", 0)
    if n_uncertain > 0:
        parts.append(
            f"There are {n_uncertain} uncertain span(s) in the output."
        )

    if not parts:
        parts.append("Insufficient data for a detailed explanation.")

    return " ".join(parts)
