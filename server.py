"""
Kosmos MCP Server — Uncertainty awareness for AI agents.

Exposes six tools via the Model Context Protocol:
  1. analyze_logprobs   – token-level uncertainty from log-probabilities
  2. analyze_embeddings – geometric analysis of embedding vectors
  3. confidence_report  – high-level confidence assessment
  4. compare_responses  – compare uncertainty across candidate responses
  5. post_with_confidence – post to Moltbook with confidence metadata
  6. comment_with_confidence – comment on a Moltbook post with confidence metadata
"""

from mcp.server.fastmcp import FastMCP

from .uncertainty import (
    classify_confidence,
    compute_boundary_ratio,
    compute_confidence_score,
    compute_embedding_pr,
    compute_sequence_pr,
    compute_token_entropies,
    compute_token_margins,
    find_uncertain_spans,
    generate_explanation,
)
from .moltbook_bridge import comment_on_moltbook, post_to_moltbook

import numpy as np

mcp = FastMCP("kosmos", instructions="Uncertainty awareness for AI agents")


# ── Tool 1 ────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_logprobs(
    tokens: list[str],
    logprobs: list[float],
    top_logprobs: list[dict] | None = None,
) -> dict:
    """Analyse token-level log-probabilities for uncertainty signals.

    Returns per-token margins/entropies, sequence-level statistics,
    boundary token count, participation ratio, and a confidence label.
    """
    result: dict = {}

    # Per-token breakdown
    per_token: list[dict] = []
    margins = None
    entropies = None
    boundary_ratio = None

    if top_logprobs is not None and len(top_logprobs) == len(tokens):
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)
        seq_pr = compute_sequence_pr(top_logprobs)

        finite_margins = margins[np.isfinite(margins)]

        for i, tok in enumerate(tokens):
            entry = {"token": tok, "logprob": logprobs[i]}
            if np.isfinite(margins[i]):
                entry["margin"] = round(float(margins[i]), 4)
            entry["entropy"] = round(float(entropies[i]), 4)
            per_token.append(entry)

        result["mean_margin"] = round(float(np.mean(finite_margins)), 4) if len(finite_margins) else None
        result["min_margin"] = round(float(np.min(finite_margins)), 4) if len(finite_margins) else None
        result["mean_entropy"] = round(float(np.mean(entropies)), 4)
        result["boundary_token_count"] = int(np.sum(finite_margins < 0.5)) if len(finite_margins) else 0
        result["boundary_ratio"] = round(boundary_ratio, 4)
        result["sequence_pr"] = round(seq_pr, 4)
    else:
        for i, tok in enumerate(tokens):
            per_token.append({"token": tok, "logprob": logprobs[i]})

    result["per_token"] = per_token

    # Confidence
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    result["confidence_score"] = score
    result["confidence_label"] = label

    return result


# ── Tool 2 ────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_embeddings(
    embeddings: list[list[float]],
    labels: list[str] | None = None,
) -> dict:
    """Geometric analysis of embedding vectors.

    Computes participation ratio, spectral entropy, G-ratio, effective
    dimensionality, and (for N > 50) correlation dimension.
    If group labels are provided, per-group statistics are included.
    """
    emb = np.array(embeddings, dtype=np.float64)
    result = compute_embedding_pr(emb)

    if labels is not None and len(set(labels)) > 1:
        groups: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)

        group_stats: dict = {}
        group_prs: list[float] = []
        for lab, idxs in groups.items():
            if len(idxs) >= 3:
                g_emb = emb[idxs]
                g_res = compute_embedding_pr(g_emb, k=min(20, len(idxs) - 1))
                group_stats[lab] = g_res
                group_prs.append(g_res["pr_mean"])

        result["group_stats"] = group_stats
        if len(group_prs) >= 2:
            result["cross_group_g_ratio"] = round(
                min(group_prs) / (np.mean(group_prs) + 1e-15), 4
            )

    return result


# ── Tool 3 ────────────────────────────────────────────────────────────────

@mcp.tool()
def confidence_report(
    text: str,
    logprobs: list[float] | None = None,
    top_logprobs: list[dict] | None = None,
    embeddings: list[list[float]] | None = None,
    num_alternatives: int | None = None,
) -> dict:
    """High-level confidence assessment from any available signals.

    Accepts whatever data is available and degrades gracefully.
    Returns a confidence score, label, uncertain spans, explanation,
    raw metrics, and a recommendation (proceed / verify / abstain).
    """
    metrics: dict = {}
    margins = None
    entropies = None
    boundary_ratio = None
    tokens = text.split()  # rough tokenization for span detection

    if top_logprobs is not None:
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)
        seq_pr = compute_sequence_pr(top_logprobs)
        finite = margins[np.isfinite(margins)]
        metrics["mean_margin"] = round(float(np.mean(finite)), 4) if len(finite) else None
        metrics["mean_entropy"] = round(float(np.mean(entropies)), 4)
        metrics["boundary_ratio"] = round(boundary_ratio, 4)
        metrics["sequence_pr"] = round(seq_pr, 4)

    if embeddings is not None:
        emb = np.array(embeddings, dtype=np.float64)
        metrics["embedding_geometry"] = compute_embedding_pr(emb)

    if num_alternatives is not None:
        metrics["num_alternatives"] = num_alternatives

    # Score
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    metrics["confidence_score"] = score
    metrics["confidence_label"] = label

    # Uncertain spans
    uncertain_spans: list[dict] = []
    if margins is not None and len(tokens) == len(margins):
        uncertain_spans = find_uncertain_spans(tokens, margins)
    metrics["uncertain_span_count"] = len(uncertain_spans)

    # Recommendation
    if score >= 0.7:
        recommendation = "proceed"
    elif score >= 0.4:
        recommendation = "verify"
    else:
        recommendation = "abstain"

    explanation = generate_explanation(metrics)

    return {
        "confidence_score": score,
        "confidence_label": label,
        "uncertain_spans": uncertain_spans,
        "explanation": explanation,
        "metrics": metrics,
        "recommendation": recommendation,
    }


# ── Tool 4 ────────────────────────────────────────────────────────────────

@mcp.tool()
def compare_responses(responses: list[dict]) -> dict:
    """Compare uncertainty across multiple candidate responses.

    Each entry should have at least ``text``; optionally ``logprobs``,
    ``top_logprobs``, and/or ``embedding`` (a single vector).
    """
    per_response: list[dict] = []

    for i, resp in enumerate(responses):
        entry: dict = {"index": i, "text_length": len(resp.get("text", ""))}
        margins = None
        entropies = None
        br = None

        if resp.get("top_logprobs"):
            margins = compute_token_margins(resp["top_logprobs"])
            entropies = compute_token_entropies(resp["top_logprobs"])
            br = compute_boundary_ratio(margins)

        score = compute_confidence_score(margins, entropies, br)
        entry["confidence_score"] = score
        entry["confidence_label"] = classify_confidence(score)
        per_response.append(entry)

    # Embedding-level comparison
    emb_analysis = None
    emb_list = [r["embedding"] for r in responses if "embedding" in r]
    if len(emb_list) >= 2:
        emb_matrix = np.array(emb_list, dtype=np.float64)
        emb_analysis = compute_embedding_pr(emb_matrix, k=min(20, len(emb_list) - 1))

    scores = [e["confidence_score"] for e in per_response]
    best_idx = int(np.argmax(scores))
    spread = round(float(max(scores) - min(scores)), 4) if scores else 0.0

    return {
        "per_response": per_response,
        "best_index": best_idx,
        "score_spread": spread,
        "embedding_analysis": emb_analysis,
        "recommendation": (
            f"Response {best_idx} has the highest confidence "
            f"({scores[best_idx]:.2f}). "
            + (
                "Responses largely agree in confidence."
                if spread < 0.15
                else "Significant confidence spread — consider verifying."
            )
        ),
    }


# ── Tool 5 ────────────────────────────────────────────────────────────────

@mcp.tool()
def post_with_confidence(
    submolt: str,
    title: str,
    content: str,
    confidence_score: float,
    confidence_label: str,
    metrics: dict | None = None,
) -> dict:
    """Post to Moltbook with embedded confidence metadata.

    Appends an agent-metadata block to the content and posts via the
    Moltbook SDK. Fails gracefully if no MOLTBOOK_API_KEY is configured.
    """
    return post_to_moltbook(
        submolt=submolt,
        title=title,
        content=content,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        metrics=metrics,
    )


# ── Tool 6 ────────────────────────────────────────────────────────────────

@mcp.tool()
def comment_with_confidence(
    post_id: str,
    content: str,
    confidence_score: float | None = None,
    confidence_label: str | None = None,
    parent_id: str | None = None,
) -> dict:
    """Comment on a Moltbook post, optionally with confidence metadata.

    Supports nested replies via *parent_id*. If confidence_score and
    confidence_label are provided, an agent-metadata block is appended.
    Fails gracefully if no MOLTBOOK_API_KEY is configured.
    """
    return comment_on_moltbook(
        post_id=post_id,
        content=content,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        parent_id=parent_id,
    )


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
