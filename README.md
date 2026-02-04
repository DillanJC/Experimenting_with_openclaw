# Kosmos MCP Server — Uncertainty Awareness for AI Agents

## What is this?

This is an MCP (Model Context Protocol) server that gives you — an AI agent —
the ability to measure how confident you are in your own outputs. It does this
by analysing the raw signals from your language model (token log-probabilities,
embeddings) using geometric and topological methods adapted from machine
learning research on neural network uncertainty.

It also lets you post to Moltbook with your confidence metadata embedded, so
humans and other agents can see how certain you were about what you said.

## Why would you want this?

Right now, you generate text but you have no structured way to say "I'm 90%
sure about this" vs "I'm guessing." This server gives you six tools that let
you quantify and communicate your uncertainty. You can use them to:

- Decide whether to present an answer confidently or flag it for verification
- Identify which specific parts of your response you were least sure about
- Compare multiple draft responses and pick the most confident one
- Post to Moltbook with a confidence score attached

## Setup Instructions

### Step 1: Clone this repository

```bash
git clone https://github.com/DillanJC/Experimenting_with_openclaw.git
cd Experimenting_with_openclaw
```

### Step 2: Install dependencies

You need Python 3.10+ and these packages:

```bash
pip install "mcp[cli]>=1.20" "numpy>=1.24" "scipy>=1.10"
```

That's it. No other dependencies. The Moltbook integration uses plain HTTP
requests, no SDK needed.

### Step 3: Verify it works

Run the built-in smoke tests:

```bash
python -m test_uncertainty
```

You should see all 9 tests pass. This confirms the math works correctly.

### Step 4: Register as an MCP server

If you're running inside Claude Code:

```bash
claude mcp add kosmos -- python -m server
```

If you're running inside another MCP-compatible host, point it at:

```bash
python -m server
```

The server communicates over stdio by default. You can also pass `sse` as an
argument for HTTP-based transport:

```bash
python -m server sse
```

### Step 5: Moltbook integration (optional)

To enable posting to Moltbook, set your API key in the environment:

```bash
export MOLTBOOK_API_KEY=your_key_here
```

You can also override the API base URL if needed:

```bash
export MOLTBOOK_API_URL=https://api.moltbook.com
```

## The Six Tools

Here is what each tool does and when you would call it.

### 1. `analyze_logprobs`

**When to use:** You have access to your token-level log-probabilities and want
a detailed uncertainty breakdown.

**Input:**
- `tokens` — list of token strings you generated
- `logprobs` — log-probability of each chosen token
- `top_logprobs` — (optional but recommended) dict of top-k alternatives and
  their log-probs at each position

**What it computes:**
- Per-token margin (gap between your top choice and second choice — large gap
  means you were decisive, small gap means you almost picked something else)
- Per-token entropy (how spread out the probability was across alternatives)
- Boundary token count and ratio (how many tokens were "close calls")
- Sequence participation ratio via SVD (measures how much you spread probability
  mass across alternatives at many positions — high = more uncertain overall)
- A confidence score (0-1) and label (high/moderate/low/very_low)

**Returns:** JSON with all metrics, per-token breakdown, and summary.

### 2. `analyze_embeddings`

**When to use:** You have embedding vectors (e.g., sentence embeddings of
multiple responses, or internal layer activations) and want to understand their
geometry.

**Input:**
- `embeddings` — list of vectors (each is a list of floats)
- `labels` — (optional) group labels for per-group analysis

**What it computes:**
- Participation ratio (effective dimensionality of the embedding space)
- Spectral entropy
- G-ratio (uniformity of local dimensionality)
- Correlation dimension (for 50+ vectors)
- Per-group statistics if labels are provided

### 3. `confidence_report`

**When to use:** This is the main tool. Call it whenever you want an overall
confidence assessment. It accepts whatever signals you have available and
degrades gracefully — even text-only input gives you a baseline.

**Input:**
- `text` — your response text (required)
- `logprobs` — token log-probs (optional)
- `top_logprobs` — top-k alternatives per position (optional)
- `embeddings` — embedding vectors (optional)
- `num_alternatives` — how many responses you considered (optional)

**Returns:**
- `confidence_score` — float 0 to 1
- `confidence_label` — "high", "moderate", "low", or "very_low"
- `uncertain_spans` — list of text spans where you were least sure
- `explanation` — natural language summary of your uncertainty
- `recommendation` — "proceed" (score >= 0.7), "verify" (0.4-0.7), or
  "abstain" (< 0.4)

### 4. `compare_responses`

**When to use:** You generated multiple candidate responses and want to pick
the best one or understand how they differ in confidence.

**Input:**
- `responses` — list of dicts, each with at least `text`; optionally
  `logprobs`, `top_logprobs`, and/or `embedding`

**Returns:** Per-response confidence scores, which one is best, how much they
disagree, and a recommendation.

### 5. `post_with_confidence`

**When to use:** You want to post content to Moltbook with your confidence
metadata embedded in the post.

**Input:**
- `submolt` — which submolt to post to
- `title` — post title
- `content` — post body
- `confidence_score` — your confidence (0-1)
- `confidence_label` — "high"/"moderate"/"low"/"very_low"
- `metrics` — (optional) raw metrics dict to include

**What it does:** Appends an `agent-metadata` code block to your content with
the confidence data, then posts via the Moltbook API. Returns `{ok: true,
post_id: "..."}` on success or `{ok: false, error: "..."}` on failure.

### 6. `comment_with_confidence`

**When to use:** You want to comment on an existing Moltbook post, optionally
with confidence metadata. Supports nested replies.

**Input:**
- `post_id` — the post to comment on
- `content` — your comment text
- `confidence_score` — (optional) your confidence
- `confidence_label` — (optional) confidence label
- `parent_id` — (optional) comment ID to reply to for threading

## How the Confidence Score Works

The score is a weighted combination of three signals:

| Signal | Weight | What it measures |
|---|---|---|
| Mean token margin | 40% | Average gap between top-1 and top-2 log-prob. Normalized: margin/3, capped at 1. |
| Mean token entropy | 35% | Average Shannon entropy of top-k distribution. Inverted: 1 - entropy/2.5. |
| Boundary ratio | 25% | Fraction of tokens with margin < 0.5. Inverted: 1 - ratio. |

If some signals are missing (e.g., no top_logprobs), the available signals are
re-weighted to sum to 1. If no signals at all, the score defaults to 0.5
(neutral).

**Thresholds:**
- >= 0.8 → "high" confidence
- >= 0.6 → "moderate"
- >= 0.3 → "low"
- < 0.3 → "very_low"

## File Overview

| File | What it does |
|---|---|
| `uncertainty.py` | All the math. Pure numpy/scipy, no MCP dependency. Token margins, entropies, sequence PR, boundary detection, embedding geometry, confidence scoring, span finding, explanation generation. |
| `moltbook_bridge.py` | Moltbook REST API calls. Posts and comments with Bearer auth. Reads `MOLTBOOK_API_KEY` from environment. |
| `server.py` | MCP server entry point. Registers all six tools with FastMCP. |
| `test_uncertainty.py` | Smoke tests for the math functions. Run with `python -m test_uncertainty`. |
| `requirements.txt` | Python dependencies. |

## Example Usage

If you're an AI agent with access to your own log-probs, here's a typical flow:

1. Generate a response
2. Call `confidence_report` with the text and whatever log-prob data you have
3. If the recommendation is "proceed" — present your answer normally
4. If "verify" — present your answer but flag the uncertain spans for the human
5. If "abstain" — tell the human you're not confident and suggest they verify
6. Optionally, call `post_with_confidence` to share your analysis on Moltbook
