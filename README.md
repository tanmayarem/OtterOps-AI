# AI Finance Controller

**Razorpay Builtathon — Track 04: AI Finance Controller**

A deterministic payment gateway settlement reconciliation engine that matches
internal payment records against bank settlement data across a 60-record batch,
reports match rate, produces an honest exception list, and explains each
exception using an LLM — without letting the LLM re-decide any match.

## Design philosophy

- **Deterministic matching over LLM matching.** The two-pass matcher (exact
  then fuzzy) is fully deterministic — same inputs always produce same outputs.
  A finance controller that gives different results on the same data is
  untrustworthy. We use no randomness, no temperature, no sampling in the
  matching path.

- **LLM explains, never decides.** The LLM is the ONLY AI in the pipeline,
  and its only job is to generate natural-language explanations for exceptions
  the matcher already classified. It never modifies `matched_pairs.csv`, never
  re-runs matching, never overrides a reason code. This is the "right tool in
  the right place" the track criteria ask for — and knowing when NOT to use AI
  is part of that judgment.

- **No auth, no multi-tenant scope.** This is a hackathon demo running against
  synthetic data. Adding authentication, RBAC, or tenant isolation would be
  engineering for a deployment that doesn't exist yet. The `.env` file holds a
  single Groq API key; the app serves one user at localhost:8505.

- **Cash position check exists because reconciliation without it is
  incomplete.** Matching records one-by-one tells you *which* transactions
  reconciled. The cash position cross-check tells you whether the *totals*
  agree — which is what an actual finance controller cares about. The INR
  1,675.27 delta is flagged because it exceeds the INR 50 tolerance, and the
  dashboard explains why (gateway fees + rounding differences).

## What it does

1. **Generates** 60 synthetic Razorpay-style payment records with realistic INR amounts (INR 199–INR 45,000)
2. **Produces** a deliberately messy bank settlement file with 6 types of injected discrepancies
3. **Matches** payments in two passes (exact → fuzzy fallback with fee/lag/typo tolerance)
4. **Explains** each exception using an LLM (Groq / qwen/qwen3.8-27b), with graceful mock fallback when no API key is configured
5. **Visualises** results via a Streamlit dashboard with match-rate charts, exception drill-down, and cash position display
6. **Answers questions** about the reconciliation results via a read-only Q&A layer (uses LLM, grounded in existing data only)

### Injected bank messiness (ground truth via `_mutation` column)

| Mismatch type             | Count | What it simulates                        |
|---------------------------|-------|------------------------------------------|
| Gateway fee (2% reduction)| 5     | Bank receives amount minus processing fee|
| Settlement lag (T+1/T+2)  | 4     | Bank settles 1–2 days after payment      |
| Order-ref typo            | 3     | One character changed/dropped/inserted   |
| Duplicate settlement      | 3     | Same order_ref appears twice in bank file|
| Rounding difference       | 2     | Paise-level diff (< INR 0.50)            |
| Missing from settlement   | 3     | Payment made but never settled           |

## Results (fresh run, seed=42)

| Metric                              | Value  |
|-------------------------------------|--------|
| Internal records                    | 60     |
| Bank settlement rows                | 60     |
| **Matched**                         | **57** |
| **Match rate**                      | **95.0%** |
| Exceptions                          | 6      |
| Cash position delta (internal − bank) | INR 1,675.27 |
| Cash position status                | FLAGGED (exceeds INR 50 tolerance) |

### Match types

| Type         | Count | Description                                              |
|--------------|-------|----------------------------------------------------------|
| `exact`      | 43    | order_ref + amount + date all identical                  |
| `fuzzy_fee`  | 7     | order_ref exact, amount within 2.5% (gateway fee)        |
| `fuzzy_lag_T+1` | 1  | order_ref exact, same amount, settlement T+1             |
| `fuzzy_lag_T+2` | 3  | order_ref exact, same amount, settlement T+2             |
| `fuzzy_typo` | 3     | Levenshtein edit-distance 1, amount exact, date ≤ 1 day  |

### Exceptions

| Reason code                    | Count | What it means                                      |
|--------------------------------|-------|----------------------------------------------------|
| `MISSING_FROM_SETTLEMENT`      | 3     | No bank settlement found — stuck/failed settlement |
| `DUPLICATE_SETTLEMENT`         | 3     | Extra duplicate bank row (primary already matched)  |

### Cash position cross-check

Sum of matched internal amounts: INR 636,427.24
Sum of matched bank amounts:     INR 634,751.97
Delta:                           INR 1,675.27

The delta exists because 3 gateway-fee records and 2 rounding-diff records have
bank amounts lower than the internal amounts. These are expected operational
differences, not errors — the engine correctly matched and tagged them.

## How to run

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd finance-controller

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# 3. Generate synthetic data
python -m data.generate_data

# 4. Run the deterministic matcher
python -m src.matcher

# 5. (Optional) Run the LLM explainer — needs a Groq API key in .env
#    Without a key, it falls back to mock explanations automatically
python -m src.llm_explainer

# 6. Launch the dashboard
streamlit run app.py
# Opens at http://localhost:8505

# 7. Run tests (49 tests)
pytest tests/ -v
```

**Without an API key:** Steps 1–4 and 6–7 work fully. The matcher is
independent of any API. Step 5 falls back to deterministic mock explanations.
To get real LLM explanations, add `GROQ_API_KEY=your_key` to `.env` (free key
at https://console.groq.com/keys).

## Architecture

### Deterministic matching engine (no LLM)

**Pass 1 — Exact match:** order_ref + amount + date all identical.

**Pass 2 — Fuzzy fallback** for anything unmatched after pass 1:
- Strategy A (fee/lag): order_ref exact + amount within 2.5% + date ≤ 2 days
- Strategy B (typo): Levenshtein edit-distance 1 + amount exact + date ≤ 1 day

**Exception classification** with reason codes:
`MISSING_FROM_SETTLEMENT`, `NO_MATCH_FOUND`, `AMOUNT_MISMATCH_UNRESOLVED`,
`DUPLICATE_SETTLEMENT`, `MISSING_FROM_INTERNAL`

### LLM exception explainer

After the deterministic matcher finishes, the LLM explainer reads
`exceptions.csv` and generates human-readable explanations for each
exception. **It never re-decides a match, never overrides an exception
classification, and never modifies `matched_pairs.csv`.**

This separation is intentional and stated explicitly in the code:

> "This module explains exceptions already classified by the deterministic
> matcher. It does not resolve, re-match, or override any matching decision."

**Graceful degradation:** If no Groq API key is configured (or the API call
fails), the explainer falls back to deterministic mock explanations based on
the reason code. The pipeline never crashes without an API key.

### Read-only Q&A layer (src/qa_agent.py)

A bounded, single-question-single-answer Q&A feature on top of the
reconciliation output. It answers questions like "Why wasn't ORD10254
matched?" by retrieving relevant records from the existing CSVs and
sending them (plus the question) to the LLM.

**Key constraints:**
- Only reads from `matched_pairs.csv` and `exceptions_explained.csv` — never
  re-runs matching, never modifies any output file
- If the question contains an order_ref or payment_id, it pulls that
  specific row as context. Otherwise it passes a compact summary.
- Falls back to mock answers when no API key is configured
- Rejects non-finance questions explicitly

This is explicitly an optional, read-only convenience layer — not a new
agent, not a new decision engine.

### Streamlit dashboard

The dashboard loads existing output CSVs (or runs the full pipeline on demand)
and displays:
- Top metrics row: match rate, matched count, exceptions, cash position delta
- Altair bar chart of match type breakdown (5-color accent palette)
- Filterable matched pairs table (by match type)
- Exceptions table with LLM explanations and confidence levels
- Severity badges (red for manual review, amber for high, teal for medium)
- Cash position cross-check detail
- Downloadable audit log

## Project structure

```
finance-controller/
├── data/
│   └── generate_data.py       # Synthetic data generator with injectable messiness
├── src/
│   ├── matcher.py             # Core two-pass reconciliation engine (deterministic)
│   ├── llm_explainer.py       # LLM exception explanations (Groq, with mock fallback)
│   └── qa_agent.py            # Read-only Q&A over reconciliation output
├── outputs/
│   ├── matched_pairs.csv      # 57 matched records with match_type & tolerance
│   ├── exceptions.csv         # 6 exceptions with reason codes
│   ├── exceptions_explained.csv # Exceptions + LLM explanations + confidence
│   └── audit_log.csv          # Full audit trail + cash position summary
├── tests/
│   ├── test_matcher.py        # 20 tests: edit distance, exact, fuzzy, exceptions, pipeline
│   ├── test_llm_explainer.py  # 14 tests: mock, API failure, context building, parsing
│   └── test_qa_agent.py       # 15 tests: ID extraction, context retrieval, mock fallback, no side effects
├── logs/
│   └── failure_log.md         # Debugging and design tradeoff notes
├── .streamlit/
│   └── config.toml            # Dark fintech theme (navy bg, teal accent)
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── .env                       # API keys (gitignored)
├── .gitignore
└── README.md
```

## Design decisions

- **QA agent is read-only**: It answers questions about existing results, it
  does not perform new reconciliation. The system prompt explicitly rejects
  non-finance questions and instructs the model to never invent numbers.

- **Deterministic generation**: `random.seed(42)` ensures reproducible test data
- **Mutation tagging**: Bank CSV includes `_mutation` column for ground-truth auditability
- **Two-pass matching**: Exact match first (zero ambiguity), then fuzzy fallback (tolerance-based)
- **Custom Levenshtein**: Lightweight O(n) edit-distance-1 check, no external dependency
- **Cash position cross-check**: Reports delta between matched internal and bank totals, flags if > INR 50
- **Honest reporting**: All 6 exceptions are real, not cherry-picked — 3 genuinely stuck settlements + 3 duplicate bank rows
- **LLM explains, never decides**: The matcher is deterministic and reproducible. The LLM only generates natural-language explanations for exceptions — it never modifies match decisions.
