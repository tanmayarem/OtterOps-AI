# AI Finance Controller

**Razorpay Builtathon — Track 04: AI Finance Controller**

A deterministic payment gateway settlement reconciliation engine that matches
internal payment records against bank settlement data across a 60-record batch,
reports match rate, produces an honest exception list, and explains each
exception using an LLM — without letting the LLM re-decide any match.

## What it does

1. **Generates** 60 synthetic Razorpay-style payment records with realistic INR amounts (INR 199–INR 45,000)
2. **Produces** a deliberately messy bank settlement file with 6 types of injected discrepancies
3. **Matches** payments in two passes (exact → fuzzy fallback with fee/lag/typo tolerance)
4. **Explains** each exception using an LLM (Groq / llama-3.3-70b-versatile), with graceful mock fallback when no API key is configured
5. **Visualises** results via a Streamlit dashboard with match-rate charts, exception drill-down, and cash position display

### Injected bank messiness (ground truth via `_mutation` column)

| Mismatch type             | Count | What it simulates                        |
|---------------------------|-------|------------------------------------------|
| Gateway fee (2% reduction)| 5     | Bank receives amount minus processing fee|
| Settlement lag (T+1/T+2)  | 4     | Bank settles 1–2 days after payment      |
| Order-ref typo            | 3     | One character changed/dropped/inserted   |
| Duplicate settlement      | 3     | Same order_ref appears twice in bank file|
| Rounding difference       | 2     | Paise-level diff (< INR 0.50)            |
| Missing from settlement   | 3     | Payment made but never settled           |

## Results

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

## Architecture

### Deterministic matching engine (no LLM)

**Pass 1 — Exact match:** order_ref + amount + date all identical.

**Pass 2 — Fuzzy fallback** for anything unmatched after pass 1:
- Strategy A (fee/lag): order_ref exact + amount within 2.5% + date ≤ 2 days
- Strategy B (typo): Levenshtein edit-distance 1 + amount exact + date ≤ 1 day

**Exception classification** with reason codes:
`MISSING_FROM_SETTLEMENT`, `NO_MATCH_FOUND`, `AMOUNT_MISMATCH_UNRESOLVED`,
`DUPLICATE_SETTLEMENT`, `MISSING_FROM_INTERNAL`

### LLM exception explainer (the ONLY AI in the pipeline)

After the deterministic matcher finishes, the LLM explainer reads
`exceptions.csv` and generates human-readable explanations for each
exception. **It never re-decides a match, never overrides an exception
classification, and never modifies `matched_pairs.csv`.**

This separation is intentional and stated explicitly in the code:

> "This module explains exceptions already classified by the deterministic
> matcher. It does not resolve, re-match, or override any matching decision."

**Why this design?** The track criteria ask for "AI judgment — the right tool
in the right place, and where you chose not to use one." Matching must be
reproducible and auditable — a finance controller that gives different results
on the same data is untrustworthy. LLMs add genuine value when explaining
*why* an exception occurred in natural language for human review, but they
should never be the source of truth for whether two records match.

**Graceful degradation:** If no Groq API key is configured (or the API call
fails), the explainer falls back to deterministic mock explanations based on
the reason code. The pipeline never crashes without an API key.

### Streamlit dashboard

The dashboard loads existing output CSVs (or runs the full pipeline on demand)
and displays:
- Top metrics row: match rate, matched count, exceptions, cash position delta
- Filterable matched pairs table (by match type)
- Exceptions table with LLM explanations and confidence levels
- Manual review badges (red for low confidence, amber for medium, green for high)
- Cash position cross-check detail
- Downloadable audit log

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python -m data.generate_data

# Run reconciliation (matcher + LLM explainer)
python -m src.matcher
python -m src.llm_explainer

# Run tests (34 tests)
pytest tests/ -v

# Launch dashboard
streamlit run app.py
```

**Without an API key:** The pipeline runs in mock mode — matcher is fully
functional, LLM explanations use template responses. Add a Groq key to
`.env` for real LLM explanations.

## Project structure

```
finance-controller/
├── data/
│   └── generate_data.py       # Synthetic data generator with injectable messiness
├── src/
│   ├── matcher.py             # Core two-pass reconciliation engine (deterministic)
│   └── llm_explainer.py       # LLM exception explanations (Groq, with mock fallback)
├── outputs/
│   ├── matched_pairs.csv      # 57 matched records with match_type & tolerance
│   ├── exceptions.csv         # 6 exceptions with reason codes
│   ├── exceptions_explained.csv # Exceptions + LLM explanations + confidence
│   └── audit_log.csv          # Full audit trail + cash position summary
├── tests/
│   ├── test_matcher.py        # 20 tests: edit distance, exact, fuzzy, exceptions, pipeline
│   └── test_llm_explainer.py  # 14 tests: mock, API failure, context building, parsing
├── logs/
│   └── failure_log.md         # Debugging and design tradeoff notes
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── .env                       # API keys (gitignored)
├── .gitignore
└── README.md
```

## Design decisions

- **Deterministic generation**: `random.seed(42)` ensures reproducible test data
- **Mutation tagging**: Bank CSV includes `_mutation` column for ground-truth auditability
- **Two-pass matching**: Exact match first (zero ambiguity), then fuzzy fallback (tolerance-based)
- **Custom Levenshtein**: Lightweight O(n) edit-distance-1 check, no external dependency
- **Cash position cross-check**: Reports delta between matched internal and bank totals, flags if > INR 50
- **Honest reporting**: All 6 exceptions are real, not cherry-picked — 3 genuinely stuck settlements + 3 duplicate bank rows
- **LLM explains, never decides**: The matcher is deterministic and reproducible. The LLM only generates natural-language explanations for exceptions — it never modifies match decisions. This is the "right tool in the right place" the track criteria ask for.
