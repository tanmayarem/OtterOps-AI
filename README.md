# AI Finance Controller

**Razorpay Builtathon — Track 04: AI Finance Controller**

A deterministic payment gateway settlement reconciliation engine that matches
internal payment records against bank settlement data across a 60-record batch,
reports match rate, and produces an honest exception list.

## What it does

1. **Generates** 60 synthetic Razorpay-style payment records with realistic INR amounts (₹199–₹45,000)
2. **Produces** a deliberately messy bank settlement file with 6 types of injected discrepancies
3. **Matches** payments in two passes (exact → fuzzy fallback with fee/lag/typo tolerance)
4. **Reports** match rate, cash position delta, and unresolved exceptions
5. **Visualises** results via a Streamlit dashboard

### Injected bank messiness (ground truth via `_mutation` column)

| Mismatch type             | Count | What it simulates                        |
|---------------------------|-------|------------------------------------------|
| Gateway fee (2% reduction)| 5     | Bank receives amount minus processing fee|
| Settlement lag (T+1/T+2)  | 4     | Bank settles 1–2 days after payment      |
| Order-ref typo            | 3     | One character changed/dropped/inserted   |
| Duplicate settlement      | 3     | Same order_ref appears twice in bank file|
| Rounding difference       | 2     | Paise-level diff (< ₹0.50)              |
| Missing from settlement   | 3     | Payment made but never settled           |

## Results

| Metric                              | Value  |
|-------------------------------------|--------|
| Internal records                    | 60     |
| Bank settlement rows                | 60     |
| **Matched**                         | **57** |
| **Match rate**                      | **95.0%** |
| Exceptions                          | 6      |
| Cash position delta (internal − bank) | ₹1,675.27 |
| Cash position status                | ⚠ FLAGGED (exceeds ₹50 tolerance) |

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

Sum of matched internal amounts: ₹636,427.24
Sum of matched bank amounts:     ₹634,751.97
Delta:                           ₹1,675.27

The delta exists because 3 gateway-fee records and 2 rounding-diff records have
bank amounts lower than the internal amounts. These are expected operational
differences, not errors — the engine correctly matched and tagged them.

## Architecture

### Two-pass matching engine (no LLM, fully deterministic)

**Pass 1 — Exact match:** order_ref + amount + date all identical.

**Pass 2 — Fuzzy fallback** for anything unmatched after pass 1:
- Strategy A (fee/lag): order_ref exact + amount within 2.5% + date ≤ 2 days
- Strategy B (typo): Levenshtein edit-distance 1 + amount exact + date ≤ 1 day

**Exception classification** with reason codes:
`MISSING_FROM_SETTLEMENT`, `NO_MATCH_FOUND`, `AMOUNT_MISMATCH_UNRESOLVED`,
`DUPLICATE_SETTLEMENT`, `MISSING_FROM_INTERNAL`

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python -m data.generate_data

# Run reconciliation
python -m src.matcher

# Run tests (20 tests)
pytest tests/ -v

# Launch dashboard
streamlit run app.py
```

## Project structure

```
finance-controller/
├── data/
│   └── generate_data.py       # Synthetic data generator with injectable messiness
├── src/
│   ├── matcher.py             # Core two-pass reconciliation engine
│   └── llm_explainer.py       # LLM-based exception explanations (Day 2)
├── outputs/
│   ├── matched_pairs.csv      # 57 matched records with match_type & tolerance
│   ├── exceptions.csv         # 6 exceptions with reason codes
│   └── audit_log.csv          # Full audit trail + cash position summary
├── tests/
│   └── test_matcher.py        # 20 tests: edit distance, exact, fuzzy, exceptions, pipeline
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
- **Cash position cross-check**: Reports delta between matched internal and bank totals, flags if > ₹50
- **Honest reporting**: All 6 exceptions are real, not cherry-picked — 3 genuinely stuck settlements + 3 duplicate bank rows
- **No LLM in the matcher**: Core engine is fully deterministic and reproducible; LLM is reserved for exception explanations only
