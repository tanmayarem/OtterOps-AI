# AI Finance Controller

**Razorpay Builtathon - Track 04: AI Finance Controller**

A payment gateway settlement reconciliation system that matches internal payment records against bank settlement data, detects discrepancies, and reports match rates with honest exception lists.

## What it does

- Generates 60 synthetic Razorpay-style payment records with realistic INR amounts
- Produces a deliberately messy bank settlement file with 6 types of injected discrepancies:
  - 5 records: 2% gateway fee reduction
  - 4 records: T+1/T+2 settlement lag
  - 3 records: order_ref typos (fuzzy matching target)
  - 3 records: duplicate settlement rows
  - 3 records: missing from settlement entirely
  - 2 records: paise-level rounding differences
- Matches payments by order_ref with smart classification (exact, gateway fee, date lag, rounding diff)
- Reports match rate and unresolved exceptions
- Streams results via a Streamlit dashboard

## Results

| Metric | Value |
|--------|-------|
| Internal records | 60 |
| Bank rows | 60 (includes duplicates) |
| Matched | 54 (90.0%) |
| Exceptions | 6 (3 missing + 3 typos) |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python -m data.generate_data

# Run tests
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
│   ├── matcher.py             # Core matching engine
│   └── llm_explainer.py       # LLM-based exception explanations (Day 2)
├── outputs/                   # Generated reconciliation artifacts
├── tests/
│   └── test_matcher.py        # 8 test cases covering all match types
├── logs/
│   └── failure_log.md         # Running bug/failure-recovery notes
├── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Match types

| Type | Description |
|------|-------------|
| `exact` | Amount matches within tolerance |
| `gateway_fee` | Bank amount is ~2% less (gateway fee applied) |
| `gateway_fee_or_rounding` | Small delta, ambiguous between fee and rounding |
| `date_lag` | Amount matches but settlement date is T+1/T+2 |
| `rounding_diff` | Paise-level difference (< ₹0.50) |
| `missing` | No settlement found in bank file |
| `mismatch` | Amount significantly different, needs manual review |

## Design decisions

- **Deterministic generation**: `random.seed(42)` ensures reproducible test data
- **Mutation tagging**: Bank CSV includes `_mutation` column for auditability
- **Tolerance-based matching**: Configurable thresholds for exact vs fuzzy matches
- **Honest reporting**: Includes all exceptions, not cherry-picked matches
