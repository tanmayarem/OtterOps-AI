"""Tests for src/matcher.py — two-pass reconciliation engine."""
import pandas as pd
import pytest

from src.matcher import (
    exact_match,
    fuzzy_match,
    classify_exceptions,
    reconcile,
    _edit_distance_1,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _make_internal(rows):
    """Build an internal-payments DataFrame from a list of dicts."""
    if not rows:
        return pd.DataFrame(columns=["payment_id", "date", "amount", "order_ref", "status"])
    df = pd.DataFrame(rows)
    df["amount"] = pd.to_numeric(df["amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_bank(rows):
    """Build a bank-settlement DataFrame from a list of dicts."""
    if not rows:
        return pd.DataFrame(columns=["settlement_id", "order_ref", "amount", "settlement_date", "_mutation"])
    df = pd.DataFrame(rows)
    df["amount"] = pd.to_numeric(df["amount"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


# ── Levenshtein helper ──────────────────────────────────────────────
class TestEditDistance1:
    def test_identical(self):
        assert _edit_distance_1("ORD100", "ORD100") is False

    def test_substitution(self):
        assert _edit_distance_1("ORD100", "ORD101") is True

    def test_insertion(self):
        assert _edit_distance_1("ORD100", "ORD1000") is True

    def test_deletion(self):
        assert _edit_distance_1("ORD1000", "ORD100") is True

    def test_two_edits(self):
        assert _edit_distance_1("ORD100", "ORD301") is False

    def test_completely_different(self):
        assert _edit_distance_1("ORD100", "XYZ999") is False

    def test_empty_strings(self):
        assert _edit_distance_1("", "A") is True

    def test_length_diff_two(self):
        assert _edit_distance_1("ORD100", "ORD") is False


# ── Pass 1: exact match ─────────────────────────────────────────────
class TestExactMatch:
    def test_perfect_match(self):
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
            {"payment_id": "PAY0002", "date": "2026-08-21", "amount": 2500.50,
             "order_ref": "ORD101", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 1000.00,
             "settlement_date": "2026-08-20", "_mutation": "clean"},
            {"settlement_id": "SET002", "order_ref": "ORD101", "amount": 2500.50,
             "settlement_date": "2026-08-21", "_mutation": "clean"},
        ])
        matched, remaining, consumed = exact_match(internal, bank)
        assert len(matched) == 2
        assert all(matched["match_type"] == "exact")
        assert len(remaining) == 0

    def test_no_match_different_amount(self):
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 980.00,
             "settlement_date": "2026-08-20", "_mutation": "gateway_fee_2pct"},
        ])
        matched, remaining, consumed = exact_match(internal, bank)
        assert len(matched) == 0
        assert len(remaining) == 1

    def test_duplicate_bank_rows(self):
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 3000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 3000.00,
             "settlement_date": "2026-08-20", "_mutation": "primary"},
            {"settlement_id": "DUP001", "order_ref": "ORD100", "amount": 3000.00,
             "settlement_date": "2026-08-20", "_mutation": "duplicate"},
        ])
        matched, remaining, consumed = exact_match(internal, bank)
        assert len(matched) == 1  # matched once, not twice
        assert len(consumed) == 1  # only one bank row consumed


# ── Pass 2: fuzzy match ─────────────────────────────────────────────
class TestFuzzyMatch:
    def test_fee_adjusted(self):
        """Bank amount is 2% less — should match via fee tolerance."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 5000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 4900.00,
             "settlement_date": "2026-08-20", "_mutation": "gateway_fee_2pct"},
        ])
        matched, unmatched_int, unmatched_bank = fuzzy_match(internal, bank)
        assert len(matched) == 1
        assert matched.iloc[0]["match_type"] == "fuzzy_fee"

    def test_date_lag(self):
        """Same amount, settlement 2 days later."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 3306.16,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 3306.16,
             "settlement_date": "2026-08-22", "_mutation": "settlement_lag_T+2"},
        ])
        matched, unmatched_int, unmatched_bank = fuzzy_match(internal, bank)
        assert len(matched) == 1
        assert matched.iloc[0]["match_type"] == "fuzzy_lag_T+2"
        assert matched.iloc[0]["date_lag_days"] == 2

    def test_typo_correction(self):
        """Order ref off by one character, amount and date exact."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD101", "amount": 1000.00,
             "settlement_date": "2026-08-20", "_mutation": "order_ref_typo"},
        ])
        matched, unmatched_int, unmatched_bank = fuzzy_match(internal, bank)
        assert len(matched) == 1
        assert matched.iloc[0]["match_type"] == "fuzzy_typo"
        assert matched.iloc[0]["bank_order_ref"] == "ORD101"

    def test_no_match_completely_different(self):
        """Nothing matches — both sides remain unmatched."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD999", "amount": 9999.00,
             "settlement_date": "2026-08-20", "_mutation": "clean"},
        ])
        matched, unmatched_int, unmatched_bank = fuzzy_match(internal, bank)
        assert len(matched) == 0
        assert len(unmatched_int) == 1
        assert len(unmatched_bank) == 1


# ── Exception classification ─────────────────────────────────────────
class TestClassifyExceptions:
    def test_missing_from_settlement(self):
        """Internal record has no bank row → MISSING_FROM_SETTLEMENT."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD999", "amount": 500.00,
             "settlement_date": "2026-08-20", "_mutation": "clean"},
        ])
        # Pass unmatched internal + bank, but also pass full data for lookups
        exceptions = classify_exceptions(internal, bank, internal, bank)
        # 2 exceptions: internal MISSING_FROM_SETTLEMENT + bank MISSING_FROM_INTERNAL
        int_exc = exceptions[exceptions["record_type"] == "internal"]
        assert len(int_exc) == 1
        assert int_exc.iloc[0]["reason_code"] == "MISSING_FROM_SETTLEMENT"
        bank_exc = exceptions[exceptions["record_type"] == "bank"]
        assert len(bank_exc) == 1
        assert bank_exc.iloc[0]["reason_code"] == "MISSING_FROM_INTERNAL"

    def test_amount_mismatch(self):
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 5000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 3000.00,
             "settlement_date": "2026-08-20", "_mutation": "clean"},
        ])
        exceptions = classify_exceptions(internal, bank, internal, bank)
        int_exc = exceptions[exceptions["record_type"] == "internal"]
        assert len(int_exc) == 1
        assert int_exc.iloc[0]["reason_code"] == "AMOUNT_MISMATCH_UNRESOLVED"

    def test_duplicate_settlement_bank_exception(self):
        """Duplicate bank row for an order_ref that has an internal record."""
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 3000.00,
             "order_ref": "ORD100", "status": "captured"},
        ])
        bank = _make_bank([
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 3000.00,
             "settlement_date": "2026-08-20", "_mutation": "primary"},
            {"settlement_id": "DUP001", "order_ref": "ORD100", "amount": 3000.00,
             "settlement_date": "2026-08-20", "_mutation": "duplicate_of_SET001"},
        ])
        # Pass: both bank rows consumed in pass 1 (exact), nothing unmatched
        # Let's simulate a scenario where both are unmatched
        empty_int = _make_internal([])
        exceptions = classify_exceptions(empty_int, bank, internal, bank)
        dup_rows = exceptions[exceptions["reason_code"] == "DUPLICATE_SETTLEMENT"]
        assert len(dup_rows) >= 1  # at least the duplicate flagged


# ── Full pipeline ────────────────────────────────────────────────────
class TestReconcilePipeline:
    def test_end_to_end(self, tmp_path):
        """Run the full pipeline on synthetic data and verify outputs."""
        # Create mini datasets
        internal = _make_internal([
            {"payment_id": "PAY0001", "date": "2026-08-20", "amount": 1000.00,
             "order_ref": "ORD100", "status": "captured"},
            {"payment_id": "PAY0002", "date": "2026-08-21", "amount": 5000.00,
             "order_ref": "ORD101", "status": "captured"},
            {"payment_id": "PAY0003", "date": "2026-08-22", "amount": 3000.00,
             "order_ref": "ORD102", "status": "captured"},
            {"payment_id": "PAY0004", "date": "2026-08-23", "amount": 1500.00,
             "order_ref": "ORD103", "status": "captured"},
            {"payment_id": "PAY0005", "date": "2026-08-24", "amount": 2000.00,
             "order_ref": "ORD104", "status": "captured"},
        ])
        bank = _make_bank([
            # Exact match
            {"settlement_id": "SET001", "order_ref": "ORD100", "amount": 1000.00,
             "settlement_date": "2026-08-20", "_mutation": "clean"},
            # Gateway fee (2% less)
            {"settlement_id": "SET002", "order_ref": "ORD101", "amount": 4900.00,
             "settlement_date": "2026-08-21", "_mutation": "gateway_fee_2pct"},
            # Date lag (T+2)
            {"settlement_id": "SET003", "order_ref": "ORD102", "amount": 3000.00,
             "settlement_date": "2026-08-24", "_mutation": "settlement_lag_T+2"},
            # Rounding diff
            {"settlement_id": "SET004", "order_ref": "ORD103", "amount": 1500.25,
             "settlement_date": "2026-08-23", "_mutation": "rounding_diff"},
            # Missing: no bank row for ORD104
        ])

        int_path = tmp_path / "internal.csv"
        bank_path = tmp_path / "bank.csv"
        out_path = tmp_path / "outputs"

        internal.to_csv(int_path, index=False)
        bank.to_csv(bank_path, index=False)

        matched, exceptions, audit = reconcile(int_path, bank_path, out_path)

        # Verify outputs were written
        assert (out_path / "matched_pairs.csv").exists()
        assert (out_path / "exceptions.csv").exists()
        assert (out_path / "audit_log.csv").exists()

        # 4 should match (exact, fee, lag, rounding all within fuzzy tolerances)
        # 1 missing (ORD104 not in bank) + 0 bank exceptions (all bank rows matched)
        assert len(matched) == 4
        int_exc = exceptions[exceptions["record_type"] == "internal"]
        assert len(int_exc) == 1
        assert int_exc.iloc[0]["reason_code"] == "MISSING_FROM_SETTLEMENT"
        assert int_exc.iloc[0]["order_ref"] == "ORD104"


# ── Real data smoke test ────────────────────────────────────────────
class TestRealData:
    """Run against the actual generated CSVs if they exist."""

    def test_full_dataset(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        int_path = root / "data" / "internal_payments.csv"
        bank_path = root / "data" / "bank_settlement.csv"

        if not int_path.exists() or not bank_path.exists():
            pytest.skip("Generated data not found — run generate_data.py first")

        matched, exceptions, audit = reconcile(int_path, bank_path, root / "outputs")

        # Sanity checks on the 60-record dataset
        assert len(matched) >= 50, f"Expected ≥50 matches, got {len(matched)}"
        int_exc = exceptions[exceptions["record_type"] == "internal"]
        assert len(int_exc) <= 10, f"Expected ≤10 internal exceptions, got {len(int_exc)}"

        # Match rate should be ≥85%
        rate = len(matched) / 60 * 100
        assert rate >= 85, f"Match rate {rate:.1f}% below 85% threshold"

        # Should have multiple match types
        if not matched.empty:
            assert matched["match_type"].nunique() >= 2
