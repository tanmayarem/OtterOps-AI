"""Tests for src/llm_explainer.py — LLM exception explanations.

All tests use mocked LLM responses.  No real API calls are made.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.llm_explainer import (
    explain_single,
    explain_exceptions,
    build_context,
    _parse_llm_response,
    MOCK_RESPONSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_internal():
    return pd.DataFrame({
        "payment_id": ["PAY0001", "PAY0002", "PAY0003"],
        "date": pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-22"]),
        "amount": [1000.00, 5000.00, 3000.00],
        "order_ref": ["ORD100", "ORD101", "ORD102"],
        "status": ["captured", "captured", "captured"],
    })


@pytest.fixture
def sample_bank():
    return pd.DataFrame({
        "settlement_id": ["SET001", "SET002", "SET003"],
        "order_ref": ["ORD100", "ORD101", "ORD999"],
        "amount": [1000.00, 4900.00, 2500.00],
        "settlement_date": pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-22"]),
        "_mutation": ["clean", "gateway_fee_2pct", "clean"],
    })


@pytest.fixture
def missing_settlement_exception():
    return {
        "record_type": "internal",
        "payment_id": "PAY0001",
        "settlement_id": "",
        "order_ref": "ORD100",
        "amount": 1000.00,
        "reason_code": "MISSING_FROM_SETTLEMENT",
        "notes": "No bank settlement found for this order_ref",
    }


@pytest.fixture
def duplicate_exception():
    return {
        "record_type": "bank",
        "payment_id": "",
        "settlement_id": "DUP001",
        "order_ref": "ORD100",
        "amount": 1000.00,
        "reason_code": "DUPLICATE_SETTLEMENT",
        "notes": "Extra duplicate bank row for ORD100",
    }


# ---------------------------------------------------------------------------
# Test 1: Mock explanation returned (no API key)
# ---------------------------------------------------------------------------
class TestMockExplanation:
    def test_returns_mock_for_missing_settlement(
        self, missing_settlement_exception, sample_internal, sample_bank
    ):
        """Without an API key, should return the mock explanation."""
        explanation, confidence = explain_single(
            missing_settlement_exception, sample_internal, sample_bank,
            api_key=None,
        )
        assert explanation  # non-empty
        assert confidence in ("high", "medium", "low")
        # Should match the mock template
        assert "stuck" in explanation.lower() or "settlement" in explanation.lower()

    def test_returns_mock_for_duplicate(
        self, duplicate_exception, sample_internal, sample_bank
    ):
        explanation, confidence = explain_single(
            duplicate_exception, sample_internal, sample_bank,
            api_key=None,
        )
        assert explanation
        assert confidence in ("high", "medium", "low")

    def test_all_reason_codes_have_mock(self):
        """Every known reason code should have a mock response."""
        for code in [
            "MISSING_FROM_SETTLEMENT",
            "AMOUNT_MISMATCH_UNRESOLVED",
            "NO_MATCH_FOUND",
            "DUPLICATE_SETTLEMENT",
            "MISSING_FROM_INTERNAL",
        ]:
            assert code in MOCK_RESPONSES
            assert MOCK_RESPONSES[code][0]  # explanation non-empty
            assert MOCK_RESPONSES[code][1] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Test 2: API failure handled gracefully
# ---------------------------------------------------------------------------
class TestAPIFailure:
    def test_groq_failure_falls_back_to_mock(
        self, missing_settlement_exception, sample_internal, sample_bank
    ):
        """If the LLM call raises, should fall back to mock without crashing."""
        with patch("src.llm_explainer._call_llm", side_effect=RuntimeError("API down")):
            explanation, confidence = explain_single(
                missing_settlement_exception, sample_internal, sample_bank,
                api_key="fake-key-for-testing",
            )
        assert explanation  # non-empty
        assert confidence in ("high", "medium", "low")

    def test_timeout_falls_back_to_mock(
        self, missing_settlement_exception, sample_internal, sample_bank
    ):
        with patch("src.llm_explainer._call_llm", side_effect=TimeoutError("timeout")):
            explanation, confidence = explain_single(
                missing_settlement_exception, sample_internal, sample_bank,
                api_key="fake-key",
            )
        assert explanation
        assert confidence in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Test 3: Mock mode works without a key
# ---------------------------------------------------------------------------
class TestMockMode:
    def test_no_key_uses_mock(self, missing_settlement_exception, sample_internal, sample_bank):
        """When GROQ_API_KEY is not set, should use mock mode."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove GROQ_API_KEY if present
            os_environ_backup = {}
            for k in list(__import__("os").environ.keys()):
                if "GROQ" in k:
                    os_environ_backup[k] = __import__("os").environ.pop(k)
            try:
                explanation, confidence = explain_single(
                    missing_settlement_exception, sample_internal, sample_bank,
                    api_key=None,
                )
                assert explanation
                assert confidence in ("high", "medium", "low")
            finally:
                __import__("os").environ.update(os_environ_backup)


# ---------------------------------------------------------------------------
# Test 4: Context payload building
# ---------------------------------------------------------------------------
class TestBuildContext:
    def test_internal_exception_includes_bank_candidates(
        self, missing_settlement_exception, sample_internal, sample_bank
    ):
        """Context for an internal exception should include bank candidates."""
        context = build_context(
            missing_settlement_exception, sample_internal, sample_bank
        )
        assert "ORD100" in context
        assert "CLOSEST BANK CANDIDATES" in context or "NO bank records" in context
        assert "MISSING_FROM_SETTLEMENT" in context

    def test_bank_exception_includes_internal_candidates(
        self, duplicate_exception, sample_internal, sample_bank
    ):
        """Context for a bank exception should include internal candidates."""
        context = build_context(
            duplicate_exception, sample_internal, sample_bank
        )
        assert "ORD100" in context
        assert "CLOSEST INTERNAL CANDIDATES" in context or "NO internal records" in context

    def test_context_includes_all_exception_fields(
        self, missing_settlement_exception, sample_internal, sample_bank
    ):
        context = build_context(
            missing_settlement_exception, sample_internal, sample_bank
        )
        assert "Record type: internal" in context
        assert "Order ref: ORD100" in context
        assert "Payment ID: PAY0001" in context
        assert "Reason code: MISSING_FROM_SETTLEMENT" in context
        assert "INR 1,000.00" in context


# ---------------------------------------------------------------------------
# Test 5: LLM response parser
# ---------------------------------------------------------------------------
class TestParseLLMResponse:
    def test_standard_format(self):
        raw = "EXPLANATION: Payment stuck in settlement pipeline.\nCONFIDENCE: high"
        explanation, confidence = _parse_llm_response(raw)
        assert explanation == "Payment stuck in settlement pipeline."
        assert confidence == "high"

    def test_no_format_fallback(self):
        raw = "This is just a plain response with no special format."
        explanation, confidence = _parse_llm_response(raw)
        assert "plain response" in explanation
        assert confidence == "low"  # default when no CONFIDENCE line

    def test_invalid_confidence_defaults_to_medium(self):
        raw = "EXPLANATION: Something happened.\nCONFIDENCE: uncertain"
        explanation, confidence = _parse_llm_response(raw)
        assert confidence == "medium"

    def test_empty_response(self):
        raw = ""
        explanation, confidence = _parse_llm_response(raw)
        assert confidence == "low"


# ---------------------------------------------------------------------------
# Test 6: Full pipeline with mocked LLM
# ---------------------------------------------------------------------------
class TestExplainExceptionsPipeline:
    def test_full_pipeline_mock(self, tmp_path, sample_internal, sample_bank):
        """Run explain_exceptions with mocked LLM on a real exceptions file."""
        # Create test files
        exceptions_df = pd.DataFrame([
            {
                "record_type": "internal",
                "payment_id": "PAY0001",
                "settlement_id": "",
                "order_ref": "ORD100",
                "amount": 1000.00,
                "reason_code": "MISSING_FROM_SETTLEMENT",
                "notes": "No bank settlement found",
            },
            {
                "record_type": "bank",
                "payment_id": "",
                "settlement_id": "DUP001",
                "order_ref": "ORD101",
                "amount": 5000.00,
                "reason_code": "DUPLICATE_SETTLEMENT",
                "notes": "Extra duplicate row",
            },
        ])

        exc_path = tmp_path / "exceptions.csv"
        int_path = tmp_path / "internal.csv"
        bank_path = tmp_path / "bank.csv"
        out_path = tmp_path / "explained.csv"

        exceptions_df.to_csv(exc_path, index=False)
        sample_internal.to_csv(int_path, index=False)
        sample_bank.to_csv(bank_path, index=False)

        result = explain_exceptions(exc_path, int_path, bank_path, out_path)

        # Verify output
        assert len(result) == 2
        assert "llm_explanation" in result.columns
        assert "llm_confidence" in result.columns
        assert all(result["llm_explanation"].str.len() > 0)
        assert all(result["llm_confidence"].isin(["high", "medium", "low"]))

        # Verify CSV was written
        assert out_path.exists()
        written = pd.read_csv(out_path)
        assert len(written) == 2
        assert "llm_explanation" in written.columns
