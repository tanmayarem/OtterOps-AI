"""Tests for src/qa_agent.py — read-only Q&A layer over reconciliation output.

All tests use mocked LLM responses; no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.qa_agent import (
    _extract_ids,
    _find_relevant_records,
    _load_outputs,
    _mock_answer,
    answer_question,
)


# ---------------------------------------------------------------------------
# Test fixtures — lightweight DataFrames matching real schema
# ---------------------------------------------------------------------------

MATCHED_DATA = pd.DataFrame([
    {
        "internal_payment_id": "PAY0002",
        "bank_settlement_id": "SET0002",
        "order_ref": "ORD10235",
        "internal_amount": 19503.89,
        "bank_amount": 19503.89,
        "internal_date": "2026-08-22",
        "bank_date": "2026-08-22",
        "amount_delta": 0.0,
        "date_lag_days": 0,
        "match_type": "exact",
        "tolerance_used": "none (exact)",
        "internal_status": "captured",
        "bank_order_ref": "",
    },
    {
        "internal_payment_id": "PAY0010",
        "bank_settlement_id": "SET0011",
        "order_ref": "ORD10243",
        "internal_amount": 5000.00,
        "bank_amount": 4900.00,
        "internal_date": "2026-08-25",
        "bank_date": "2026-08-25",
        "amount_delta": 100.00,
        "date_lag_days": 0,
        "match_type": "fuzzy_fee",
        "tolerance_used": "amount within 2.5% (fee/lag)",
        "internal_status": "captured",
        "bank_order_ref": "",
    },
])

EXCEPTIONS_DATA = pd.DataFrame([
    {
        "record_type": "internal",
        "payment_id": "PAY0021",
        "settlement_id": "",
        "order_ref": "ORD10254",
        "amount": 36343.60,
        "reason_code": "MISSING_FROM_SETTLEMENT",
        "notes": "No bank settlement found for this order_ref",
        "llm_explanation": "Payment was recorded internally but no bank settlement found.",
        "llm_confidence": "medium",
    },
    {
        "record_type": "internal",
        "payment_id": "PAY0034",
        "settlement_id": "",
        "order_ref": "ORD10267",
        "amount": 5458.31,
        "reason_code": "MISSING_FROM_SETTLEMENT",
        "notes": "No bank settlement found for this order_ref",
        "llm_explanation": "Payment was recorded internally but no bank settlement found.",
        "llm_confidence": "medium",
    },
])


# ---------------------------------------------------------------------------
# Test 1: ID extraction from question text
# ---------------------------------------------------------------------------

class TestExtractIds:
    def test_extracts_order_ref(self):
        ids = _extract_ids("Why wasn't ORD10254 matched?", {})
        assert ids == ["ORD10254"]

    def test_extracts_payment_id(self):
        ids = _extract_ids("Show me PAY0021 details", {})
        assert ids == ["PAY0021"]

    def test_extracts_multiple_ids(self):
        ids = _extract_ids("Compare ORD10235 and PAY0021", {})
        assert "ORD10235" in ids
        assert "PAY0021" in ids

    def test_no_ids_found(self):
        ids = _extract_ids("How many exceptions were found?", {})
        assert ids == []

    def test_case_insensitive(self):
        ids = _extract_ids("tell me about ord10254", {})
        assert ids == ["ORD10254"]


# ---------------------------------------------------------------------------
# Test 2: Question with matching order_ref pulls the right row
# ---------------------------------------------------------------------------

class TestFindRelevantRecords:
    def test_specific_id_pulls_matching_row(self):
        data = {
            "matched": MATCHED_DATA,
            "exceptions": EXCEPTIONS_DATA,
            "summary": {},
        }
        context = _find_relevant_records("Why wasn't ORD10254 matched?", data)
        assert "ORD10254" in context
        assert "MISSING_FROM_SETTLEMENT" in context
        assert "PAY0021" in context

    def test_no_id_returns_summary(self):
        data = {
            "matched": MATCHED_DATA,
            "exceptions": EXCEPTIONS_DATA,
            "summary": {"total_matched": 2, "total_exceptions": 2, "match_types": {"exact": 1}},
        }
        context = _find_relevant_records("How many exceptions?", data)
        assert "RECONCILIATION SUMMARY" in context
        assert "total_matched" in context.lower() or "Total matched" in context

    def test_unknown_id_returns_not_found(self):
        data = {
            "matched": MATCHED_DATA,
            "exceptions": EXCEPTIONS_DATA,
            "summary": {},
        }
        context = _find_relevant_records("What about ORD99999?", data)
        assert "No records found" in context


# ---------------------------------------------------------------------------
# Test 3: Mock fallback works without API key
# ---------------------------------------------------------------------------

class TestMockAnswer:
    def test_mock_with_id(self):
        answer = _mock_answer("Why wasn't ORD10254 matched?", "context")
        assert "ORD10254" in answer

    def test_mock_general_question(self):
        answer = _mock_answer("How many exceptions?", "context")
        assert len(answer) > 0
        assert "reconciliation" in answer.lower()


# ---------------------------------------------------------------------------
# Test 4: answer_question with mocked LLM
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    @patch("src.qa_agent._call_llm")
    def test_llm_called_with_context(self, mock_call_llm, tmp_path):
        mock_call_llm.return_value = (
            "ORD10254 was not matched because no bank settlement was found.\n"
            "CONFIDENCE: high"
        )

        # Write real data to temp files so load_outputs doesn't return empty
        matched_csv = tmp_path / "matched_pairs.csv"
        exceptions_csv = tmp_path / "exceptions.csv"
        explained_csv = tmp_path / "exceptions_explained.csv"
        MATCHED_DATA.to_csv(matched_csv, index=False)
        EXCEPTIONS_DATA.to_csv(exceptions_csv, index=False)
        EXCEPTIONS_DATA.to_csv(explained_csv, index=False)

        answer, context_ids = answer_question(
            "Why wasn't ORD10254 matched?",
            matched_path=str(matched_csv),
            explained_path=str(explained_csv),
            exceptions_path=str(exceptions_csv),
            api_key="fake-key",
            model="test-model",
        )

        assert mock_call_llm.called
        assert "ORD10254" in answer

    @patch("src.qa_agent._call_llm")
    def test_fallback_on_api_error(self, mock_call_llm):
        mock_call_llm.side_effect = ConnectionError("API unavailable")

        answer, context_ids = answer_question(
            "How many exceptions?",
            matched_path=None,
            explained_path=None,
            exceptions_path=None,
            api_key="fake-key",
            model="test-model",
        )

        # Should fall back to mock answer, not crash
        assert len(answer) > 0
        assert "reconciliation" in answer.lower()

    def test_no_api_key_uses_mock(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove GROQ_API_KEY if present
            import os
            os.environ.pop("GROQ_API_KEY", None)

            answer, context_ids = answer_question(
                "How many exceptions?",
                matched_path=None,
                explained_path=None,
                exceptions_path=None,
                api_key=None,
                model=None,
            )

            # Should get a mock answer (no crash)
            assert len(answer) > 0

    def test_no_files_returns_empty_message(self):
        answer, context_ids = answer_question(
            "What about ORD10234?",
            matched_path="nonexistent/matched.csv",
            explained_path="nonexistent/explained.csv",
            exceptions_path="nonexistent/exceptions.csv",
        )
        assert "No reconciliation data" in answer


# ---------------------------------------------------------------------------
# Test 5: Module never modifies output files
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    def test_answer_question_does_not_write_files(self, tmp_path):
        """Verify that answering a question never modifies any file."""
        # Create temp output files
        matched = tmp_path / "matched_pairs.csv"
        exceptions = tmp_path / "exceptions.csv"
        explained = tmp_path / "exceptions_explained.csv"

        MATCHED_DATA.to_csv(matched, index=False)
        EXCEPTIONS_DATA.to_csv(exceptions, index=False)
        EXCEPTIONS_DATA.to_csv(explained, index=False)

        # Record modification times
        matched_mtime = matched.stat().st_mtime
        explained_mtime = explained.stat().st_mtime

        # Answer a question
        answer_question(
            "Why wasn't ORD10254 matched?",
            matched_path=str(matched),
            explained_path=str(explained),
            exceptions_path=str(exceptions),
            api_key=None,
            model=None,
        )

        # Verify files were NOT modified
        assert matched.stat().st_mtime == matched_mtime
        assert explained.stat().st_mtime == explained_mtime
