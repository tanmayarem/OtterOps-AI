"""
llm_explainer.py — LLM-based exception explanations (Day 2).

Uses Groq API (free tier) to generate human-readable explanations for
each reconciliation exception. Falls back to rule-based explanations
when no API key is configured.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMExplainer:
    """Generate human-readable explanations for reconciliation exceptions."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def explain_exception(self, exception_row: dict) -> str:
        """Explain a single exception. Uses LLM if available, else rule-based."""
        if not self.api_key:
            return self._fallback_explanation(exception_row)

        try:
            return self._llm_explanation(exception_row)
        except Exception:
            return self._fallback_explanation(exception_row)

    def _llm_explanation(self, exception_row: dict) -> str:
        """Call Groq API to explain the exception."""
        from groq import Groq

        client = Groq(api_key=self.api_key)

        record_type = exception_row.get("record_type", "unknown")
        payment_id = exception_row.get("payment_id", "")
        settlement_id = exception_row.get("settlement_id", "")
        order_ref = exception_row.get("order_ref", "unknown")
        amount = exception_row.get("amount")
        reason_code = exception_row.get("reason_code", "unknown")
        notes = exception_row.get("notes", "")

        prompt = (
            f"You are a finance reconciliation assistant. "
            f"Explain this exception in 1-2 plain-English sentences "
            f"that a finance operations team member would understand.\n\n"
            f"Record type: {record_type}\n"
            f"Payment ID: {payment_id}\n"
            f"Settlement ID: {settlement_id}\n"
            f"Order ref: {order_ref}\n"
            f"Amount: INR {amount:,.2f}\n" if amount else ""
            f"Reason code: {reason_code}\n"
            f"System notes: {notes}\n\n"
            f"What happened and what should the ops team do?"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _fallback_explanation(self, exception_row: dict) -> str:
        """Rule-based explanation when no LLM is available."""
        record_type = exception_row.get("record_type", "unknown")
        order_ref = exception_row.get("order_ref", "unknown")
        reason_code = exception_row.get("reason_code", "unknown")
        amount = exception_row.get("amount")
        notes = exception_row.get("notes", "")

        if reason_code == "MISSING_FROM_SETTLEMENT":
            return (
                f"Payment {order_ref} (INR {amount:,.2f}) was recorded internally "
                f"but has no matching bank settlement. This likely means the "
                f"settlement is stuck or failed. Check with the payment gateway."
            )
        if reason_code == "DUPLICATE_SETTLEMENT":
            return (
                f"Bank settlement for {order_ref} (INR {amount:,.2f}) appears "
                f"multiple times. The primary row was matched; this extra row "
                f"should be reversed or flagged for investigation."
            )
        if reason_code == "AMOUNT_MISMATCH_UNRESOLVED":
            return (
                f"Order {order_ref} has a bank settlement but the amount "
                f"(INR {amount:,.2f}) doesn't match within tolerance. "
                f"Manual review needed to determine if this is a fee, "
                f"partial refund, or data error."
            )
        if reason_code == "NO_MATCH_FOUND":
            return (
                f"Bank rows exist for {order_ref} but none matched after "
                f"both exact and fuzzy passes. Possible data entry error "
                f"or timing mismatch."
            )
        if reason_code == "MISSING_FROM_INTERNAL":
            return (
                f"Bank settlement {exception_row.get('settlement_id', '')} "
                f"for {order_ref} (INR {amount:,.2f}) has no corresponding "
                f"internal record. May be an unrecorded payment or "
                f"cross-reference error."
            )
        return f"Exception for {order_ref}: {notes or reason_code}"
