from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


class LLMExplainer:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def explain_exception(self, exception_row: dict) -> str:
        if not self.api_key:
            return self._fallback_explanation(exception_row)

        return self._fallback_explanation(exception_row)

    def _fallback_explanation(self, exception_row: dict) -> str:
        invoice_id = exception_row.get("invoice_id", "unknown")
        issue = exception_row.get("issue", "unknown issue")
        internal_amount = exception_row.get("internal_amount")
        bank_amount = exception_row.get("bank_amount")
        delta = exception_row.get("amount_delta")

        parts = [f"Invoice {invoice_id} has a {issue.lower()}."]
        if internal_amount is not None and bank_amount is not None:
            parts.append(f"Internal amount was {internal_amount} and bank amount was {bank_amount}.")
        if delta is not None:
            parts.append(f"Difference recorded: {delta}.")
        parts.append("This likely requires manual review or a settlement correction.")
        return " ".join(parts)
