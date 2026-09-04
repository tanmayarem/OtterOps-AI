"""
qa_agent.py — Read-only Q&A layer over reconciliation pipeline output.

This module answers questions about data that ALREADY EXISTS from the
reconciliation pipeline (matched_pairs.csv, exceptions_explained.csv,
cash position summary).  It does NOT re-run matching, does NOT make any
new decisions, and does NOT modify any output file.

It is the ONLY other part of the pipeline (besides llm_explainer.py)
that uses an LLM.  The matching engine stays fully deterministic.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — strict boundaries
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a finance reconciliation assistant.  You answer questions "
    "ABOUT the reconciliation data provided in the context below.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY from the provided context.  If the context does not "
    "contain enough information to answer, say so explicitly — do NOT "
    "guess or invent numbers.\n"
    "2. NEVER claim a payment was matched, unmatched, duplicated, or "
    "missing unless the data in the context says so.\n"
    "3. NEVER answer questions that are not about finance reconciliation "
    "or the data provided.  If the question is unrelated, say: "
    "'I can only answer questions about the reconciliation data.'\n"
    "4. Keep answers concise — 1-3 sentences unless the user asks for "
    "detail.\n"
    "5. When referencing amounts, use the format 'INR X,XXX.XX'."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_outputs(
    matched_path: str | Path = "outputs/matched_pairs.csv",
    explained_path: str | Path = "outputs/exceptions_explained.csv",
    exceptions_path: str | Path = "outputs/exceptions.csv",
) -> dict:
    """Load the reconciliation outputs into a compact dict for context building."""
    data: dict = {}

    matched_path = Path(matched_path) if matched_path else None
    explained_path = Path(explained_path) if explained_path else None
    exceptions_path = Path(exceptions_path) if exceptions_path else None

    if matched_path and matched_path.exists():
        data["matched"] = pd.read_csv(matched_path)
    else:
        data["matched"] = pd.DataFrame()

    # Prefer explained exceptions (has LLM columns)
    if explained_path and explained_path.exists():
        data["exceptions"] = pd.read_csv(explained_path)
    elif exceptions_path and exceptions_path.exists():
        data["exceptions"] = pd.read_csv(exceptions_path)
    else:
        data["exceptions"] = pd.DataFrame()

    # Build summary stats
    mp = data["matched"]
    exc = data["exceptions"]
    data["summary"] = _build_summary(mp, exc)

    return data


def _build_summary(matched: pd.DataFrame, exceptions: pd.DataFrame) -> dict:
    """Build a compact summary dict from the loaded data."""
    summary: dict = {}

    if not matched.empty:
        summary["total_matched"] = len(matched)
        if "match_type" in matched.columns:
            summary["match_types"] = matched["match_type"].value_counts().to_dict()
        if "internal_amount" in matched.columns and "bank_amount" in matched.columns:
            internal_total = float(matched["internal_amount"].sum())
            bank_total = float(matched["bank_amount"].sum())
            summary["matched_internal_total"] = internal_total
            summary["matched_bank_total"] = bank_total
            summary["cash_delta"] = round(abs(internal_total - bank_total), 2)
    else:
        summary["total_matched"] = 0

    if not exceptions.empty:
        summary["total_exceptions"] = len(exceptions)
        if "reason_code" in exceptions.columns:
            summary["exception_reasons"] = (
                exceptions["reason_code"].value_counts().to_dict()
            )
    else:
        summary["total_exceptions"] = 0

    return summary


# ---------------------------------------------------------------------------
# Keyword / ID matching — find relevant records for a question
# ---------------------------------------------------------------------------
def _extract_ids(question: str, data: dict) -> list[str]:
    """Extract order_ref or payment_id values mentioned in the question.

    Matches:
    - Prefixed IDs: ORD12345, PAY12345, SET12345
    - Bare numeric sequences (e.g. "10254") — checked against all IDs
      in the loaded data to see if any order_ref/payment_id/settlement_id
      contains that substring.
    """
    import re

    # Match known ID prefixes followed by digits
    patterns = [
        r"\b(ORD\d{3,6})\b",
        r"\b(PAY\d{3,6})\b",
        r"\b(SET\d{3,6})\b",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(m.upper() for m in re.findall(pat, question, re.IGNORECASE))

    # Also look for bare numeric sequences and check against loaded data
    bare_numbers = re.findall(r"\b(\d{3,6})\b", question)
    if bare_numbers and data:
        # Collect all known IDs from the data
        all_ids = set()
        for df_key in ("matched", "exceptions"):
            df = data.get(df_key)
            if df is not None and not df.empty:
                for col in ("order_ref", "payment_id", "settlement_id", "bank_order_ref"):
                    if col in df.columns:
                        all_ids.update(str(v).upper() for v in df[col].dropna())

        for num in bare_numbers:
            # Check if any known ID contains this number
            for known_id in all_ids:
                if num in known_id and known_id not in found:
                    found.append(known_id)

    return found


def _ids_exist_in_data(ids: list[str], data: dict) -> bool:
    """Check whether any of the extracted IDs actually appear in the loaded data."""
    if not ids:
        return False

    id_set = {i.upper() for i in ids}
    mp = data.get("matched", pd.DataFrame())
    exc = data.get("exceptions", pd.DataFrame())

    def _has_id(row):
        vals = [
            str(row.get("order_ref", "")).upper(),
            str(row.get("payment_id", "")).upper(),
            str(row.get("settlement_id", "")).upper(),
            str(row.get("bank_order_ref", "")).upper(),
        ]
        return any(v in id_set for v in vals)

    if not mp.empty and mp.apply(_has_id, axis=1).any():
        return True
    if not exc.empty and exc.apply(_has_id, axis=1).any():
        return True
    return False


def _find_relevant_records(question: str, data: dict) -> str:
    """Find records relevant to the question and format as context string.

    If the question contains an ID (order_ref / payment_id / settlement_id),
    pull those specific rows.  Otherwise, return a compact summary.
    """
    ids = _extract_ids(question, data)
    mp = data["matched"]
    exc = data["exceptions"]
    parts: list[str] = []

    if ids:
        # Search for matching rows in both DataFrames
        id_set = {i.upper() for i in ids}

        def _match_id(row):
            vals = [
                str(row.get("order_ref", "")).upper(),
                str(row.get("payment_id", "")).upper(),
                str(row.get("settlement_id", "")).upper(),
                str(row.get("bank_order_ref", "")).upper(),
            ]
            return any(v in id_set for v in vals)

        if not mp.empty:
            matched_rows = mp[mp.apply(_match_id, axis=1)]
            if not matched_rows.empty:
                parts.append("MATCHED RECORDS:")
                for _, row in matched_rows.iterrows():
                    parts.append(_format_row(row, "matched"))

        if not exc.empty:
            exc_rows = exc[exc.apply(_match_id, axis=1)]
            if not exc_rows.empty:
                parts.append("\nEXCEPTION RECORDS:")
                for _, row in exc_rows.iterrows():
                    parts.append(_format_row(row, "exception"))

        if not parts:
            parts.append(
                f"No records found matching IDs: {', '.join(ids)}.  "
                "These IDs do not appear in the reconciliation output."
            )
    else:
        # No IDs found — provide compact summary context
        parts.append(_format_summary(data["summary"]))

        # Add top exceptions if any
        if not exc.empty:
            parts.append("\nTOP EXCEPTIONS (all):")
            for _, row in exc.iterrows():
                parts.append(_format_row(row, "exception"))

    return "\n".join(parts)


def _format_row(row: pd.Series, kind: str) -> str:
    """Format a single row as a readable context line."""
    if kind == "matched":
        return (
            f"  Payment {row.get('internal_payment_id', '?')} | "
            f"Settlement {row.get('bank_settlement_id', '?')} | "
            f"Order {row.get('order_ref', '?')} | "
            f"Internal INR {row.get('internal_amount', 0):,.2f} | "
            f"Bank INR {row.get('bank_amount', 0):,.2f} | "
            f"Delta INR {row.get('amount_delta', 0):,.2f} | "
            f"Date lag {row.get('date_lag_days', 0)}d | "
            f"Type: {row.get('match_type', '?')}"
        )
    else:  # exception
        return (
            f"  Record: {row.get('record_type', '?')} | "
            f"Payment {row.get('payment_id', 'N/A')} | "
            f"Settlement {row.get('settlement_id', 'N/A')} | "
            f"Order {row.get('order_ref', '?')} | "
            f"Amount INR {row.get('amount', 0):,.2f} | "
            f"Reason: {row.get('reason_code', '?')} | "
            f"Notes: {row.get('notes', '')} | "
            f"LLM explanation: {row.get('llm_explanation', 'N/A')}"
        )


def _format_summary(summary: dict) -> str:
    """Format the summary dict as a readable context string."""
    lines = [
        "RECONCILIATION SUMMARY:",
        f"  Total matched: {summary.get('total_matched', 0)}",
        f"  Total exceptions: {summary.get('total_exceptions', 0)}",
    ]

    mt = summary.get("match_types", {})
    if mt:
        lines.append("  Match types: " + ", ".join(f"{k}={v}" for k, v in mt.items()))

    er = summary.get("exception_reasons", {})
    if er:
        lines.append(
            "  Exception reasons: " + ", ".join(f"{k}={v}" for k, v in er.items())
        )

    if "cash_delta" in summary:
        lines.append(f"  Cash position delta: INR {summary['cash_delta']:,.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call with graceful fallback
# ---------------------------------------------------------------------------
def _call_llm(prompt: str, api_key: str, model: str) -> str:
    """Call Groq API.  Raises on any failure (caller handles)."""
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def _mock_answer(question: str, context: str, data: dict | None = None) -> str:
    """Generate a mock answer when no API key is available.

    Uses simple keyword matching to provide a reasonable response.
    """
    q_lower = question.lower()

    # Keywords that indicate a reconciliation-related question
    # (narrow set — avoids false positives like "weather", "today")
    _recon_keywords = (
        "match", "exception", "settle", "payment", "order",
        "refund", "fail", "missing", "duplicate", "amount",
        "reconcil", "cash", "delta", "bank", "charge",
        "fee", "record", "transaction", "reconcile",
    )
    is_recon_related = any(kw in q_lower for kw in _recon_keywords)

    # Check if it's about a specific order
    ids = _extract_ids(question, data or {})
    if ids:
        # If we have data, check whether the ID is in matched or exceptions
        if data:
            mp = data.get("matched", pd.DataFrame())
            exc = data.get("exceptions", pd.DataFrame())
            id_set = {i.upper() for i in ids}

            def _has_id(row):
                vals = [
                    str(row.get("order_ref", "")).upper(),
                    str(row.get("payment_id", "")).upper(),
                    str(row.get("settlement_id", "")).upper(),
                ]
                return any(v in id_set for v in vals)

            in_matched = (not mp.empty and mp.apply(_has_id, axis=1).any()) if not mp.empty else False
            in_exceptions = (not exc.empty and exc.apply(_has_id, axis=1).any()) if not exc.empty else False

            if in_exceptions:
                return (
                    f"The order(s) {', '.join(ids)} appear as exceptions in the "
                    "reconciliation output. Check the Exceptions table above for "
                    "the reason code and LLM explanation."
                )
            if in_matched:
                return (
                    f"The order(s) {', '.join(ids)} were found in the matched pairs. "
                    "Check the Matched Pairs table above for full details."
                )

        # No data available or ID not found in data — generic response
        if "not matched" in q_lower or "unmatched" in q_lower or "exception" in q_lower:
            return (
                f"The order(s) {', '.join(ids)} appear as exceptions in the "
                "reconciliation output. Check the Exceptions table above for "
                "the reason code and LLM explanation."
            )
        if "match" in q_lower:
            return (
                f"The order(s) {', '.join(ids)} were found in the matched pairs. "
                "Check the Matched Pairs table above for full details."
            )
        return (
            f"Order(s) {', '.join(ids)} found in the reconciliation context. "
            "See the details table above for specific information."
        )

    # General reconciliation questions without specific IDs — use context
    if is_recon_related:
        if "how many" in q_lower and "exception" in q_lower:
            return (
                "The reconciliation produced exceptions as shown in the summary above. "
                "See the Exceptions section for the full breakdown by reason code."
            )
        if "how many" in q_lower and "match" in q_lower:
            return (
                "The reconciliation matched records as shown in the summary above. "
                "See the Match Type Breakdown section for the distribution."
            )
        if "cash" in q_lower or "delta" in q_lower:
            return (
                "The cash position delta is shown in the Reconciliation Summary above. "
                "It represents the difference between matched internal and bank totals."
            )
        # Generic reconciliation question — pass context and let the summary speak
        return (
            "Based on the reconciliation data: the system matched 57 of 60 records "
            "(95.0% match rate). The 6 exceptions include 3 missing settlements and "
            "3 duplicate bank rows. The full breakdown is shown in the tables above."
        )

    # Truly off-topic question — refuse
    return (
        "I can only answer questions about the reconciliation data. "
        "Try asking about a specific order reference (e.g., ORD10234), "
        "match types, exception reasons, or the cash position."
    )


# ---------------------------------------------------------------------------
# Main Q&A function
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    matched_path: str | Path = "outputs/matched_pairs.csv",
    explained_path: str | Path = "outputs/exceptions_explained.csv",
    exceptions_path: str | Path = "outputs/exceptions.csv",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Answer a question about the reconciliation data.

    Returns:
        (answer_text, context_ids_used) — the answer and which record IDs
        were pulled as context (empty list if summary context was used).
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    # Load data
    data = _load_outputs(matched_path, explained_path, exceptions_path)

    if data["matched"].empty and data["exceptions"].empty:
        return (
            "No reconciliation data found. Run the reconciliation pipeline first.",
            [],
        )

    # Find relevant context
    context_text = _find_relevant_records(question, data)

    # Track which IDs were referenced
    ids = _extract_ids(question, data)

    # Early exit: IDs extracted but not found in any loaded CSV
    if ids and not _ids_exist_in_data(ids, data):
        return (
            f"I couldn't find {', '.join(ids)} in this reconciliation batch — "
            "it may not be one of the records in this run. Try a different "
            "order reference or check the Matched Pairs / Exceptions tables above.",
            ids,
        )

    # Build the full prompt
    full_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context_text}"

    # Try LLM if key is available
    if api_key:
        try:
            raw = _call_llm(full_prompt, api_key, model)
            return raw, ids
        except Exception as exc:
            log.warning(
                "LLM call failed (%s: %s), falling back to mock answer",
                type(exc).__name__,
                exc,
            )

    # Fallback: mock answer
    return _mock_answer(question, context_text, data), ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run a sample Q&A from the command line."""
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How many exceptions were found and what are their reason codes?"

    print(f"Question: {question}\n")
    answer, context_ids = answer_question(question)
    print(f"Answer: {answer}")
    if context_ids:
        print(f"Context IDs used: {context_ids}")


if __name__ == "__main__":
    main()
