"""
llm_explainer.py — LLM-based exception explanations for reconciliation output.

This module explains exceptions already classified by the deterministic
matcher.  It does not resolve, re-match, or override any matching decision.

It is the ONLY part of the pipeline that uses an LLM.  The matching
engine (src/matcher.py) stays fully deterministic and never calls this module.

If no GROQ_API_KEY is configured, the module falls back to a deterministic
mock explanation so the pipeline never crashes without an API key.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REASON_PROMPTS = {
    "MISSING_FROM_SETTLEMENT": (
        "This payment was recorded internally but has no matching bank "
        "settlement. Explain the likely cause in one sentence and state "
        "whether it requires manual review."
    ),
    "AMOUNT_MISMATCH_UNRESOLVED": (
        "This payment has a bank settlement but the amount differs beyond "
        "tolerance. State the most likely explanation (fee, refund, rounding, "
        "or data error) and your confidence level."
    ),
    "NO_MATCH_FOUND": (
        "Bank rows exist for this order_ref but none matched after both "
        "exact and fuzzy passes. Explain what likely went wrong and whether "
        "this needs manual investigation."
    ),
    "DUPLICATE_SETTLEMENT": (
        "This bank settlement row is a duplicate. Explain whether this is "
        "harmless (extra row to ignore) or indicates a real problem, and "
        "state your confidence."
    ),
    "MISSING_FROM_INTERNAL": (
        "This bank settlement has no corresponding internal payment record. "
        "Explain the likely cause and whether it requires investigation."
    ),
}

SYSTEM_PROMPT = (
    "You are a finance reconciliation assistant.  For each exception, "
    "provide:\n"
    "1. A plain-language explanation in ONE sentence.\n"
    "2. A confidence level: high, medium, or low.\n"
    "3. If you cannot form a plausible explanation, say 'requires manual "
    "review' and set confidence to low.\n\n"
    "Never force a confident-sounding guess when the data is insufficient. "
    "Format your response as:\n"
    "EXPLANATION: <one sentence>\n"
    "CONFIDENCE: <high|medium|low>"
)

MOCK_RESPONSES = {
    "MISSING_FROM_SETTLEMENT": (
        "Payment was recorded internally but no bank settlement was found, "
        "suggesting a stuck or failed settlement that requires gateway "
        "investigation.",
        "medium",
    ),
    "AMOUNT_MISMATCH_UNRESOLVED": (
        "The bank amount differs from the internal amount beyond tolerance, "
        "likely due to an applied gateway fee or partial refund.",
        "medium",
    ),
    "NO_MATCH_FOUND": (
        "Bank rows exist for this order reference but none matched on amount "
        "or date, indicating a possible data entry error or timing mismatch.",
        "low",
    ),
    "DUPLICATE_SETTLEMENT": (
        "This is an extra bank settlement row for an order that was already "
        "matched; it should be reversed or flagged as a no-op duplicate.",
        "high",
    ),
    "MISSING_FROM_INTERNAL": (
        "A bank settlement exists with no corresponding internal payment "
        "record, which may indicate an unrecorded transaction or "
        "cross-reference error.",
        "medium",
    ),
}


# ---------------------------------------------------------------------------
# Context payload builder
# ---------------------------------------------------------------------------
def build_context(
    exception_row: dict,
    internal: pd.DataFrame,
    bank: pd.DataFrame,
) -> str:
    """Build a short context payload for a single exception.

    Includes the exception details plus the closest candidate bank/internal
    records (up to 3) so the LLM has enough signal to form an explanation.
    """
    record_type = exception_row.get("record_type", "unknown")
    order_ref = exception_row.get("order_ref", "")
    amount = exception_row.get("amount")
    reason_code = exception_row.get("reason_code", "unknown")
    payment_id = exception_row.get("payment_id", "")
    settlement_id = exception_row.get("settlement_id", "")
    notes = exception_row.get("notes", "")

    parts = [
        f"EXCEPTION DETAILS:",
        f"  Record type: {record_type}",
        f"  Order ref: {order_ref}",
        f"  Payment ID: {payment_id or 'N/A'}",
        f"  Settlement ID: {settlement_id or 'N/A'}",
        f"  Amount: INR {amount:,.2f}" if amount else "  Amount: N/A",
        f"  Reason code: {reason_code}",
        f"  Matcher notes: {notes}",
    ]

    # Find closest candidates from the opposite side
    if record_type == "internal" and pd.notna(amount):
        candidates = _find_closest_bank(order_ref, amount, bank, limit=3)
        if candidates:
            parts.append("")
            parts.append("CLOSEST BANK CANDIDATES:")
            for c in candidates:
                parts.append(
                    f"  - {c['settlement_id']} | {c['order_ref']} | "
                    f"INR {c['amount']:,.2f} | {c['settlement_date']} | "
                    f"mutation: {c.get('_mutation', '?')}"
                )
        else:
            parts.append("")
            parts.append("NO bank records found with similar order_ref or amount.")

    elif record_type == "bank" and pd.notna(amount):
        candidates = _find_closest_internal(order_ref, amount, internal, limit=3)
        if candidates:
            parts.append("")
            parts.append("CLOSEST INTERNAL CANDIDATES:")
            for c in candidates:
                parts.append(
                    f"  - {c['payment_id']} | {c['order_ref']} | "
                    f"INR {c['amount']:,.2f} | {c['date']} | "
                    f"status: {c.get('status', '?')}"
                )
        else:
            parts.append("")
            parts.append("NO internal records found with similar order_ref or amount.")

    return "\n".join(parts)


def _find_closest_bank(
    order_ref: str, amount: float, bank: pd.DataFrame, limit: int = 3
) -> list[dict]:
    """Find bank records closest to the given order_ref and amount."""
    if bank.empty:
        return []

    # Score: 0 = exact order_ref match, +1 per edit-distance-1, +100 otherwise
    scored = []
    for _, row in bank.iterrows():
        ref = row["order_ref"]
        if ref == order_ref:
            score = 0
        elif _edit_distance_1_safe(ref, order_ref):
            score = 1
        else:
            continue  # skip completely different refs
        amt_delta = abs(row["amount"] - amount)
        scored.append((score + amt_delta / max(amount, 1), row))

    scored.sort(key=lambda x: x[0])
    return [row.to_dict() for _, row in scored[:limit]]


def _find_closest_internal(
    order_ref: str, amount: float, internal: pd.DataFrame, limit: int = 3
) -> list[dict]:
    """Find internal records closest to the given order_ref and amount."""
    if internal.empty:
        return []

    scored = []
    for _, row in internal.iterrows():
        ref = row["order_ref"]
        if ref == order_ref:
            score = 0
        elif _edit_distance_1_safe(ref, order_ref):
            score = 1
        else:
            continue
        amt_delta = abs(row["amount"] - amount)
        scored.append((score + amt_delta / max(amount, 1), row))

    scored.sort(key=lambda x: x[0])
    return [row.to_dict() for _, row in scored[:limit]]


def _edit_distance_1_safe(a: str, b: str) -> bool:
    """Edit-distance-1 check (safe wrapper, imports from matcher if available)."""
    try:
        from src.matcher import _edit_distance_1
        return _edit_distance_1(a, b)
    except ImportError:
        # Inline fallback if matcher not importable
        la, lb = len(a), len(b)
        if la == lb:
            return sum(c1 != c2 for c1, c2 in zip(a, b)) == 1
        if abs(la - lb) == 1:
            longer, shorter = (a, b) if la > lb else (b, a)
            i = j = skipped = 0
            while i < len(longer) and j < len(shorter):
                if longer[i] == shorter[j]:
                    i += 1; j += 1
                elif not skipped:
                    skipped = 1; i += 1
                else:
                    return False
            return True
        return False


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
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _parse_llm_response(raw: str) -> tuple[str, str]:
    """Parse the EXPLANATION:/CONFIDENCE: format from the LLM response."""
    explanation = ""
    confidence = "low"

    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().lower()

    if not explanation:
        # Fallback: use the whole response as explanation
        explanation = raw[:300]

    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return explanation, confidence


# ---------------------------------------------------------------------------
# Core: explain a single exception
# ---------------------------------------------------------------------------
def explain_single(
    exception_row: dict,
    internal: pd.DataFrame,
    bank: pd.DataFrame,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, str]:
    """Explain one exception. Returns (explanation, confidence).

    Uses LLM if api_key is set, otherwise falls back to mock responses.
    Never crashes — returns a safe fallback on any error.
    """
    reason_code = exception_row.get("reason_code", "unknown")
    api_key = api_key or os.getenv("GROQ_API_KEY")
    model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    # Build context payload
    context = build_context(exception_row, internal, bank)
    reason_hint = REASON_PROMPTS.get(reason_code, "Explain this reconciliation exception.")
    full_prompt = f"{context}\n\n{reason_hint}"

    # Try LLM if key is available
    if api_key:
        try:
            raw = _call_llm(full_prompt, api_key, model)
            return _parse_llm_response(raw)
        except Exception as exc:
            log.warning("LLM call failed (%s: %s), falling back to mock", type(exc).__name__, exc)

    # Fallback: deterministic mock response
    return MOCK_RESPONSES.get(reason_code, (
        f"Exception {reason_code} for order {exception_row.get('order_ref', '?')} "
        f"requires manual review.",
        "low",
    ))


# ---------------------------------------------------------------------------
# Batch: explain all exceptions, write CSV
# ---------------------------------------------------------------------------
def explain_exceptions(
    exceptions_path: str | Path = "outputs/exceptions.csv",
    internal_path: str | Path = "data/internal_payments.csv",
    bank_path: str | Path = "data/bank_settlement.csv",
    output_path: str | Path = "outputs/exceptions_explained.csv",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Load exceptions, explain each one, write results to CSV.

    This function is designed to be called after matcher.py has produced
    exceptions.csv.  It NEVER modifies the original exceptions — only
    appends llm_explanation and llm_confidence columns.
    """
    exc_df = pd.read_csv(exceptions_path)
    internal = pd.read_csv(internal_path)
    bank = pd.read_csv(bank_path)

    # Coerce types for candidate search
    internal["amount"] = pd.to_numeric(internal["amount"], errors="coerce")
    internal["date"] = pd.to_datetime(internal["date"], errors="coerce")
    bank["amount"] = pd.to_numeric(bank["amount"], errors="coerce")
    bank["settlement_date"] = pd.to_datetime(bank["settlement_date"], errors="coerce")

    explanations = []
    confidences = []

    for _, row in exc_df.iterrows():
        exc_dict = row.to_dict()
        # Handle NaN values
        exc_dict = {k: (v if pd.notna(v) else "") for k, v in exc_dict.items()}

        explanation, confidence = explain_single(
            exc_dict, internal, bank, api_key=api_key, model=model,
        )
        explanations.append(explanation)
        confidences.append(confidence)

    exc_df["llm_explanation"] = explanations
    exc_df["llm_confidence"] = confidences

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exc_df.to_csv(out, index=False)

    log.info("Wrote %d explained exceptions to %s", len(exc_df), out)
    return exc_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run exception explanation from the command line."""
    root = Path(__file__).resolve().parent.parent
    exceptions_path = root / "outputs" / "exceptions.csv"
    internal_path = root / "data" / "internal_payments.csv"
    bank_path = root / "data" / "bank_settlement.csv"
    output_path = root / "outputs" / "exceptions_explained.csv"

    if not exceptions_path.exists():
        print("ERROR: outputs/exceptions.csv not found. Run src.matcher first.")
        sys.exit(1)

    result = explain_exceptions(exceptions_path, internal_path, bank_path, output_path)

    print()
    print("=" * 65)
    print("  EXCEPTION EXPLANATIONS")
    print("=" * 65)
    for _, row in result.iterrows():
        print(f"\n  [{row['reason_code']}] {row['order_ref']} | INR {row['amount']:,.2f}")
        print(f"  -> {row['llm_explanation']}")
        print(f"  Confidence: {row['llm_confidence']}")
    print()
    print("=" * 65)
    print(f"  Wrote {len(result)} explanations to {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
