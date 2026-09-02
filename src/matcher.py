"""
matcher.py — Deterministic payment-settlement reconciliation engine.

Two-pass matching:
  Pass 1  exact   — order_ref + amount + date all identical
  Pass 2  fuzzy   — fee/lag tolerance OR Levenshtein edit-distance-1 typo

No LLM calls. Fully reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GATEWAY_FEE_RATE = 0.02          # 2 % gateway fee
FEE_AMOUNT_TOLERANCE = 0.025     # 2.5 % of internal amount
DATE_LAG_TOLERANCE_DAYS = 2      # Pass-2 fee/lag window
TYPO_DATE_TOLERANCE_DAYS = 1     # Pass-2 typo window (tighter)
ROUNDING_TOLERANCE = 0.50        # paise-level rounding bound (INR)

# Exception reason codes
RC_NO_MATCH_FOUND = "NO_MATCH_FOUND"
RC_DUPLICATE_SETTLEMENT = "DUPLICATE_SETTLEMENT"
RC_AMOUNT_MISMATCH = "AMOUNT_MISMATCH_UNRESOLVED"
RC_MISSING_FROM_SETTLEMENT = "MISSING_FROM_SETTLEMENT"
RC_MISSING_FROM_INTERNAL = "MISSING_FROM_INTERNAL"


# ---------------------------------------------------------------------------
# Levenshtein edit-distance (lightweight, no deps)
# ---------------------------------------------------------------------------
def _edit_distance_1(a: str, b: str) -> bool:
    """Return True if *a* and *b* differ by exactly one edit
    (insertion, deletion, or substitution).  O(|a|+|b|)."""
    la, lb = len(a), len(b)
    if la == lb:
        # Substitution: exactly one position differs
        return sum(c1 != c2 for c1, c2 in zip(a, b)) == 1
    if la - lb == 1:
        # a is one longer → check if deleting one char from a yields b
        return _is_one_deletion(a, b)
    if lb - la == 1:
        return _is_one_deletion(b, a)
    return False


def _is_one_deletion(longer: str, shorter: str) -> bool:
    """True if removing exactly one character from *longer* gives *shorter*."""
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            i += 1          # skip one char in the longer string
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    internal_path: str | Path = "data/internal_payments.csv",
    bank_path: str | Path = "data/bank_settlement.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and coerce types for both CSVs."""
    internal = pd.read_csv(internal_path)
    bank = pd.read_csv(bank_path)

    # Coerce types
    internal["amount"] = pd.to_numeric(internal["amount"], errors="coerce")
    internal["date"] = pd.to_datetime(internal["date"], errors="coerce")
    bank["amount"] = pd.to_numeric(bank["amount"], errors="coerce")
    bank["settlement_date"] = pd.to_datetime(bank["settlement_date"], errors="coerce")

    return internal, bank


# ---------------------------------------------------------------------------
# Pass 1 — exact match
# ---------------------------------------------------------------------------
def exact_match(
    internal: pd.DataFrame,
    bank: pd.DataFrame,
    amount_tolerance: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match rows where order_ref, amount, and date are all identical.

    Returns (matched_df, remaining_internal, consumed_bank).
    """
    matched_rows = []
    consumed_bank_idx: set[int] = set()

    for i_idx, irow in internal.iterrows():
        for b_idx, brow in bank.iterrows():
            if b_idx in consumed_bank_idx:
                continue
            if irow["order_ref"] != brow["order_ref"]:
                continue
            if abs(irow["amount"] - brow["amount"]) > amount_tolerance:
                continue
            if irow["date"] != brow["settlement_date"]:
                continue

            # Exact match found
            date_lag = (brow["settlement_date"] - irow["date"]).days
            matched_rows.append({
                "internal_payment_id": irow["payment_id"],
                "bank_settlement_id": brow.get("settlement_id", ""),
                "order_ref": irow["order_ref"],
                "internal_amount": irow["amount"],
                "bank_amount": brow["amount"],
                "internal_date": irow["date"].strftime("%Y-%m-%d"),
                "bank_date": brow["settlement_date"].strftime("%Y-%m-%d"),
                "amount_delta": round(abs(irow["amount"] - brow["amount"]), 2),
                "date_lag_days": date_lag,
                "match_type": "exact",
                "tolerance_used": "none (exact)",
                "internal_status": irow.get("status", ""),
            })
            consumed_bank_idx.add(b_idx)
            break  # first match wins

    matched_df = pd.DataFrame(matched_rows)
    remaining_internal = internal.drop(
        index=[i for i in internal.index if i in matched_df.get("_i_idx", [])]
    ) if not matched_df.empty else internal.copy()

    # More precise: track which internal indices were matched
    matched_i_indices = set()
    for mr in matched_rows:
        for i_idx, irow in internal.iterrows():
            if irow["payment_id"] == mr["internal_payment_id"]:
                matched_i_indices.add(i_idx)
                break

    remaining_internal = internal.drop(index=list(matched_i_indices))
    consumed_bank = bank.loc[list(consumed_bank_idx)]

    return matched_df, remaining_internal, consumed_bank


# ---------------------------------------------------------------------------
# Pass 2 — fuzzy fallback
# ---------------------------------------------------------------------------
def fuzzy_match(
    internal: pd.DataFrame,
    bank: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fuzzy matching on remaining unmatched records.

    Strategy A (fee/lag):  order_ref exact  + amount within 2.5%  + date ≤ 2 days
    Strategy B (typo):     order_ref edit-dist-1  + amount exact  + date ≤ 1 day

    Returns (matched_df, unmatched_internal, unmatched_bank).
    """
    matched_rows = []
    consumed_bank_idx: set[int] = set()
    consumed_internal_idx: set[int] = set()

    for i_idx, irow in internal.iterrows():
        i_amount = irow["amount"]
        i_date = irow["date"]
        i_ref = irow["order_ref"]

        best = None  # (score, match_dict) — lower score = better

        for b_idx, brow in bank.iterrows():
            if b_idx in consumed_bank_idx:
                continue

            b_amount = brow["amount"]
            b_date = brow["settlement_date"]
            b_ref = brow["order_ref"]

            # --- Strategy A: fee/lag adjusted (order_ref exact) ---
            if i_ref == b_ref:
                amount_delta = abs(i_amount - b_amount)
                pct_diff = amount_delta / i_amount if i_amount > 0 else float("inf")
                date_lag = (b_date - i_date).days

                # Check fee tolerance (2.5%)
                fee_ok = pct_diff <= FEE_AMOUNT_TOLERANCE
                # Check date lag (≤ 2 days)
                lag_ok = 0 <= date_lag <= DATE_LAG_TOLERANCE_DAYS

                if fee_ok and lag_ok:
                    # Determine which tolerance triggered
                    triggers = []
                    if pct_diff > 0.001:  # > 0.1% means some amount diff
                        triggers.append("fee")
                    if date_lag > 0:
                        triggers.append(f"lag_T+{date_lag}")
                    tag = "fuzzy_" + "_".join(triggers) if triggers else "fuzzy_fee"

                    score = pct_diff * 10 + date_lag  # lower = better
                    if best is None or score < best[0]:
                        best = (score, {
                            "match_type": tag,
                            "tolerance_used": f"amount_pct={pct_diff:.4f}, date_lag={date_lag}d",
                            "bank_idx": b_idx,
                            "amount_delta": round(amount_delta, 2),
                            "date_lag_days": date_lag,
                        })

            # --- Strategy B: typo corrected (edit-distance 1) ---
            elif _edit_distance_1(i_ref, b_ref):
                amount_delta = abs(i_amount - b_amount)
                date_lag = (b_date - i_date).days

                amount_exact = amount_delta < 0.01
                lag_ok = 0 <= date_lag <= TYPO_DATE_TOLERANCE_DAYS

                if amount_exact and lag_ok:
                    score = 100 + date_lag  # always worse than strategy A
                    if best is None or score < best[0]:
                        best = (score, {
                            "match_type": "fuzzy_typo",
                            "tolerance_used": f"edit_dist=1({i_ref}->{b_ref}), date_lag={date_lag}d",
                            "bank_idx": b_idx,
                            "amount_delta": round(amount_delta, 2),
                            "date_lag_days": date_lag,
                        })

        if best is not None:
            _, info = best
            brow = bank.loc[info["bank_idx"]]
            matched_rows.append({
                "internal_payment_id": irow["payment_id"],
                "bank_settlement_id": brow.get("settlement_id", ""),
                "order_ref": irow["order_ref"],
                "bank_order_ref": brow["order_ref"],
                "internal_amount": i_amount,
                "bank_amount": brow["amount"],
                "internal_date": i_date.strftime("%Y-%m-%d"),
                "bank_date": brow["settlement_date"].strftime("%Y-%m-%d"),
                "amount_delta": info["amount_delta"],
                "date_lag_days": info["date_lag_days"],
                "match_type": info["match_type"],
                "tolerance_used": info["tolerance_used"],
                "internal_status": irow.get("status", ""),
            })
            consumed_bank_idx.add(info["bank_idx"])
            consumed_internal_idx.add(i_idx)

    matched_df = pd.DataFrame(matched_rows)
    unmatched_internal = internal.drop(index=list(consumed_internal_idx))
    unmatched_bank = bank.drop(index=list(consumed_bank_idx))

    return matched_df, unmatched_internal, unmatched_bank


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------
def classify_exceptions(
    unmatched_internal: pd.DataFrame,
    unmatched_bank: pd.DataFrame,
    all_internal: pd.DataFrame,
    all_bank: pd.DataFrame,
) -> pd.DataFrame:
    """Assign a reason code to every unmatched record.

    Only generates exceptions for records that were NOT consumed by
    either matching pass.  unmatched_bank should be the truly unconsumed
    bank rows (compute via ``_consumed_bank_indices``).

    Internal exceptions:
      - MISSING_FROM_SETTLEMENT  — no bank row with this order_ref at all
      - NO_MATCH_FOUND           — bank rows exist but none matched
      - AMOUNT_MISMATCH_UNRESOLVED — order_ref matches but amount is off

    Bank exceptions:
      - MISSING_FROM_INTERNAL    — bank row has no matching internal record
      - DUPLICATE_SETTLEMENT     — multiple bank rows for the same order_ref
    """
    rows: list[dict] = []

    # Pre-compute lookups on the FULL data
    bank_ref_set = set(all_bank["order_ref"].values)
    internal_ref_set = set(all_internal["order_ref"].values)
    bank_ref_counts = all_bank["order_ref"].value_counts()
    duplicate_bank_refs = set(bank_ref_counts[bank_ref_counts > 1].index)

    # --- Internal exceptions ---
    for _, irow in unmatched_internal.iterrows():
        ref = irow["order_ref"]
        i_amount = irow["amount"]

        if ref not in bank_ref_set:
            rows.append({
                "record_type": "internal",
                "payment_id": irow["payment_id"],
                "settlement_id": "",
                "order_ref": ref,
                "amount": i_amount,
                "reason_code": RC_MISSING_FROM_SETTLEMENT,
                "notes": "No bank settlement found for this order_ref",
            })
            continue

        # Bank rows exist for this ref but none matched — find why
        bank_rows_for_ref = all_bank[all_bank["order_ref"] == ref]
        bank_amounts = bank_rows_for_ref["amount"].values
        min_delta = min(abs(i_amount - ba) for ba in bank_amounts)

        if min_delta > FEE_AMOUNT_TOLERANCE * i_amount and min_delta > ROUNDING_TOLERANCE:
            rows.append({
                "record_type": "internal",
                "payment_id": irow["payment_id"],
                "settlement_id": "",
                "order_ref": ref,
                "amount": i_amount,
                "reason_code": RC_AMOUNT_MISMATCH,
                "notes": f"Closest bank amount delta={min_delta:.2f}, "
                         f"bank_amounts={list(bank_amounts)}",
            })
        else:
            rows.append({
                "record_type": "internal",
                "payment_id": irow["payment_id"],
                "settlement_id": "",
                "order_ref": ref,
                "amount": i_amount,
                "reason_code": RC_NO_MATCH_FOUND,
                "notes": f"Bank rows exist (count={len(bank_rows_for_ref)}) "
                         f"but none matched after both passes",
            })

    # --- Bank exceptions (only for truly unmatched bank rows) ---
    for _, brow in unmatched_bank.iterrows():
        ref = brow["order_ref"]
        is_dup = ref in duplicate_bank_refs

        if ref not in internal_ref_set:
            # Bank row has no corresponding internal record at all
            if is_dup:
                rows.append({
                    "record_type": "bank",
                    "payment_id": "",
                    "settlement_id": brow.get("settlement_id", ""),
                    "order_ref": ref,
                    "amount": brow["amount"],
                    "reason_code": RC_DUPLICATE_SETTLEMENT,
                    "notes": f"Duplicate bank row, no internal match "
                             f"(mutation: {brow.get('_mutation', 'unknown')})",
                })
            else:
                rows.append({
                    "record_type": "bank",
                    "payment_id": "",
                    "settlement_id": brow.get("settlement_id", ""),
                    "order_ref": ref,
                    "amount": brow["amount"],
                    "reason_code": RC_MISSING_FROM_INTERNAL,
                    "notes": f"No internal record for this order_ref "
                             f"(mutation: {brow.get('_mutation', 'unknown')})",
                })
        elif is_dup:
            # Internal record exists but this is an extra duplicate bank row
            rows.append({
                "record_type": "bank",
                "payment_id": "",
                "settlement_id": brow.get("settlement_id", ""),
                "order_ref": ref,
                "amount": brow["amount"],
                "reason_code": RC_DUPLICATE_SETTLEMENT,
                "notes": f"Extra duplicate bank row for {ref} "
                         f"(mutation: {brow.get('_mutation', 'unknown')})",
            })
        # else: internal ref exists, not a duplicate → already matched, skip

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "record_type", "payment_id", "settlement_id", "order_ref",
        "amount", "reason_code", "notes",
    ])


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
CASH_POSITION_TOLERANCE = 50.0  # INR — flag if delta exceeds this


def cash_position_check(matched: pd.DataFrame) -> dict:
    """Compare sum of matched internal amounts vs bank amounts.

    Returns a dict with the totals, delta, and whether it's within tolerance.
    """
    if matched.empty:
        return {"internal_total": 0, "bank_total": 0, "delta": 0, "ok": True}

    internal_total = matched["internal_amount"].sum()
    bank_total = matched["bank_amount"].sum()
    delta = abs(internal_total - bank_total)
    return {
        "internal_total": round(internal_total, 2),
        "bank_total": round(bank_total, 2),
        "delta": round(delta, 2),
        "ok": delta <= CASH_POSITION_TOLERANCE,
    }


def print_summary(
    total_internal: int,
    total_bank: int,
    matched: pd.DataFrame,
    exceptions: pd.DataFrame,
) -> None:
    """Print a human-readable reconciliation summary."""
    n_matched = len(matched)
    match_rate = (n_matched / total_internal * 100) if total_internal > 0 else 0
    cash = cash_position_check(matched)

    print()
    print("=" * 65)
    print("  RECONCILIATION RESULTS")
    print("=" * 65)
    print(f"  Internal records   : {total_internal}")
    print(f"  Bank settlement rows: {total_bank}")
    print(f"  Matched             : {n_matched}")
    print(f"  Exceptions          : {len(exceptions)}")
    print(f"  Match rate          : {match_rate:.1f}%")
    print()

    # Match type breakdown
    if not matched.empty and "match_type" in matched.columns:
        print("  Match type breakdown:")
        for mtype, count in matched["match_type"].value_counts().items():
            print(f"    {mtype:<20s} : {count}")
        print()

    # Exception breakdown by reason code
    if not exceptions.empty and "reason_code" in exceptions.columns:
        print("  Exception breakdown:")
        for code, count in exceptions["reason_code"].value_counts().items():
            print(f"    {code:<35s} : {count}")
        print()

    # Cash position cross-check
    print("  Cash Position Cross-Check:")
    print(f"    Matched internal total : INR {cash['internal_total']:>12,.2f}")
    print(f"    Matched bank total     : INR {cash['bank_total']:>12,.2f}")
    print(f"    Delta                  : INR {cash['delta']:>12,.2f}")
    status = "OK" if cash["ok"] else f"FLAGGED (exceeds INR {CASH_POSITION_TOLERANCE:.0f} tolerance)"
    print(f"    Status                 : {status}")
    print()

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main reconciliation pipeline
# ---------------------------------------------------------------------------
def reconcile(
    internal_path: str | Path = "data/internal_payments.csv",
    bank_path: str | Path = "data/bank_settlement.csv",
    output_dir: str | Path = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full two-pass reconciliation and write CSVs.

    Returns (matched_df, exceptions_df, audit_df).
    """
    # Load
    internal, bank = load_data(internal_path, bank_path)

    # --- Pass 1: exact match ---
    exact_matched, remaining_int, consumed_bank_1 = exact_match(internal, bank)
    consumed_1 = set(consumed_bank_1.index)

    # --- Pass 2: fuzzy fallback ---
    fuzzy_matched, unmatched_int, _ = fuzzy_match(remaining_int, bank)

    # Compute all consumed bank indices from both passes
    consumed_all = consumed_1.copy()
    if not fuzzy_matched.empty and "bank_settlement_id" in fuzzy_matched.columns:
        sid_to_idx = {row["settlement_id"]: idx
                      for idx, row in bank.iterrows()}
        for sid in fuzzy_matched["bank_settlement_id"]:
            if sid in sid_to_idx:
                consumed_all.add(sid_to_idx[sid])

    truly_unmatched_bank = bank.drop(index=list(consumed_all))

    # Combine matched results
    all_matched = pd.concat([exact_matched, fuzzy_matched], ignore_index=True) \
        if not fuzzy_matched.empty else exact_matched

    # --- Classify exceptions ---
    exceptions = classify_exceptions(
        unmatched_int, truly_unmatched_bank, internal, bank,
    )

    # --- Build audit log ---
    audit_rows = []
    for _, row in all_matched.iterrows():
        audit_rows.append({
            "event_type": "matched",
            "internal_payment_id": row["internal_payment_id"],
            "bank_settlement_id": row.get("bank_settlement_id", ""),
            "order_ref": row["order_ref"],
            "match_type": row["match_type"],
            "amount_delta": row.get("amount_delta", 0),
            "date_lag_days": row.get("date_lag_days", 0),
        })
    for _, row in exceptions.iterrows():
        audit_rows.append({
            "event_type": "exception",
            "internal_payment_id": row.get("payment_id", ""),
            "bank_settlement_id": row.get("settlement_id", ""),
            "order_ref": row["order_ref"],
            "match_type": row["reason_code"],
            "amount_delta": 0,
            "date_lag_days": 0,
        })
    # Cash position check — append as summary row
    cash = cash_position_check(all_matched)
    audit_rows.append({
        "event_type": "cash_position_summary",
        "internal_payment_id": "",
        "bank_settlement_id": "",
        "order_ref": "",
        "match_type": f"delta=INR {cash['delta']:.2f}",
        "amount_delta": cash["delta"],
        "date_lag_days": 0,
    })

    audit_df = pd.DataFrame(audit_rows)

    # --- Write outputs ---
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    all_matched.to_csv(out / "matched_pairs.csv", index=False)
    exceptions.to_csv(out / "exceptions.csv", index=False)
    audit_df.to_csv(out / "audit_log.csv", index=False)

    # --- Console summary ---
    print_summary(len(internal), len(bank), all_matched, exceptions)

    return all_matched, exceptions, audit_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run reconciliation from the command line."""
    # Resolve paths relative to project root
    root = Path(__file__).resolve().parent.parent
    internal_path = root / "data" / "internal_payments.csv"
    bank_path = root / "data" / "bank_settlement.csv"
    output_dir = root / "outputs"

    if not internal_path.exists() or not bank_path.exists():
        print("ERROR: Data files not found. Run data/generate_data.py first.")
        sys.exit(1)

    reconcile(internal_path, bank_path, output_dir)


if __name__ == "__main__":
    main()
