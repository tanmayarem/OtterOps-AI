from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI Finance Controller", layout="wide")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

MATCHED_PATH = OUTPUT_DIR / "matched_pairs.csv"
EXCEPTIONS_PATH = OUTPUT_DIR / "exceptions.csv"
EXPLAINED_PATH = OUTPUT_DIR / "exceptions_explained.csv"
AUDIT_PATH = OUTPUT_DIR / "audit_log.csv"
INTERNAL_PATH = DATA_DIR / "internal_payments.csv"
BANK_PATH = DATA_DIR / "bank_settlement.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV if it exists, otherwise return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _format_inr(val: float) -> str:
    """Format a number as INR with comma grouping."""
    return f"INR {val:,.2f}"


def _badge(text: str, color: str) -> str:
    """Return a small colored badge using HTML."""
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:600;">{text}</span>'


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("AI Finance Controller")
st.caption(
    "Payment gateway settlement reconciliation — Razorpay Builtathon Track 04. "
    "Deterministic two-pass matcher with LLM-powered exception explanations."
)

st.divider()

# ---------------------------------------------------------------------------
# Run Reconciliation
# ---------------------------------------------------------------------------
col_btn, col_toggle = st.columns([1, 3])
with col_btn:
    run_button = st.button("Run Reconciliation", type="primary", use_container_width=True)
with col_toggle:
    regenerate = st.checkbox(
        "Regenerate synthetic data before matching",
        value=False,
        help="If checked, generates fresh internal_payments.csv and bank_settlement.csv before running the pipeline.",
    )

if run_button:
    with st.spinner("Running pipeline..."):
        # Step 1: Generate data (if requested or if missing)
        if regenerate or not INTERNAL_PATH.exists() or not BANK_PATH.exists():
            st.write("Generating synthetic data...")
            from data.generate_data import generate_internal_payments, inject_bank_messiness

            internal_df = generate_internal_payments()
            bank_df = inject_bank_messiness(internal_df)
            DATA_DIR.mkdir(exist_ok=True)
            internal_df.to_csv(INTERNAL_PATH, index=False)
            bank_df.to_csv(BANK_PATH, index=False)

        # Step 2: Run matcher
        st.write("Running deterministic matcher...")
        from src.matcher import reconcile

        matched_df, exceptions_df, audit_df = reconcile(
            str(INTERNAL_PATH), str(BANK_PATH), str(OUTPUT_DIR),
        )

        # Step 3: Run LLM explainer
        st.write("Generating exception explanations...")
        from src.llm_explainer import explain_exceptions

        explain_exceptions(
            str(EXCEPTIONS_PATH), str(INTERNAL_PATH), str(BANK_PATH), str(EXPLAINED_PATH),
        )

    st.success("Pipeline complete!")

# ---------------------------------------------------------------------------
# Load existing outputs (works whether we just ran or files pre-existed)
# ---------------------------------------------------------------------------
matched = _load_csv(MATCHED_PATH)

# Prefer explained exceptions (has LLM columns); fall back to raw exceptions
if EXPLAINED_PATH.exists():
    exceptions = _load_csv(EXPLAINED_PATH)
else:
    exceptions = _load_csv(EXCEPTIONS_PATH)

audit = _load_csv(AUDIT_PATH)
internal_raw = _load_csv(INTERNAL_PATH)
bank_raw = _load_csv(BANK_PATH)

if matched is None and exceptions is None:
    st.info(
        "No reconciliation results found. **Click \"Run Reconciliation\"** above to "
        "generate data, run the matcher, and explain exceptions."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
st.subheader("Reconciliation Summary")

total_internal = len(internal_raw) if internal_raw is not None else 0
total_bank = len(bank_raw) if bank_raw is not None else 0
n_matched = len(matched) if matched is not None else 0
n_exceptions = len(exceptions) if exceptions is not None else 0
match_rate = (n_matched / total_internal * 100) if total_internal > 0 else 0.0

# Cash position delta
cash_delta = 0.0
cash_ok = True
if matched is not None and not matched.empty and "internal_amount" in matched.columns and "bank_amount" in matched.columns:
    cash_delta = abs(matched["internal_amount"].sum() - matched["bank_amount"].sum())
    cash_ok = cash_delta <= 50.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Match Rate", f"{match_rate:.1f}%")
m2.metric("Matched", f"{n_matched} / {total_internal}")
m3.metric("Exceptions", n_exceptions)
m4.metric(
    "Cash Position Delta",
    _format_inr(cash_delta),
    delta="OK" if cash_ok else "FLAGGED",
    delta_color="normal" if cash_ok else "off",
)
m5.metric("Bank Rows", total_bank)

# ---------------------------------------------------------------------------
# Match type breakdown
# ---------------------------------------------------------------------------
if matched is not None and not matched.empty and "match_type" in matched.columns:
    st.subheader("Match Type Breakdown")
    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        match_counts = matched["match_type"].value_counts().reset_index()
        match_counts.columns = ["match_type", "count"]
        st.bar_chart(match_counts.set_index("match_type"))
    with col_table:
        st.dataframe(
            match_counts.rename(columns={"count": "records"}),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Matched Pairs table (filterable)
# ---------------------------------------------------------------------------
if matched is not None and not matched.empty:
    st.subheader("Matched Pairs")

    match_types = sorted(matched["match_type"].unique().tolist())
    selected_types = st.multiselect(
        "Filter by match type",
        options=match_types,
        default=match_types,
        key="match_filter",
    )

    filtered = matched[matched["match_type"].isin(selected_types)]

    # Format for display
    display_cols = [
        "internal_payment_id", "bank_settlement_id", "order_ref", "bank_order_ref",
        "internal_amount", "bank_amount", "amount_delta", "date_lag_days",
        "match_type", "tolerance_used",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=350)

# ---------------------------------------------------------------------------
# Exceptions table with LLM explanations
# ---------------------------------------------------------------------------
if exceptions is not None and not exceptions.empty:
    st.subheader("Exceptions")

    display_exc = exceptions.copy()

    # Add manual review badge
    def _review_badge(row):
        conf = str(row.get("llm_confidence", "")).lower()
        notes = str(row.get("notes", "")).lower()
        if "manual review" in notes or conf == "low":
            return _badge("REVIEW", "#e74c3c")
        elif conf == "medium":
            return _badge("MEDIUM", "#f39c12")
        else:
            return _badge("HIGH", "#27ae60")

    display_exc["review_flag"] = display_exc.apply(_review_badge, axis=1)

    # Columns to show
    exc_cols = ["review_flag", "record_type", "order_ref", "payment_id", "settlement_id",
                "amount", "reason_code", "notes"]
    llm_cols = ["llm_explanation", "llm_confidence"]
    exc_cols_available = [c for c in exc_cols if c in display_exc.columns]
    llm_cols_available = [c for c in llm_cols if c in display_exc.columns]

    # Render with HTML badges
    st.markdown("**Reason codes:** `MISSING_FROM_SETTLEMENT` | `DUPLICATE_SETTLEMENT` | "
                "`AMOUNT_MISMATCH_UNRESOLVED` | `NO_MATCH_FOUND` | `MISSING_FROM_INTERNAL`")

    col_exc, col_llm = st.columns([1, 1])
    with col_exc:
        st.markdown("**Exception Details**")
        st.dataframe(display_exc[exc_cols_available], use_container_width=True, height=300)
    with col_llm:
        st.markdown("**LLM Explanations**")
        if llm_cols_available:
            st.dataframe(display_exc[["order_ref", "reason_code"] + llm_cols_available],
                         use_container_width=True, height=300)
        else:
            st.info("No LLM explanations available. Run the pipeline to generate them.")

# ---------------------------------------------------------------------------
# Cash Position Cross-Check (expanded detail)
# ---------------------------------------------------------------------------
if matched is not None and not matched.empty and "internal_amount" in matched.columns:
    with st.expander("Cash Position Cross-Check Details"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Matched Internal Total", _format_inr(matched["internal_amount"].sum()))
        col_b.metric("Matched Bank Total", _format_inr(matched["bank_amount"].sum()))
        col_c.metric("Delta", _format_inr(cash_delta),
                      delta="Within tolerance" if cash_ok else "Exceeds INR 50 tolerance",
                      delta_color="normal" if cash_ok else "off")

        st.caption(
            "The delta arises from gateway fee deductions and rounding differences "
            "that the matcher successfully reconciled via fuzzy tolerance. This is expected."
        )

# ---------------------------------------------------------------------------
# Audit Log + Download
# ---------------------------------------------------------------------------
if audit is not None and not audit.empty:
    st.subheader("Audit Log")
    with st.expander("View full audit log"):
        st.dataframe(audit, use_container_width=True, height=250)

# Download button for audit log
if AUDIT_PATH.exists():
    with open(AUDIT_PATH, "rb") as f:
        st.download_button(
            label="Download Audit Log (CSV)",
            data=f,
            file_name="audit_log.csv",
            mime="text/csv",
        )
