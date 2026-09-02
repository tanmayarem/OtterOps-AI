from __future__ import annotations

import pandas as pd
import streamlit as st

from data.generate_data import generate_internal_payments, inject_bank_messiness
from src.matcher import reconcile, load_data


st.set_page_config(page_title="AI Finance Controller", layout="wide")

st.title("AI Finance Controller")
st.caption("Payment gateway settlement reconciliation — Razorpay Builtathon Track 04")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("Generate fresh data"):
        internal = generate_internal_payments()
        bank = inject_bank_messiness(internal)
        from pathlib import Path
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        internal.to_csv(data_dir / "internal_payments.csv", index=False)
        bank.to_csv(data_dir / "bank_settlement.csv", index=False)
        st.success("Data generated. Click 'Run Reconciliation' below.")

# --- Load data ---
try:
    internal, bank = load_data("data/internal_payments.csv", "data/bank_settlement.csv")
except FileNotFoundError:
    st.warning("No data found. Click 'Generate fresh data' in the sidebar.")
    st.stop()

# --- Run matcher ---
if st.button("Run Reconciliation"):
    matched, exceptions, audit = reconcile(
        "data/internal_payments.csv",
        "data/bank_settlement.csv",
        "outputs",
    )

    # --- Dashboard metrics ---
    st.subheader("Reconciliation Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Internal records", len(internal))
    c2.metric("Bank rows", len(bank))
    c3.metric("Matched", len(matched))
    c4.metric("Exceptions", len(exceptions))
    c5.metric("Match rate", f"{len(matched)/len(internal)*100:.1f}%")

    # --- Match breakdown ---
    st.subheader("Match Type Breakdown")
    if not matched.empty and "match_type" in matched.columns:
        match_counts = matched["match_type"].value_counts()
        st.bar_chart(match_counts)

    # --- Matched pairs ---
    st.subheader("Matched Pairs")
    st.dataframe(matched, use_container_width=True)

    # --- Exceptions ---
    st.subheader("Exceptions (Unresolved)")
    if not exceptions.empty:
        st.dataframe(exceptions, use_container_width=True)
    else:
        st.success("No exceptions — all records matched!")

    # --- Audit log ---
    with st.expander("Audit Log"):
        st.dataframe(audit, use_container_width=True)
else:
    st.info("Click **Run Reconciliation** to start the two-pass matching engine.")
