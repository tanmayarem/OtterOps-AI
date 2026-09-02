"""
generate_data.py - Razorpay-style payment settlement reconciliation dataset.
"""
from __future__ import annotations
import random
from pathlib import Path
import pandas as pd

NUM_RECORDS = 60
GATEWAY_FEE_RATE = 0.02
random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

def _random_amount():
    wr = [(0.35,(199,1500)),(0.30,(1500,8000)),(0.20,(8000,25000)),(0.15,(25000,45000))]
    r, c = random.random(), 0.0
    for w,(lo,hi) in wr:
        c += w
        if r <= c:
            return round(random.uniform(lo,hi),2)
    return round(random.uniform(199,45000),2)

def _random_status():
    r = random.random()
    if r < 0.90: return "captured"
    elif r < 0.95: return "refunded"
    return "failed"

def generate_internal_payments(n=NUM_RECORDS):
    sd = pd.Timestamp("2026-08-20")
    records = []
    for i in range(1, n+1):
        do = random.randint(0,13)
        pd2 = sd + pd.Timedelta(days=do)
        records.append({
            "payment_id": f"PAY{i:04d}",
            "date": pd2.strftime("%Y-%m-%d"),
            "amount": _random_amount(),
            "order_ref": f"ORD{10234 + i - 1}",
            "status": _random_status(),
        })
    return pd.DataFrame(records)

def _typo_order_ref(ref):
    chars = list(ref)
    pos = random.randint(3, len(chars)-1)
    m = random.choice(["change","drop","insert"])
    if m == "change":
        o = chars[pos]
        if o.isdigit():
            chars[pos] = str((int(o)+random.randint(1,9))%10)
        else:
            chars[pos] = chr(ord(o)+random.randint(1,3))
    elif m == "drop":
        chars.pop(pos)
    else:
        chars.insert(pos, str(random.randint(0,9)))
    return "".join(chars)

def inject_bank_messiness(internal):
    bank_rows = []
    all_indices = list(range(len(internal)))
    random.shuffle(all_indices)

    mutation_plan = {
        "gateway_fee": all_indices[:5],
        "settlement_lag": all_indices[5:9],
        "order_ref_typo": all_indices[9:12],
        "duplicate": all_indices[12:15],
        "rounding_diff": all_indices[15:17],
    }

    clean_pool = all_indices[17:]
    random.shuffle(clean_pool)
    missing_indices = set(clean_pool[:3])
    clean_pool = clean_pool[3:]

    index_to_mutation = {}
    for mt, indices in mutation_plan.items():
        for idx in indices:
            index_to_mutation[idx] = mt
    for idx in clean_pool:
        index_to_mutation[idx] = None

    duplicate_indices = []

    for idx in range(len(internal)):
        row = internal.iloc[idx]
        mutation = index_to_mutation.get(idx)
        if idx in missing_indices:
            continue

        pay_date = pd.Timestamp(row["date"])
        amount = row["amount"]
        order_ref = row["order_ref"]
        tags = []

        if mutation == "gateway_fee":
            amount = round(amount * (1 - GATEWAY_FEE_RATE), 2)
            tags.append("gateway_fee_2pct")
        if mutation == "settlement_lag":
            lag_days = random.choice([1, 2])
            pay_date = pay_date + pd.Timedelta(days=lag_days)
            tags.append(f"settlement_lag_T+{lag_days}")
        if mutation == "order_ref_typo":
            original_ref = order_ref
            order_ref = _typo_order_ref(order_ref)
            tags.append(f"order_ref_typo:{original_ref}->{order_ref}")
        if mutation == "rounding_diff":
            diff = round(random.uniform(-0.50, 0.50), 2)
            while diff == 0.0:
                diff = round(random.uniform(-0.50, 0.50), 2)
            amount = round(amount + diff, 2)
            tags.append(f"rounding_diff_{diff:+.2f}")
        if mutation == "duplicate":
            tags.append("primary")
            duplicate_indices.append(idx)

        bank_rows.append({
            "settlement_id": f"SET{idx+1:04d}",
            "order_ref": order_ref,
            "amount": amount,
            "settlement_date": pay_date.strftime("%Y-%m-%d"),
            "_mutation": ",".join(tags) if tags else "clean",
        })

    for idx in duplicate_indices:
        row = internal.iloc[idx]
        bank_rows.append({
            "settlement_id": f"DUP{idx+1:04d}",
            "order_ref": row["order_ref"],
            "amount": row["amount"],
            "settlement_date": row["date"],
            "_mutation": f"duplicate_of_SET{idx+1:04d}",
        })

    return pd.DataFrame(bank_rows)

def print_summary(internal, bank, mutation_counts, missing_count):
    print("=" * 60)
    print("  DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"  internal_payments.csv : {len(internal)} records")
    print(f"  bank_settlement.csv   : {len(bank)} rows (includes duplicates)")
    print(f"  Unique order_refs in bank : {bank['order_ref'].nunique()}")
    print()
    print("  Mismatch breakdown (injected into bank_settlement.csv):")
    print(f"    Gateway fee (2% reduction) : {mutation_counts.get('gateway_fee', 0)} records")
    print(f"    Settlement lag (T+1/T+2)   : {mutation_counts.get('settlement_lag', 0)} records")
    print(f"    Order ref typo             : {mutation_counts.get('order_ref_typo', 0)} records")
    print(f"    Duplicate settlement rows  : {mutation_counts.get('duplicate', 0)} records")
    print(f"    Rounding difference        : {mutation_counts.get('rounding_diff', 0)} records")
    print(f"    Missing from settlement    : {missing_count} records")
    print()
    clean_count = len(bank) - sum(1 for _,r in bank.iterrows() if r["_mutation"] != "clean")
    print(f"  Clean rows in bank file  : {clean_count}")
    print(f"  Mutated rows in bank file: {len(bank) - clean_count}")
    print("=" * 60)

def main():
    DATA_DIR.mkdir(exist_ok=True)
    internal = generate_internal_payments(NUM_RECORDS)
    bank = inject_bank_messiness(internal)
    internal.to_csv(DATA_DIR / "internal_payments.csv", index=False)
    bank.to_csv(DATA_DIR / "bank_settlement.csv", index=False)

    mtc = {}
    for _, row in bank.iterrows():
        m = row["_mutation"]
        if m == "clean": continue
        for t in m.split(","):
            if "fee" in t: mtc["gateway_fee"] = mtc.get("gateway_fee", 0) + 1
            elif "lag" in t: mtc["settlement_lag"] = mtc.get("settlement_lag", 0) + 1
            elif "typo" in t: mtc["order_ref_typo"] = mtc.get("order_ref_typo", 0) + 1
            elif "duplicate" in t: mtc["duplicate"] = mtc.get("duplicate", 0) + 1
            elif "rounding" in t: mtc["rounding_diff"] = mtc.get("rounding_diff", 0) + 1

    missing_count = NUM_RECORDS - bank["order_ref"].nunique()
    print_summary(internal, bank, mtc, missing_count)
    print()
    print("Files written to", DATA_DIR)
    print("  - internal_payments.csv")
    print("  - bank_settlement.csv")

if __name__ == "__main__":
    main()
