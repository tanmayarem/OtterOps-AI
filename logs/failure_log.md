# Failure Log

Running record of bugs, edge cases, and design tradeoffs during development.

---

## 2026-09-02: Matcher build — tolerance and exception classification

### Bug 1: gateway_fee vs rounding_diff ambiguity

The original matcher had `gateway_fee` and `rounding_diff` as separate match
types, but in practice they overlap. A INR 5,000 record with a 2% gateway fee
becomes INR 4,900 (delta INR 100), while a rounding diff is always < INR 0.50. The
threshold between them was ambiguous — should a INR 0.49 diff be "rounding" or
"fee"?

**Resolution:** Unified them into a single fuzzy matching pass with a 2.5%
tolerance. The engine now tags which tolerance triggered the match (`fuzzy_fee`
vs `fuzzy_lag_T+N`), rather than trying to classify the root cause. This is
more honest — we can't always know *why* an amount differs, only *that* it
falls within tolerance.

### Bug 2: Levenshtein edit-distance-1 edge case

The initial `_edit_distance_1` implementation used a naive character-by-character
comparison that broke on insertions at the end of strings (e.g., "ORD100" vs
"ORD1000"). The function returned `False` because it didn't handle the "skip
one char" logic correctly when the mismatch happened at the last position.

**Resolution:** Rewrote as a two-phase check: (1) for same-length strings, count
differing positions; (2) for length-diff-1 strings, use a sliding skip pointer
(`_is_one_deletion`). Verified with8 unit tests covering substitution,
insertion, deletion, two-edit, completely different, and empty strings.

### Bug 3: Exception classifier double-counting bank rows

The `classify_exceptions()` function was called with the full bank DataFrame as
`all_bank`, which meant every bank row — including those already matched in
passes 1 and 2 — was re-classified as a "bank exception." This produced 49
exceptions instead of6.

**Root cause:** The `fuzzy_match()` function returns `unmatched_bank` by
subtracting `consumed_bank_idx` from the bank DataFrame. But `consumed_bank_idx`
is tracked *within* fuzzy_match's own iteration scope. Rows consumed in pass 1
(exact_match) were not tracked in pass 2, so they reappeared as "unmatched."

**Resolution:** Moved consumed-bank-index tracking into `reconcile()`. After both
passes, it computes the union of consumed indices from exact_match and
fuzzy_match, then filters the bank DataFrame before passing to
`classify_exceptions()`. This ensures only truly unconsumed bank rows enter the
exception bucket.

### Bug 4: Empty DataFrame in test helpers

The `_make_internal([])` helper tried to coerce columns on an empty DataFrame,
raising a `KeyError: 'amount'` because pandas creates a RangeIndex for empty
lists with no columns.

**Resolution:** Added early return with explicit column schema when the input
list is empty.

### Design tradeoff: Why no fuzzy matching on amount for typo strategy

Strategy B (typo correction) requires amount to be *exact* (delta < INR 0.01),
while Strategy A (fee/lag) allows 2.5% tolerance. This is intentional: if the
order_ref has a typo AND the amount is off by 2.5%, we can't be confident it's
the same transaction — it could be a different order that happens to have a
similar ref. Requiring exact amounts for typo matches reduces false positives
at the cost of missing some legitimate matches. This is the conservative choice
for a finance reconciliation system where false matches are worse than false
exceptions.

### Design tradeoff: Why57 matches instead of60

The 3 unmatched records are genuinely missing from the bank settlement file
(模拟 stuck/failed settlements). The generator intentionally excludes them from
`bank_settlement.csv` to simulate real-world settlement failures. The engine
correctly identifies them as `MISSING_FROM_SETTLEMENT` exceptions. This is the
"honest exception list" the track criteria require — we don't hide the3
failures.
