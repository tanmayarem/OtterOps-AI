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
(`_is_one_deletion`). Verified with 8 unit tests covering substitution,
insertion, deletion, two-edit, completely different, and empty strings.

### Bug 3: Exception classifier double-counting bank rows

The `classify_exceptions()` function was called with the full bank DataFrame as
`all_bank`, which meant every bank row — including those already matched in
passes 1 and 2 — was re-classified as a "bank exception." This produced 49
exceptions instead of 6.

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

### Design tradeoff: Why 57 matches instead of 60

The 3 unmatched records are genuinely missing from the bank settlement file
(simulating stuck/failed settlements). The generator intentionally excludes them
from `bank_settlement.csv` to simulate real-world settlement failures. The
engine correctly identifies them as `MISSING_FROM_SETTLEMENT` exceptions. This
is the "honest exception list" the track criteria require — we don't hide the 3
failures.

---

## 2026-09-02: LLM explainer integration and final pipeline run

### Issue 5: Groq API key invalid — graceful fallback worked as designed

During the final end-to-end pipeline run, the Groq API returned 401
`AuthenticationError: Invalid API Key`. This was expected — the `.env` file
contains a placeholder key.

**What happened:** The explainer attempted 6 LLM calls (one per exception),
each failed with 401, and each fell back to the mock explanation path. The
pipeline completed successfully with 6/6 exceptions explained using
template responses. No crash, no data loss.

**Resolution:** None needed — this is the designed behavior. The mock
fallback produces reasonable, structured explanations (e.g., "Payment was
recorded internally but no bank settlement was found...") with appropriate
confidence levels. When a valid Groq key is added to `.env`, the explainer
will automatically use real LLM responses instead.

**Note:** The stderr output shows the 401 errors as warnings, not exceptions.
This is intentional — the log line `log.warning("LLM call failed...,
falling back to mock")` makes the failure visible without being alarming.

### Issue 6: Unicode rupee symbol breaks Windows console

The `matcher.py` and `llm_explainer.py` used the rupee symbol in print
statements and log messages. Windows console with cp1252 encoding cannot
render this character, causing UnicodeEncodeError.

**Resolution:** Replaced all rupee symbols with "INR" in Python files and
logs. README.md keeps the symbol since GitHub renders it correctly in Markdown.

### Design note: LLM only explains, never re-decides

The architecture enforces a strict separation: the deterministic matcher
(exact_match + fuzzy_match + classify_exceptions) produces all decisions.
The LLM explainer reads exceptions.csv and adds human-readable explanations
but never modifies match decisions, never re-runs matching, and never
overrides exception classifications. This is stated in the llm_explainer.py
docstring header and enforced by the code structure — the explainer has no
access to the matching functions and no write path to matched_pairs.csv.

This separation was a deliberate design choice for the hackathon submission.
The track criteria emphasize "AI judgment — the right tool in the right
place, and where you chose not to use one." By keeping matching
deterministic and LLM-only for explanation, we demonstrate that we know
when NOT to use AI (matching must be reproducible and auditable) and when
AI adds value (explaining exceptions in natural language for human review).
