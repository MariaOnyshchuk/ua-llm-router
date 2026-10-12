# Decisions made without asking

One entry per decision: date, step number, what was chosen, why, and what the
other option was.

## 13 Oct 2026

- **Step 1. `docs/HANDOFF.md` is not committed.** Maria said earlier in the
  session not to commit it. Step 6 still updates its text in place. Other
  option: commit it with the report.
- **Step 1. The night steps run in the same session that built the work**, not
  in a separate loop, because no other session was started.
- **Step 1. `tests/test_composite_router.py` went into the planner commit**,
  although it also holds the AgentCoMa oracle test: one file cannot be split
  across two commits without interactive staging. Other option: one combined
  commit.
- **Before step 1. The 17 `Co-Authored-By` trailers on this branch were removed**
  by recreating 22 commits (Maria asked for this). Old tip `02520a5`, new tip
  `d3b5fe0`; backup branch `backup/before-trailer-rewrite-20261013`; hash map in
  `logs/hash_map_trailer_rewrite_20261013.txt`. The branch needs a
  `git push --force-with-lease` by Maria.
- **Step 4. Statements that the router "wins knowledge" over Mamay-12B on v4 were removed.**
  Rescored Mamay-12B knowledge is 0.750, equal to the router's 0.750, and the
  router run is not rescored. Other option: keep the claim with a caveat.
- **Step 4. "The FP8 pack loses knowledge (0.750 → 0.688)" was removed** from
  the same docs: the rescored pack knowledge is 0.750, so the drop was the
  scorer bug.
