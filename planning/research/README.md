# research/

Scouting notes for lessons that are not built yet. **This is not `evidence.md`.**

`evidence.md` holds numbers this course stands behind: measured here, reproducible, and
load-bearing for a decision already made. The reports in this folder are one step before
that. They were produced by research agents to answer "is there a measurable notebook in
this topic at all", and while each one separates what it verified by running code from
what it recalled from a paper, **nothing here has been re-run by hand.**

The rule, therefore:

> A number moves from `research/` to `evidence.md` only after it has been re-run
> independently and the script that produces it is checked in. Until then it is a lead,
> and a notebook may not cite it as a result.

This matters more than usual here, because the findings are mostly *negative* — RankMe
ranks encoders backwards, HippoRAG's headline is a max over three datasets, retrieval
inverts under negation — and a negative result asserted on someone else's authority is
exactly the failure mode `principles.md` §2 exists to prevent. A measured negative result
is the best thing this course can teach. An unverified one is worse than nothing.

| Report | Topic | Scripts |
|---|---|---|
| `embedding_space_measurement.md` | Training-free geometry of frozen encoders; can you rank encoders without a probe? (No.) | headline re-run and promoted to `evidence.md` §10; rest unverified |
| `memory_beyond_rag.md` | Structured memory, forgetting, consolidation, and what survives at laptop scale | `memory_experiments/` |
| `neurosymbolic_and_structure.md` | Where symbolic structure beats a vector space, and what LLM extraction costs | `neurosymbolic_scripts/` |

The embedding report's scripts were not persisted. Its headline — the RankMe inversion,
the Gaussian-noise null, and the whitening invariance argument — has been independently
re-derived and now lives in `evidence.md` §10; the reproduction matched to the printed
precision. The rest of that report (TwoNN, alignment/uniformity, the DTD and flowers
tables) has **not** been verified.

These files are scaffolding. When a lesson built on one of them has been run, fold the
surviving numbers into `evidence.md` and delete the report — same lifetime rule as
`lesson<N>.md`.
