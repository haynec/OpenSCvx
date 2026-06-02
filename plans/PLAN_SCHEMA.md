# Plan file schema

Plans live under `plans/<kebab-case-slug>.md`. One plan = one coherent deliverable with a defensible end state.

## Section order (required)

1. **Title** — `# <Short name>`
2. **Status** — `Draft | Review | Approved | Superseded` and optional one-line scope anchor.
3. **Motivation** — Current behavior / gap; intended call site (example path, API entry). Cite `path:line` where the gap is visible.
4. **Approach** — Proposed shape: variables, constraints, modules, integration points. Use subsections; cite `path:line` for hooks. Prefer tables for dimensions and constraint inventories.
5. **Out of Scope** — Explicit non-goals for this plan (deferrals, not “maybe later” vagueness).
6. **Open Questions** — Unresolved decisions. Bare bullets are fine. Add `**Recommended:** …` only when grounded in code, a quoted snippet, or reasoning tied to a citation.
7. **Decision Log** — Dated entries: `YYYY-MM-DD — <choice> — Rejected: <alternative> because <reason>.`

## Voice

- Present tense for the target design (“The state includes …”), past tense in Decision Log.
- Implementation plan, not a paper summary: every constraint should map to an encoding strategy in OpenSCvx/Frax.
- Do not restate this schema in each plan; link `plans/PLAN_SCHEMA.md` once if helpful.

## Citations

- Format: `` `path/to/file.py:42` `` or a fenced block with the `start:end:path` form when showing code.
- Spot-check before marking Approved: stale line numbers are worse than no citation.

## Example

See `plans/mjx-dynamics-adapter.md` when present; otherwise use `plans/cito-frax-flat-ground.md` as the reference density.
