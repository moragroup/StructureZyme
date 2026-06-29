# Filterzyme / StructureZyme â Static Analysis

Mid-depth static analysis of the Filterzyme structural-filtering pipeline (repo root: `StructureZyme`, package: `filterzyme`).

**Scope:** docking + structure prediction + scoring models, code quality, bugs by severity, open-ended phased roadmap. Static analysis only â no code was executed.

**Canonical-pipeline recommendation (locked in):** keep `pipeline_v2.py`; port the few correct fixes from `pipeline.py` into it; deprecate and delete `pipeline.py`. Verification showed the `alternative_strucuture_for_vina` typo lives only in `pipeline.py` (lines 151, 154), and `pipeline_v2.py` already carries `@log_usage` instrumentation and the better `extract_docking_metrics` flow.

## Sections

| # | File | Topic |
|---|------|-------|
| 1 | [`01_executive_summary.md`](01_executive_summary.md) | Mission statement and top-5 critical findings |
| 2 | [`02_architecture.md`](02_architecture.md) | Current state, pipeline stages, operator pattern, module map |
| 3 | [`03_module_inventory.md`](03_module_inventory.md) | Every step file: purpose, key class, status |
| 4 | [`04_model_integration_audit.md`](04_model_integration_audit.md) | Per-model audit (Chai-1, Boltz-2, Vina, Squidly, AF2, PLACER, fpocket, PLIP, FreeSASA) |
| 5 | [`05_bugs_by_severity.md`](05_bugs_by_severity.md) | P0/P1/P2/P3 with `file:line` citations and copy-pasteable fixes |
| 6 | [`06_code_quality.md`](06_code_quality.md) | Duplication, shadowing, threading bugs, missing checkpoints |
| 7 | [`07_top10_improvements.md`](07_top10_improvements.md) | Top 10 code improvements |
| 8 | [`08_recommended_models.md`](08_recommended_models.md) | Top 5 recommended model additions (+ honorable mentions) |
| 9 | [`09_features_to_add.md`](09_features_to_add.md) | Multi-GPU, cofactors, config, checkpointing, logging |
| 10 | [`10_roadmap.md`](10_roadmap.md) | Open-ended phased roadmap (Phase 0 â Phase 5) |
| 11 | [`11_code_walkthrough.md`](11_code_walkthrough.md) | "What happens where" â function-level English summaries with `file:line` |
| 12 | [`12_assumptions_open_questions.md`](12_assumptions_open_questions.md) | Assumptions made and open questions for the owner |

## Reading order

For a first pass, read 1 â 5 â 10. For implementation work, jump to the relevant bug entry in `05` then to its callsite via `11`.

## Citation convention

`path/to/file.py:NN` â file path relative to repo root, with verified line number. All citations were verified by direct file read during analysis; refinements were made wherever the original outline drifted from the actual line numbers.
