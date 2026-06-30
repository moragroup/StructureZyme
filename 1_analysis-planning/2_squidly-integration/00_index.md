# Squidly Integration Plan

Goal: properly integrate **Squidly** (catalytic-residue prediction; ESM2 + LSTM) into the
Filterzyme / StructureZyme pipeline as a first-class, supported step alongside the existing
Chai-1 / Boltz-2 / Vina path, replacing today's two divergent, half-working integrations.

This plan is intentionally split into small files. Read them in order.

| # | File | Topic |
|---|------|-------|
| 1 | [`01_goals_and_scope.md`](01_goals_and_scope.md) | What "include Squidly" means, in/out of scope |
| 2 | [`02_current_state.md`](02_current_state.md) | How Squidly is wired today (v1 subprocess vs v2 enzymetk) and what's broken |
| 3 | [`03_target_architecture.md`](03_target_architecture.md) | Target design: single `SquidlyStep`, column contract, config |
| 4 | [`04_env_and_install.md`](04_env_and_install.md) | Conda env, `squidly` CLI install, GPU/CPU |
| 5 | [`05_implementation_steps.md`](05_implementation_steps.md) | Concrete file-by-file change list with `file:line` anchors |
| 6 | [`06_testing_and_validation.md`](06_testing_and_validation.md) | Unit, integration, and sanity-run checks |
| 7 | [`07_risks_and_open_questions.md`](07_risks_and_open_questions.md) | Known unknowns to clarify with PI / upstream |
| 8 | [`08_execution_order_and_vina_coupling.md`](08_execution_order_and_vina_coupling.md) | Skip flag, stage ordering, how Squidly biases Vina |
| 9 | [`09_phase0_squidly_install_and_schema.md`](09_phase0_squidly_install_and_schema.md) | **DO THIS FIRST**: install `squidly` CLI, discover pickle schema |

## Backend decision (locked, 2026-06-29)

Squidly is integrated **through `enzymetk.ActiveSitePred`**, the same pattern as Chai-1
and Boltz-2. This is a deliberate flip from an earlier draft of this plan that proposed
local-bundled inference. Rationale:

- Consistency with Chai/Boltz, which also live in `enzymetk` (`pipeline_v2.py:28-30`).
- Less vendored model code in Filterzyme; upstream `enzymetk` maintenance is shared.
- The bundled `filterzyme/squidly_final_models/*.pth` files become deletable.

Cost: the upstream `squidly` CLI must be installed alongside Python (it is **not**
currently on this machine - see `09_phase0_squidly_install_and_schema.md`).

If portions of `03_target_architecture.md` / `05_implementation_steps.md` still read as
local-bundled-first, the override above wins.

Cross-references:
- Repo analysis: [`../1_repo_analysis/04_model_integration_audit.md`](../1_repo_analysis/04_model_integration_audit.md) (Squidly = row 4)
- Setup sanity: [`../3_setup-sanity-check/`](../3_setup-sanity-check/)
- Lab sanity run: [`../4_lab-sanity-run/`](../4_lab-sanity-run/)

Assumption locked in by analysis: `pipeline_v2.py` is canonical; `pipeline.py` is deprecated.
All Squidly work targets v2.
