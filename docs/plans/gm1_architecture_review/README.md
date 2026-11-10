# GM1 global transport architecture review (2026-11-10) — index

Design-only multi-agent review. Nothing was implemented, trained, rendered or benchmarked. **No candidate is adopted and nothing is authorised.** Repository revision `Selective_Recompute` @ `c7b0698`.

| File | Content | Author |
|---|---|---|
| `00_PROBLEM_BRIEF.md` | Neutral problem brief (written before any track ran; contains no candidate mechanism) | Coordinator |
| `10_` / `11_` / `12_` | Sealed independent design reports: A Directional Delta Transport, B path-space / reservoir reuse, C alternative operator (LPTG) | 3 design agents |
| `20_`–`23_` | Adversarial reviews: physical correctness, directional fidelity, economics, experimental identifiability | 4 review agents (did not read each other) |
| `30_`–`32_` | Novelty kill-search per track | 3 search agents |
| `35_prior_art_matrix.md` | Consolidated prior-art matrix | Coordinator |
| `40_`–`42_` | Cross-review responses (seal lifted): accept / rebut, revised claims, withdrawals | 3 advocate agents (fresh) |
| `50_provenance_audit.md` | Verification of repository claims, contract consistency, literature spot-check, independence, open disagreements | 1 audit agent |
| `60_coordinator_synthesis.md` | Coordinator synthesis and recommendation (class C with D) | Coordinator |
| `61_executive_summary_ko.md` | Decision-oriented final report, Korean, sections A–H | Coordinator |

## Workflow configuration

- One orchestrated workflow (run `wf_c11c20f4-f98`), 14 agents, about 33 minutes, no errors. Stages:
  1. Design: 3 agents in parallel, sealed.
  2. Reviews and novelty: 7 agents in parallel.
  3. Cross-review: 3 agents in parallel.
  4. Audit: 1 agent.
- Independence rule: design agents could read only the brief in this directory. The audit found no leakage (`50_` §6).
- Rejected hypotheses, withdrawn claims and negative findings are kept in the original reports; nothing was edited after the fact.
- Known limits:
  - cross-review agents relied on the reviews for worklog numbers (the audit traced them to the originals);
  - about 150 literature entries appear and 23 were spot-checked;
  - all cost figures are estimates or derivations.
