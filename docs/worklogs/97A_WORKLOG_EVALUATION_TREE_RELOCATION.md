# Worklog — WL97A: Evaluation Tree Relocation (2026-11-16)

## What and why

The user decided to take the reviewer-facing evaluation tree out of `results/`, because `results/` mixes it with raw experiment outputs and is hard to browse. The tree moved from `results/evaluation/` to the top-level `evaluation/`.

This is housekeeping with a structural effect on paths. No scientific result, verdict or artifact content changed.

## Done

- **Move.** `results/evaluation/` → `evaluation/`. All 861 files moved, with folder names and contents unchanged.
  - The 564 tracked files are recorded by git as renames.
  - The 297 files that were untracked before (about 130 MB) remain untracked. Before the move, `results/` was gitignored. They are now kept out of git by the explicit list in `evaluation/.gitignore`, so that they are not committed by accident.
- **Redirect.** A redirect note is at `results/evaluation/README.md`.
- **Living documents updated.**
  - `AGENTS.md`, evaluation artifact conventions: the new path plus a relocation note.
  - `docs/README.md`: an evaluation row.
  - `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`: the provenance path.
- **Not changed.** Following the append-only rule, the old path is left in:
  - historical worklogs (86 of them name `results/evaluation/<N>/`);
  - manifests inside the evaluation folders;
  - frozen experiment scripts (87 write to the old path);
  - unadopted plans.

  Read every such path as `evaluation/<N>/`. Re-running an old export script would recreate files under `results/evaluation/`; any new export should write to `evaluation/<N>/`.

## Concurrency note

Another local session (`rest-gi-a0`) had in-progress, untracked WL21A work that writes to `results/evaluation/21A/`. Its 14 untracked files moved with the tree to `evaluation/21A/` and are listed in `evaluation/.gitignore`. That session was notified. Its untracked worklog and code were not touched.

## Commit

This worklog's commit.
