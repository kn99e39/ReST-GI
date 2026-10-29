# Document roles

| Document/location | Responsibility |
|---|---|
| `Architecture.md` | Central structure, design interfaces, ownership and validity contracts only |
| `FNTO_ARCHITECTURE_CONTRACT_V0_1.md` | Detailed physical K/C design contract |
| `RESEARCH_CENTRIC_TOPIC.md` | Research question, motivation and research objectives |
| `RESEARCH_ROADMAP.md` | Dependencies, research sequence and open decisions |
| `plans/` | Explicitly unadopted candidate designs and implementation plans |
| `worklogs/` | Authorisation, execution history, measurements, failures, interpretation, verdicts and reproducibility |
| `history/` | Superseded design documents |

Write an experimental result once in its Worklog and link to it where needed. Do not prepend status reports, benchmark tables, authorisation histories or batch verdicts to `Architecture.md`. Change the central architecture only when an explicit design decision changes its contracts. Candidate implementation or a gate outcome alone is not adoption.

Human-facing design and report documents belong under `docs/`. Executable protocols/configuration remain alongside isolated experiment code; primary data remain under the corresponding results location. Existing historical paths remain reproducible.
