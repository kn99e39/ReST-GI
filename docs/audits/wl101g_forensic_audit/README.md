# WL101G forensic audit (2026-11-18)

An independent, read-only audit of the stopped WL101G construction and validation pipeline, requested by the research owner. It does not change the WL101G worklog, protocol, amendments, outputs or verdict D, all of which are preserved as historical evidence.

| File | Role |
|---|---|
| [WL101G_FORENSIC_AUDIT.md](WL101G_FORENSIC_AUDIT.md) | Audit report: implementation fact, verified failure matrix, contact inventory, gate and estimator assessment, measurement vs inference, unresolved questions, reproducibility |
| [PROPOSED_CONSTRUCTION_QUALIFICATION_CONTRACT.md](PROPOSED_CONSTRUCTION_QUALIFICATION_CONTRACT.md) | **Proposal requiring human approval**; authorises nothing |
| [repro/](repro/) | Read-only CPU reproductions (numpy) and their outputs; `SHA256SUMS.txt` covers the scripts, the outputs and the input packets |

Run the reproductions from the repository root with any Python that has numpy. They read `results/ecro_sss101g/primary_a2/construction_{0,1,2}.npz` and need no renderer.

The audit has no worklog number, because batch classification and naming belong to the research owner.
