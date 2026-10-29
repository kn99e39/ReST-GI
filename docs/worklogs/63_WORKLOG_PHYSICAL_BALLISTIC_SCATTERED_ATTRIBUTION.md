# Worklog63 — Physical ballistic/collided attribution for primary Pair 7 (2026-10-30)

## Authorisation and fixed population

The user explicitly authorised the missing physical component measurement, not a further reconciliation of WL59–62. Start: `5a5df26`, branch `Selective_Recompute`; no branch/worktree change. [Frozen design](../plans/PHYSICAL_COMPONENT_ATTRIBUTION_V1.md) and [protocol](../../experiments/fnto_components63/protocol/components_v1.json) were committed with the tagging/gating/analysis source as **272b13a**, before generation.

Exactly **pt0 / original A2 test fields62 and63**, two independent64M replicas per field, each two32M halves: **eight packets / 256M physical reference photons**. Fresh seeds `630000000+field*100+replica*10+half`. No rest/other geometry, new model, training, B quadrature change, loss/rank/threshold change, coupled solve or successor. Original files, source and checkpoints are read-only.

## Instrumentation and prerequisite gates

The original first-exit estimator and its untagged total sink remain intact. An observational AST transformation inserts exactly one assignment before the existing exit sink: zero-collision iff the current path is in loop step1. All later surviving iterations have already undergone at least one medium collision. Removing the added assignment restores the exact original AST; no original path operation, random draw, weight or control-flow node is changed.

The sidecar sink receives the **same emitted events** after the unmodified original sink. It independently records tagged T, B and Scat first/second moments and exclusive event counts. It has no RNG call and does not alter path tensors. Every primary packet compares its original untagged total/moments to the tagged total on these actual identical sample paths, and separately checks tagged B+Scat closure. Thus Gate A/B consume no extra physical reference samples.

A deterministic scripted-intersection synthetic test also runs the original/tagged walkers from cloned generator states, compares every ray/intersection state, total weights/moments and final RNG state, and checks known one-zero/two-collided events. This is a correctness test, not an additional physical reference population. Three further tests cover covariance/mean scaling and a cancellation case where magnitude-only attribution would be wrong.

Each photon belongs to one mutually exclusive first-exit class, so its raw B×Scat cross moment is exactly0. The statistical covariance is **not zero**: per aggregated half-mean bin, `Cov(B,Scat)=-B*Scat/(N-1)`. Actual variance reconstruction includes this term. Four independent halves average with variance/covariance sums divided by16; independent fields add their contrast variances/covariances. No independent-pixel confidence interval is claimed.

## Generation and completion

The preflight source/input/geometry/checkpoint checks and all four focused tests passed. All eight fixed32M packets passed physical and actual same-path component gates. Gate C qualified T/B/Scat contrasts in every RGB channel. The run completed without failed attempts, packet replacement, budget increase or implementation amendment. Detached process used a new session and lock; server had no compute peer at launch.

A separately committed **report-only** detail exporter is declared after the main attribution measurement: it reads the completed coarse arrays to report absolute component-action errors/powers and contrast amplitude/alignment. It does not sample, infer, fit or change any gate. All main/raw decomposition and replica classification results remain unchanged.
