# GM1 architecture review — 35 — Consolidated prior-art matrix

Compiled by the Coordinator from the three novelty kill-searches (`30_`, `31_`, `32_`), which hold the full citations and verification levels. Spot-check status is in `50_provenance_audit.md` §4: 20 of 23 VERIFIED, 2 PARTLY, 1 UNVERIFIED, none WRONG.

Classes: ALREADY KNOWN / PLAUSIBLY DISTINCT / UNCERTAIN / NOT SUPPORTED. **No mechanism in any track was classified PLAUSIBLY DISTINCT.**

| Track | Central mechanism | Closest prior art | Class | What remains |
|---|---|---|---|---|
| A | Delta identity `(I − K C1) Δb = K(Δe + ΔC b0)`; incremental solve | Incremental radiosity (Chen 1990); Drettakis and Sillion 1997; Granier and Drettakis 2001 (glossy) | ALREADY KNOWN | none |
| A | Persistent directional outgoing state `b0(x, ω)` | Directional radiosity / radiance caching; neural radiosity (Hadadan et al. 2021) | ALREADY KNOWN as a state class; NOT SUPPORTED as a contribution (no representation proposed) | — |
| A | On-demand T0 form (no stored state) | Residual path integrals for re-rendering (Xu et al. 2024) | ALREADY KNOWN | comparator arm |
| A | Discover by delta, overwrite by selective regather | Bala, Dorsey and Teller 1999 (ray segment trees); selective photon tracing (Dmitriev et al. 2002) | ALREADY KNOWN; combination UNCERTAIN | measured utility only |
| A | Low-rank / delta transfer for moved objects | Delta radiance transfer (Loos et al. 2012) | ALREADY KNOWN | — |
| B | Path store + suffix replay (PR1) | Stored ray trees; Bala et al. 1999; Havran et al. 2003; correlated path tracing | ALREADY KNOWN | measurement / lower-bound comparator |
| B | Geometric segment-level invalidation | Bala et al. 1999; Drettakis and Sillion 1997 | ALREADY KNOWN | — |
| B | Difference / control-variate re-render (PR2) | Günther and Grosch 2015; Rousselle et al. 2016; Xu et al. 2024 | ALREADY KNOWN | — |
| B | Reservoir temporal reuse (PR3) | ReSTIR GI 2021; GRIS / ReSTIR PT 2022; ReSTIR PT Enhanced 2026 | ALREADY KNOWN | — |
| B | Path births / specular repair | ReSTIR BDPT 2025; PSMS-ReSTIR 2025; specular manifold sampling | ALREADY KNOWN | — |
| B | "Saving bounded by 1/ρ for any selective scheme"; "reuse cannot help births" | counter-classes: residual sampling; object-origin sampling, caustic reservoirs | NOT SUPPORTED (withdrawn / narrowed) | — |
| C | Roughness-thresholded lobe partition | Two-pass methods 1987–1991; GRIS hybrid shift; Lumen | ALREADY KNOWN | application only (partition taken from a learned closure) |
| C | Finite directional field + traced sharp transport | Sillion et al. 1991; Immel et al. 1986; Granier et al. | ALREADY KNOWN | measured utility |
| C | Persistent transport graph with compiled per-material matrix | Glossy hierarchical-radiosity links; path graphs (Deng et al. 2021) | ALREADY KNOWN | engineering |
| C | Segment validity certificates, selective regeneration, F ≡ S | Drettakis and Sillion 1997; Bala et al. 1999; Granier and Drettakis 2001; Martín et al. 2003 | ALREADY KNOWN; economics NOT SUPPORTED | — |
| C | Readout by exact rough eval + sharp chains | Two-pass final gather; radiance caching; neural radiance caching (Müller et al. 2021) | ALREADY KNOWN | — |
| all | Learned add-ons (guiding, cache as control variate) | Path guiding; NRC 2021; neural control variates 2020 | ALREADY KNOWN; hybrid with a reusable learned state UNCERTAIN (application level), unspecified | — |

**Evidence a credible contribution would require (all tracks).** Measured quality-matched lifecycle utility on a physically aligned workload, against cold and warm path-traced re-rendering and against a same-architecture full rebuild, for a mechanism that holds learned global state. No track supplies such a mechanism.
