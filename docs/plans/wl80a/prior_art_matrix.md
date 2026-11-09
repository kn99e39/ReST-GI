# WL80A prior-art matrix and novelty kill search

> WL80A post-sealing record (workflow `wf_9ac78ed1-61a`, resumed once after a session interruption; completed agents replayed from cache). Agent texts are reproduced verbatim (headings added by the Coordinator). The sealed Track A / Track B records were not modified. Design-only: nothing was implemented, trained or measured.

## Consolidated matrix (chair)


# Consolidated prior-art matrix (from the three novelty kill searches)

**Verification levels.** These are as reported by the panel searchers; the chair did not re-verify any of them.
- **V**: abstract or page read by the searcher.
- **V-s**: snippet or listing only.
- **V-repo**: verified in an earlier project record.
- **M**: from memory, not verified.

**Classification labels.** ALREADY KNOWN / PLAUSIBLY DISTINCT / UNCERTAIN / NOT SUPPORTED. Absence of an identical published system is never treated as evidence of novelty.

## A. Mechanism-level matrix

| # | Prior work (source; verification) | Shared mechanism | BGC-0 / BGC-R | C1 CP-K | C2 SBG |
|---|---|---|---|---|---|
| 1 | Jensen et al. 2001, practical BSSRDF (graphics.stanford.edu/papers/bssrdf; V-s) | Pair weight × separable angular lobes; SS + MS split | A3 at R = 1 with analytic factors: **ALREADY KNOWN** structure | — | Bounded-medium kernel lineage (with #12) |
| 2 | First-collision source: Alcouffe 1985 (OSTI 5817627; V), Gomes & Stevens 1993 (V-repo), DOCTORS arXiv 1910.11335 (V-s); Photon Beam Diffusion, Habel–Christensen–Jarosz 2013 (V-s) | Exact uncollided flux as the source of a collided solve; beam × kernel | A1 is a lossy-moment version: **ALREADY KNOWN** (weaker) | F_in / B complementarity (C-b): **ALREADY KNOWN** | D1 single/multi-flight split: **ALREADY KNOWN** |
| 3 | Integral RTE with FFT/FMM: Fan, An & Ying, JCP 2019, arXiv 1802.03061 (V); Ren, Zhang & Zhong, arXiv 1610.00835 (V) | Angle-averaged Fredholm equation; homogeneous convolution kernel by FFT; angular recovery by a scattering-free sweep | — | Core of C1 (ψ-solve then Φout): **ALREADY KNOWN**. Track B missed these closest matches | Base-kernel lineage |
| 4 | 3-D ray-traced collision probabilities: Roy, Hébert & Marleau 1989 (V-s); Marleau & Hébert NSE 1986 (V); NSE 2003 (V); CP conservation and reciprocity relations (V) | Flat-source region CPs and escape by ray tracing; conservation normalisation; reciprocity | Repaired A4 + R: **ALREADY KNOWN** | C-c complement-escape passivity: **ALREADY KNOWN** in substance (sub-stochastic absorbing chain). Port/bin escape at 16/64 resolution: **UNCERTAIN** (DPn interface currents, M) | D2 identity = CP conservation sum rule in resolvent form: **ALREADY KNOWN** |
| 5 | Zonal method, Rushmeier & Torrance 1987 (V); Sillion 1995 (V); Hottel / exchange factors (OSTI 6460797; V-s) | Volume zones with geometric exchange factors, solved jointly with surfaces; summation rules; reciprocity | Repaired A4 + R are textbook bookkeeping: **ALREADY KNOWN** | C-f (joint K + C collision densities): **ALREADY KNOWN**. Directional boundary exitance: **UNCERTAIN** (search incomplete) | Volume-zone scalar kernel: **ALREADY KNOWN** |
| 6 | Passive multiport macromodelling, Gustavsen 2010 (V-s) | Fitted port operator with enforced passivity and reciprocity | A4 + R: **ALREADY KNOWN** | — | — |
| 7 | Translucent Radiosity, Sheng et al. TVCG 2014 (V); Lensch et al. 2002/2003 (V) | Intrinsic per-object SSS matrix combined with free-space form factors | — | K/C split: **ALREADY KNOWN**. Directional, exact construction is a DISTINCT choice, not a mechanism | — |
| 8 | Body-fitted volumetric SSS: Wang et al. EG 2010 (V); Arbree, Walter & Bala TVCG 2011 (V); polygrid (M) | Interior solve on an object-fitted discretisation, valid under topology-preserving deformation | — | C-e: **ALREADY KNOWN** | — |
| 9 | Response-matrix / interface current / COMET (V-s; original Mosher–Rahnema paper not retrieved) | Precomputed boundary-in → boundary-out response per unique local problem, coupled globally | K as a response matrix | K as an explicit response matrix: **ALREADY KNOWN** | "Finite chart set at rest" = unique local problems: **ALREADY KNOWN** |
| 10 | Modular Flux Transfer, Zhao et al. 2013 (V/V-s); Shell transport functions, Moon et al. 2007 and Müller et al. 2016 (V); Hašan & Ramamoorthi 2013 (V) | Precomputed reusable block or shell transfer; albedo by order | A6 structure: **ALREADY KNOWN** | Structure of F_in / W / Φout; M9 shared-W RGB: **ALREADY KNOWN** | Canonical-context supervision: **ALREADY KNOWN** |
| 11 | DeepRTE, arXiv 2505.23190 (V) | Learned RTE Green's function from delta inflows; first-flight attenuation net; linear superposition | A1 + boundary response: **ALREADY KNOWN** | C-g argument: **ALREADY KNOWN** | Source-free learned RTE Green's function: **ALREADY KNOWN** |
| 12 | Neural Green's Functions, arXiv 2511.01924 (V); Neural Green's Operators, arXiv 2406.01857 (V); Boullé & Townsend, FoCM 2022 (V/V-s); GF-Net, BI-GreenNet (V) | Geometry-conditioned, source-agnostic learned Green's operator; embedded symmetry or conservation; singular-part subtraction | M0 class: **ALREADY KNOWN** | C-g: **ALREADY KNOWN** | Principle: **ALREADY KNOWN** (WL77A B7 upheld) |
| 13 | Element Learning, Du & Stechmann, arXiv 2308.02467 (V) | Learned geometry-conditioned RTE in2out boundary operator, composed by a global solve; trained on a reference element | Closest structural precedent: **ALREADY KNOWN** | Not a threat (C1 is not learned) | Chart or element supervision: **ALREADY KNOWN** |
| 14 | Vicini, Koltun & Jakob 2019 (V); Leonard et al. 2021 (V); Deep Scattering 2017 (V-s) | Learned mass × conditional exit distribution from local, identity-free geometry; MC on canonical shapes; cross-shape generalisation | A2 + A4 + cross-shape aim: **ALREADY KNOWN** | — | Descriptor-conditioned learned bounded-medium response: **ALREADY KNOWN**. Passive linear-operator form: **PLAUSIBLY DISTINCT** |
| 15 | Frame averaging, Puny et al. 2022 (V); gauge-equivariant mesh CNNs (V-s); local deformable PRT (V-s) | Reynolds or frame averaging; gauge and deformation covariance | A5 = exactly frame averaging: **ALREADY KNOWN**. BGC covers a strict subset of the known techniques | C-d: **ALREADY KNOWN** (routine symmetry exploitation) | Reynolds τ: **ALREADY KNOWN** |
| 16 | HyperDeepONet / low-rank hypernetwork heads (V-s); CPCA PRT (V-s); NBRDF factorisation (V); Neural Parametric Mixtures (V-s) | Conditioned low-rank output operator; normalised mixtures | A3: **ALREADY KNOWN** | — | — |
| 17 | RENO, arXiv 2602.11631 (V); Dual Photography (V-A) | Architecture-enforced source–receiver reciprocity | Repaired R only states the condition: behind prior art | — | — |
| 18 | Domain monotonicity of Green's functions (arXiv 2402.03571, 2608.19356; V); transport-level half-space Green's functions, Liemert & Kienle 2013 (V), Martelli (V-title) | G_Ω ≤ G_Ω′ for nested domains; exact or near-exact half-space RTE Green's functions | — | — | Principle **ALREADY KNOWN**. As a hard output constraint on a learned RTE kernel: **PLAUSIBLY DISTINCT** (weak). Its value with G∞ as the base: **NOT SUPPORTED** |
| 19 | Per-asset neural transport: RNA (arXiv 2312.09398), 8DNA (arXiv 2604.25129), TransGI (arXiv 2506.09909), NeuPreSS = "Neural BSSRDF" (one work, DOI 10.1111/cgf.15234, FACT per searcher), NRC (V/V-s) | Learned per-asset transport; 8DNA α × p | A4 adjoint form: **ALREADY KNOWN**. Cross-source aim distinct but unachieved: **NOT SUPPORTED** | DISTINCT only because classical, not a new mechanism | Source-free vs per-asset: a real difference, established by #12–13 |
| 20 | **Project history** (FACT): global ATO (WL68/74/75); FIA (WL76); W+UV-Q (WL78); D4h maps (WL79/80); WL53/55/56 MC oracle K; WL77A H2 RBG and H3 LCR (`docs/plans/wl77a/coordinator_decision.md` l.13–15; `adjudication_draft.md` l.194) | — | Only A3 is new relative to WL57–80; it is a hypernetwork head aimed at a secondary factor | **WL77A H3 at θ = 0** (the learning-free CPM operator whose gate performance WL77A left open). The MC oracle K already has C-g's source-agnosticism | **A revised WL77A H2 RBG**: exact Π ends plus learned scalar source-free Γ. Real changes: graded depth, chart supervision, D2, monotone ratio |

## B. Classification summary per candidate

| Candidate | Central mechanism | Residual possibly distinct element | Claim that is NOT SUPPORTED |
|---|---|---|---|
| BGC-0 | ALREADY KNOWN (#3/#13/#14/#19 class) | Combination only: UNCERTAIN | Out-of-orbit source transfer (unmeasured; family evidence negative) |
| BGC-R | ALREADY KNOWN (#4–#6 for A4 + R; #15 for A5) | Corridor descriptor: UNCERTAIN (unspecified); per-class rank only as a measurement | Any transfer claim (Track A makes none) |
| C1 | ALREADY KNOWN (#2–#5, #8, #9; internally WL77A H3 θ = 0) | Port/bin directional escape at 16/64 density: UNCERTAIN; the integrated system: UNCERTAIN (engineering) | C-g as a new contribution |
| C2 | ALREADY KNOWN (#11–#14; internally WL77A H2) | Monotone nested ratio as a hard constraint: PLAUSIBLY DISTINCT; D2 device: PLAUSIBLY DISTINCT | "Exact energy totals" via D2; corners without corner data (withdrawn) |

## C. Open literature checks (not closed by any searcher)

1. CP or interface-current codes that resolve escape at high angular order on fine boundary patches.
2. Volumetric zonal or radiosity variants with directional boundary exitance.
3. Fan et al. §4 (general boundary conditions), and Ren et al.'s boundary treatment.
4. A targeted search for monotone-ratio constraints on learned Green's functions.


---

# Novelty kill searches (one per candidate)

## BGC

# WL80A cross-review: novelty kill search for BGC-Transport, original and repaired

**Reviewer scope.** I read the sealed records `neutral_problem_brief.md`, `trackA_original_bgc.md`, `trackA_critical_audit.md` (including the full novelty specialist section, l.964–1126), `trackA_decision_log.md`, `trackA_revised_architecture.md` and `trackB_selected_candidate.md`, plus the C2 lines of `trackB_candidate_records.md`. I wrote no file and modified nothing. There was no implementation, no measurement and no code execution.

I re-checked these repository anchors read-only:
- WL75 l.195: BND4_FAR con_coarse is 0.159 for the G2 oracle against 0.803 / 0.824 for learned A75.
- WL75 l.308: "Global angular rank is not the bottleneck … the spatial geometry → T mapping is."
- WL76 l.98–110: the FIA form is K_scat = albedo·SS + albedo²·Σ_k W·R_θ, with escape mass ρ, tail share τ, a 512-NN softmax plus a 240-cell tail, ring lobes and invariant features.

**Verification levels used in the table.**
- **V**: I read the abstract, HTML or project page this session.
- **V-s**: verified only at search-snippet or listing level.
- **V-A**: Track A verified it this session; I did not re-check.
- **U**: background knowledge, not re-opened.

**Claim labels:** FACT, DERIVATION, INFERENCE, HYPOTHESIS, SPECULATION.

---

## 1. What is being tested for novelty

These are the sealed specifications. I decompose BGC into one central mechanism and its claimed contributions.

**M0, central mechanism (original).** A single shared network generates a nonnegative, low-rank angular-spatial mixture for each source/receiver pair: Z = Σ_r w·u·v. It is conditioned on first-flight statistics (A1) and on identity-free relational geometry (A2). It is normalised per source: K = dρZ/(ε+ΣZ). It is averaged over D4h (A5) and stored as sparse pair latents plus a far term (A6). The claim is that this yields transfer to held-out source positions on a finite scattering body. (FACT: `trackA_original_bgc.md`.)

**Repaired form (Track A record only; the architecture itself is rejected).** (FACT: `trackA_revised_architecture.md`.)
- K = d·ω_c·χ_D·m·z / max(1, Σ χ_D·m·z), with ρ and ε removed.
- d = 1−β, computed from a deterministic chord grid that is shared with B.
- The reciprocity condition is that d_i·z_ij is symmetric, plus a declared 16↔64 bin map.
- Corridor descriptor, clipped distances and radius-defined tie-inclusive neighbour sets.
- D4h used for training augmentation and as a predeclared test.
- Per-pair-class rank.
- A3 is removed as the transfer mechanism.

**Claimed contributions tested:**

| ID | Contribution |
|---|---|
| A1 | First-flight statistics as features |
| A2 | Relational, identity-free encoder |
| A3 | Pair-conditioned angular factors |
| A4 | Per-source energy normalisation, original and repaired |
| R | Reciprocity condition (repaired) |
| A5 | Finite-group frame averaging |
| A6 | Sparse pair latents plus a far term |
| PR | Pair-class-dependent rank (repaired) |

---

## 2. Prior-art table

"Novel?" means whether BGC's difference from that work is itself new. In every row, "evidence required" means evidence that has not been produced (FACT: nothing was measured in WL80A).

| # | Predecessor (identity, source, verification) | Exact shared mechanism | BGC's proposed difference | Why it matters | Novel? | Evidence required |
|---|---|---|---|---|---|---|
| 1 | Jensen, Marschner, Levoy, Hanrahan, "A Practical Model for Subsurface Light Transport", SIGGRAPH 2001. [graphics.stanford.edu/papers/bssrdf](https://graphics.stanford.edu/papers/bssrdf/) (V-s; separable form U) | Pair-dependent spatial weight × separable angular factors: S_d ∝ F_t(ω_i)·R_d(r)·F_t(ω_o). This is BGC's Z with R = 1, w = R_d and u, v = Fresnel lobes (DERIVATION from the standard form). Also the single-scatter plus multiple-scatter split. | R > 1, with factors generated by a network instead of an analytic diffusion profile. | Shows "pair weight × separable angular lobes" is the canonical BSSRDF structure, not a new operator class. | No. Higher rank and learned generation are generic extensions (INFERENCE). | That rank > 1 learned factors beat the rank-1 or analytic form on held-out sources. |
| 2 | Habel, Christensen, Jarosz, "Photon Beam Diffusion", CGF (EGSR) 2013. [cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html](https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html) (V-s). First-collision source method in neutron transport (U). Project FIA, WL76 (FACT, re-checked l.98–110). | The response is the first-collision density along the incident beam integrated against a post-collision kernel: K = d·E_{p1}[Q] (WL76 E0, FACT). This is A1's physical basis. | A1 replaces the exact p1 integral with moments (d, μ, Σ) fed to a network. | The moments are a lossy compression of a quantity that prior art and the project integrate exactly. Track A shows that second-order closure errors are +10.4 / +21.8 / −17.6 % (DERIVATION, checker-confirmed). | No. It is a known factorisation with a weaker, non-sufficient encoding (DERIVATION). | That moment features beat exact p1 quadrature at edge and grazing ports (Track A's A1 falsifier). |
| 3 | Vicini, Koltun, Jakob, "A Learned Shape-Adaptive Subsurface Scattering Model", TOG 38(4), 2019. [rgl.epfl.ch/publications/Vicini2019Learned](https://rgl.epfl.ch/publications/Vicini2019Learned), [code](https://github.com/rgl-epfl/learned-subsurface-scattering) (V for the conditional-VAE exit-location sampler conditioned on material properties plus local polynomial shape features; absorption model V-s via repo config and secondary slides; outgoing-direction handling not verified) | A shared network, trained across shapes, maps a local, identity-free geometric descriptor in the incident-point neighbourhood to (absorption or exit mass) × (conditional exit distribution). These are A2 and A4's mass × normalised distribution, plus cross-shape generalisation. | (a) BGC outputs a discretised port × bin operator with explicit incoming- and outgoing-angle factors. (b) Per-source normalisation to a computed d. (c) Pair-conditioning on the receiver as well as the source. (d) Exact D4h averaging. | This is the closest graphics precedent for a geometry-conditioned learned intrinsic response that generalises to unseen positions and shapes. Vicini generalises by training over many shapes; BGC tries to generalise from about 16 sources on one body (FACT, brief §2). | (a)–(d) are discretisation or contract choices, not new mechanisms (INFERENCE). Training regime (c) is the hard part, and BGC adds nothing that identifies it (Track A CE-1, DERIVATION). | A head-to-head comparison against a Vicini-style model trained over a family of shapes or charts, scored on BND_FAR at matched supervision cost. |
| 4 | Leonard, Höhlein, Westermann, "Learning Multiple-Scattering Solutions for Sphere-Tracing of Volumetric Subsurface Effects", arXiv:2011.03082. [arxiv.org/abs/2011.03082](https://arxiv.org/abs/2011.03082) (V) | Learned conditional generative models predict absorption, then exit position and direction, inside a geometric primitive (a sphere). This is a mass × conditional exit distribution over position and direction. | Pair-conditioned discrete port × bin blocks on a finite slab; energy budget d. | Learned joint exit-position-and-direction distributions are already established. | No (INFERENCE). | Same as row 3. |
| 5 | Wu, Lu, Xu, Hašan, Ramamoorthi, "8DNA: 8D Neural Asset Light Transport by Distribution Learning", SIGGRAPH 2026, arXiv:2604.25129. [arxiv.org/html/2604.25129v1](https://arxiv.org/html/2604.25129v1) (V) | F(x_o,ω_o,x_i,ω_i) = α(x_o,ω_o)·p(x_i,ω_i \| x_o,ω_o), per colour channel: survival mass × normalised conditional distribution. This is A4's form in the adjoint direction. | (a) Forward per-source normalisation bounded by d (8DNA's α is a free regression output; the paper states no ≤ 1 constraint, FACT from the HTML). (b) A geometry-conditioned shared generator instead of per-asset triplane and cubemap features. (c) A low-rank mixture instead of a spline flow. | 8DNA is the project's external neural baseline. Under AGENTS.md terminology its T1 behaviour is Neural Full Recompute when invalidated, or Native Reuse. | (a) is an elementary constraint; it is ALREADY KNOWN in the zonal and passive-macromodel literature (rows 12–13). (b) is the open problem itself, not a solution. (c) is a representation choice. | (b) needs a measured BGC-style transfer advantage over 8DNA under the mandatory baseline gate (Neural Full Recompute ≠ Cycles). |
| 6 | Kallweit, Müller, McWilliams, Gross, Novák, "Deep Scattering", TOG 2017, arXiv:1709.05418. [studios.disneyresearch.com/?p=28796](https://studios.disneyresearch.com/?p=28796) (V-s) | A hierarchical geometric descriptor relative to the shading point and the light direction feeds a shared network. The network predicts multiply-scattered radiance and generalises across unseen clouds. The features are light-oriented, first-flight-aware geometry (A1+A2). | BGC uses pairwise source–receiver context (and, once repaired, a corridor) instead of a point-light stencil. | The source-and-light-relative geometric stencil is already the established way to condition learned multiple scattering. | The corridor is PLAUSIBLY DISTINCT as a pair feature, but it is unspecified in the record, so it is UNCERTAIN (FACT: Track A marks the corridor UNRESOLVED). | A specified corridor and a measured gain on the midway-dent counterexample (Track A CE-5). |
| 7 | Zhu, Tang, Ma, "DeepRTE", arXiv:2505.23190 (v4). [arxiv.org/html/2505.23190v4](https://arxiv.org/html/2505.23190v4) (V) | A learned Green's function G(r,Ω,r′,Ω′), trained on delta-like inflows and applied by linear superposition against the boundary inflow. An attention "optical-depth" net computes first-flight attenuation e^{−τ} along characteristics. This is A1 plus the linear boundary-response operator. | BGC adds nonnegativity, passivity and geometry conditioning. DeepRTE states no positivity or energy constraint, needs retraining for a new geometry (Remark 3.1) and is tested on a square domain (FACT from the HTML). | DeepRTE is the closest learned RTE operator. It shows that first-flight attenuation features plus delta-inflow training is a published recipe. | The constraint layer is not new in itself (rows 12–13). Geometry conditioning is published in rows 8 and 20. The specific constrained combination is UNCERTAIN, and only as a combination. | Out-of-orbit transfer at matched delta-source supervision, compared with an unconstrained DeepRTE-style operator. |
| 8 | Du, Stechmann, "Element learning", arXiv:2308.02467; J. Comput. Math. 44(1), 2025. [arxiv.org/abs/2308.02467](https://arxiv.org/abs/2308.02467) (V) | One network takes the element geometry and RTE parameters and outputs a boundary-to-boundary "in2out" operator. An HDG solver composes the elements, and the abstract claims use on any domain geometry without retraining. This is BGC's geometry-conditioned learned boundary operator composed in a global solve (the K/C pattern). | Port × bin discretisation, nonnegativity and passivity, a first-exit-only K with ballistic transport separate, and D4h. | This is the closest structural precedent; it already claims geometry generalisation of a learned RTE boundary operator. | The constraints are known separately (rows 12–13 and 16). The structural mechanism is ALREADY KNOWN (INFERENCE). | That BGC's constraints change held-out accuracy relative to an element-learning-style unconstrained geometry-conditioned operator. |
| 9 | Coarse-scale and Dirichlet-to-Neumann (DtN) neural surrogates, e.g. Kröpfl, Maier, Peterseim, arXiv:2209.02624 (MMS). [arxiv.org/pdf/2209.02624](https://arxiv.org/pdf/2209.02624) (V-s), plus the ESAIM DtN-surrogate work found in the same search (V-s) | A small network maps local coefficients and geometry to a local boundary or coarse operator, which is composed globally. One DtN work enforces monotonicity of the learned maps (V-s). | Transport instead of elliptic PDEs; positivity and passivity instead of monotonicity. | The class "learned local boundary operator with an enforced structural property" is established. | No as a mechanism (INFERENCE). | — |
| 10 | Response-matrix and interface-current methods; COMET (coarse mesh transport, Rahnema group). [osti.gov/biblio/22039519](https://www.osti.gov/biblio/22039519-comet-solutions-whole-core-candu-benchmark-problems), adjoint and reciprocity: [informahealthcare 10.1080/23324309.2016.1227850](https://informahealthcare.com/doi/abs/10.1080/23324309.2016.1227850) (V-s) | Precomputed responses of a sub-domain to incident boundary flux expanded in space × angle, composed by a global interface iteration. Reciprocity relates forward and adjoint responses. The port × bin K is a discretised response matrix, and the K/C split is the interface-current architecture (DERIVATION). | The response is generated by a learned network conditioned on geometry, instead of computed per unique coarse mesh (COMET uses Monte Carlo per unique mesh). | Track B C1 is the explicit version of this lineage. BGC's only difference is that the response is learned. | The difference is the open problem, not a demonstrated mechanism (INFERENCE). | That learned responses transfer to unseen boundary contexts more cheaply than explicit per-mesh responses at matched accuracy. |
| 11 | Hottel zonal method and the exchange-factor method (Sandia, J. Heat Transfer 1985). [osti.gov/biblio/6460797](https://www.osti.gov/biblio/6460797). Larsen–Howell least-squares smoothing for summation rules (V-s via Méchi et al., [aj.tubitak.gov.tr …](https://aj.tubitak.gov.tr/engineering/issues/muh-07-31-6/muh-31-6-2-0711-2.pdf)) | (i) A discretised exchange operator between zones and surfaces of a participating medium. For an isotropic gray medium the exchange factors depend only on geometry and extinction (V-s). (ii) Per-source summation rules (energy closure), enforced by projection after fitting. (iii) Reciprocity of exchange areas (A_i F_ij = A_j F_ji). This is the exact pattern of the repaired A4 plus the R condition d_i z_ij symmetric under a measure. | A smooth max(1,·) column projection instead of least squares, bound ω·d, and port × bin angular resolution. | The repaired A4 and R are textbook radiative heat-transfer bookkeeping. | No (INFERENCE, high confidence). | — |
| 12 | Passive multiport macromodelling: Gustavsen, IEEE Trans. Adv. Packaging 2010 ([sintef.no pdf](https://www.sintef.no/globalassets/project/power-systems/publications/fast-passivity-enforcement-for-s-parameter-models-by-perturbation-of-residue-matrix-eigenvaluesjp13.pdf)); Grivet-Talocia & Gustavsen, *Passive Macromodeling* (V-s) | A fitted port scattering operator, with passivity and reciprocity enforced as constraints on the fitted model. | Power passivity for a nonnegative matrix reduces to column sums ≤ 1 (DERIVATION), so BGC's normalisation is the nonnegative special case. | "Fitted port operator plus enforced passivity and reciprocity" is a mature engineering pattern. | No (INFERENCE). | — |
| 13 | Lawrence, Rusinkiewicz, Ramamoorthi, "Efficient BRDF Importance Sampling Using a Factored Representation", SIGGRAPH 2004 ([pixl.cs.princeton.edu/proj/brdf](https://pixl.cs.princeton.edu/proj/brdf)); separable or homomorphic BRDF factorisations (McCool, Kautz; U) | Nonnegative low-rank factorisation of an angular (in × out) block with normalised factors usable as distributions. This is the u·v structure with v normalised (the repaired gauge removal). | Factors are generated per pair by a network instead of fitted per material. | The per-pair 16 × 64 nonnegative block factorisation and its gauge fixing are known. | No (INFERENCE). | — |
| 14 | Sloan, Hall, Hart, Snyder, "Clustered Principal Components for PRT", TOG 22(3), 2003 ([microsoft.com](https://www.microsoft.com/en-us/research/publication/clustered-principal-components-precomputed-radiance-transfer/)) (V-s) | Transfer matrices compressed with cluster-conditioned low-rank bases and cached per-point coefficients. This is a piecewise "conditioned low rank" with cached latents (A3 and A6). | Continuous conditioning through a network (hypernetwork) instead of discrete clusters. | Discrete-to-continuous conditioning is a standard move (row 15). | No (INFERENCE). | — |
| 15 | DeepONet hypernetwork variants: HyperDeepONet (Lee, Cho, Hwang, ICLR 2023, [arXiv:2312.15949](https://arxiv.org/pdf/2312.15949)); PI-LoRA-HyperDeepONet (2025, arXiv:2507.18346, low-rank output layer) (V-s) | A network conditioned on the input generates the parameters, or a low-rank factorisation, of the output operator. This is A3's "shared network generates pair-dependent factors". | Applied per (source, receiver) pair, with nonnegativity. | A3 is a hypernetwork low-rank head (Track A, DERIVATION). | No (INFERENCE). | — |
| 16 | Dong, Wang, Li, "Neural Parametric Mixtures for Path Guiding", SIGGRAPH 2023 ([neuropara.github.io](https://neuropara.github.io), [arXiv:2504.04315](https://arxiv.org/abs/2504.04315?context=cs)) (V-s; V-A) | A spatially conditioned neural encoding decoded into normalised convex mixtures, trained from noisy Monte Carlo. This is mass × normalised mixture conditioned on position. | Linear-operator semantics and per-source normalisation instead of guiding densities. | It shows that decoding normalised mixtures from Monte Carlo targets is established. | No (INFERENCE). | — |
| 17 | Puny et al., "Frame Averaging for Invariant and Equivariant Network Design", arXiv:2110.03336 (ICLR 2022) ([arxiv.org/abs/2110.03336](https://arxiv.org/abs/2110.03336)) (V) | Group or frame averaging gives exact equivariance. Over the full finite D4h it is the Reynolds average. This is A5 exactly. | None, except the bin-permutation action, which the project built in WL79 (FACT). | — | No. It is ALREADY KNOWN (FACT/DERIVATION). | — |
| 18 | de Haan, Weiler, Cohen, Welling, "Gauge Equivariant Mesh CNNs", ICLR 2021 ([arXiv:2003.05425](https://arxiv.org/abs/2003.05425v3)) (V-s); Sloan, Luna, Snyder, "Local, Deformable PRT", TOG 24(3), 2005 (zonal-harmonic rotation of transfer into the local or deformed frame) ([dblp](https://dblp.org/rec/journals/tog/SloanLS05)) (V-s) | Mechanisms for per-point tangent-frame gauge and continuous-rotation covariance of angular factors, including under deformation (LDPRT precomputes per patch, not per pose). | BGC (A5, repaired) covers only the finite rest D4h group. It explicitly leaves the per-port gauge (CE-7), continuous SO(2) and deformed poses unresolved (FACT, Track A). | Prior art already addresses exactly the symmetry gaps that BGC leaves open. BGC is less general than the existing equivariance mechanisms. | No. Its symmetry handling is a subset of known techniques (INFERENCE). | — |
| 19 | Melchers, Prins, Abdelmalik, "Neural Green's Operators for Parametric PDEs", CMAME 455 (2026) 118893, arXiv:2406.01857 ([arxiv.org/abs/2406.01857](https://arxiv.org/abs/2406.01857)) (V) | An explicit finite-dimensional Green's operator whose linear action is preserved exactly, while a network supplies its nonlinear dependence on coefficients. The abstract says "symmetry, spectral, and conservation properties" can be embedded, and it claims better out-of-distribution generalisation. This is the central class of BGC: a linear operator generated by a conditioning network, with structural properties built in. | Conditioning on geometry rather than PDE coefficients; nonnegativity and passivity (not mentioned in the abstract); a transport kernel with angular bins. | This is the closest neural-operator precedent for "conditioned linear operator plus embedded physical constraints". | Not novel as a mechanism. The positivity/passivity specialisation for transport is UNCERTAIN as a combination (INFERENCE). | Measured held-out transfer and constraint satisfaction relative to an NGO-style operator conditioned on the same features. |
| 20 | Zou et al., "Enforcing Reciprocity in Operator Learning for Seismic Wave Propagation" (RENO), arXiv:2602.11631, 2026 ([arxiv.org/abs/2602.11631](https://arxiv.org/abs/2602.11631)) (V). Sen et al., "Dual Photography", SIGGRAPH 2005 (V-A). | Source–receiver reciprocity built into the architecture of a learned operator (RENO), and reciprocity used to transpose transport (dual photography). | Original BGC excludes reciprocity through its independent ρ (Track A, DERIVATION). Repaired BGC only states the condition and does not build it. | Building reciprocity in is published. BGC is behind this prior art on its own repaired requirement. | No (INFERENCE). | — |
| 21 | Boullé, Townsend, "Learning elliptic PDEs with randomized linear algebra", FoCM 2022, arXiv:2102.00491 ([arxiv.org/abs/2102.00491v2](https://arxiv.org/abs/2102.00491v2)) (V-s) | Green's functions with hierarchical low-rank structure (rank of far interactions governed by separation) are learnable from input–output pairs. The error bound depends on the quality of the forcing distribution (Γ_ε). This bears on PR and on Track A's identification objection. | Repaired PR: per-pair rank by class (near / edge / far) is the admissibility idea of H-matrices applied to K. | It supports Track A CE-1: identification depends on how well the training excitations spread, and 16 one-hot sources are a narrow excitation set (INFERENCE; the theorem is elliptic, not transport). | PR is ALREADY KNOWN as H-matrix or far-field low-rank structure (INFERENCE). Its value for this medium is a measurement, not a novelty. | The per-pair nonnegative rank by class against the replica noise floor (Track A's decisive unmeasured quantity). |
| 22 | Hierarchical and far-field compression: H-matrices, FMM, hierarchical radiosity (U; V-WL77A via Track A); Modular Flux Transfer (Zhao, Hašan, Ramamoorthi, Bala, SIGGRAPH 2013, [cseweb.ucsd.edu/~ravir/mft-sg13.pdf](https://cseweb.ucsd.edu/%7Eravir/mft-sg13.pdf), V-s); Shell Transport Functions (Moon, Walter, Marschner, EGSR 2007, [cs.cornell.edu](https://www.cs.cornell.edu/~srm/publications/EGSR07-vol.html), V-s); project W+UV and FIA tails (FACT) | Explicit near interactions plus a compressed far part. Precomputed patch-to-patch or voxel flux-transfer matrices of reusable blocks, stitched at runtime (MFT); this is the reusable-intrinsic-transfer idea. | Pair latents generated by a network instead of tabulated or compressed transfer. | A6 is a cost element; its structure is standard. | No. Feasibility is UNCERTAIN (Track A, DERIVATION and INFERENCE). | Measured memory and apply time against the A75/ATO baseline on 12 GB (Track A A6 conditions). |
| 23 | Per-object or per-asset neural transport: RNA (Mullia et al., TOG 2024, [arXiv:2312.09398](https://arxiv.org/pdf/2312.09398)); NeuPreSS / "Neural BSSRDF" (TG, Frisvad, Ramamoorthi, Jensen, CGF 43(7) 2024; the arXiv:2312.15711 page lists the same DOI 10.1111/cgf.15234, so they are one work, FACT, [arxiv.org/abs/2312.15711](https://arxiv.org/abs/2312.15711)); TransGI (Deng, Han, Fang, [arXiv:2506.09909](https://arxiv.org/html/2506.09909v1)); NRC (Müller et al., TOG 40(4) 2021, [arXiv:2106.12372](https://arxiv.org/abs/2106.12372)) (V for the arXiv:2312.15711 abstract; V-s for the rest) | Reusable learned transport for an object under changing lighting (RNA, NeuPreSS). TransGI uses per-object latents plus runtime radiance sharing, i.e. the K/C class. NRC is an online nonlinear cache, not an operator (INFERENCE). | BGC aims at cross-source generalisation inside one body from sparse sources through geometry conditioning, where these methods are fitted per asset or per object. | These methods define the "per-asset neural" class whose T1 lifecycle is Native Reuse or Neural Full Recompute (AGENTS.md). | The aim is distinct, but an aim is not a mechanism. Whether BGC achieves it is NOT SUPPORTED: there is no measurement, and the project's learned-operator family fails BND_FAR (FACT, brief §3b). | Baseline-gated comparison with Neural Full Recompute (8DNA or RNA rebuilt on T1 when invalidated), the internal F/S comparator and a separate path-traced reference. |
| 24 | Project history: global ATO K = d Σ_r A_r T_ijr O_r (WL68/74/75); FIA (WL76); W+UV-Q (WL78); rest D4h maps and orbit-balanced basis (WL79/80) (FACT; WL75 l.195/308 and WL76 l.98–110 re-checked) | ATO is BGC with u and v made pair-independent. FIA already had the d-anchored first-interaction skeleton, ρ, τ, 512-NN softmax plus a tail and invariant features. W+UV-Q is A2. WL79/80 is A5. | Only A3 (pair-conditioned angular factors) is new relative to WL57–80 (Track A, re-checked). | A3 targets the factor WL75 ranks as non-binding: oracle 0.159 vs learned 0.80 at BND4_FAR (FACT). | Within the project, A3 is new but aimed at the wrong bottleneck. Externally it is a hypernetwork head (rows 14–15). | Track A's A3 falsifier: the pair-conditioned per-source oracle must beat the 0.159 G2 oracle, and the learned version must close more than half of the 0.80 → 0.159 gap. |

---

## 3. Classification

### Central mechanism M0

**ALREADY KNOWN (INFERENCE, high confidence) as an operator class.** A conditioning network that generates a linear transport response, written as mass × a normalised conditional distribution, conditioned on local or relative geometry, and composed as a boundary operator in a global solve, appears in at least four places:
- Element Learning (row 8) and Neural Green's Operators (row 19), as learned conditioned linear operators with structure;
- Vicini 2019 and Leonard 2020 (rows 3–4), as a geometry-conditioned learned exit mass and exit distribution that generalises across shapes;
- 8DNA (row 5), as α × conditional p;
- DeepRTE (row 7), as a delta-inflow Green's function with a first-flight attenuation net.

**The distinguishing claim, that this yields out-of-orbit held-out source transfer, is NOT SUPPORTED:**
- No BGC instance was measured (FACT).
- Track A's orbit-consistent bump construction shows that BGC's hard constraints do not identify that transfer (DERIVATION; checker-confirmed as generic, discriminating together with BGC's lack of cross-source sharing).
- The prior art does not supply a mechanism that would rescue it. The literature's route to generalisation is training over many shapes or geometries (rows 3, 6, 8), which the S16-R regime does not provide (FACT, brief §2), or a theory tied to well-spread excitations (row 21, INFERENCE).

### Contributions

| ID | Contribution | Classification | Basis |
|---|---|---|---|
| A1 | First-flight statistics features (collision probability, mean, covariance) | ALREADY KNOWN | Rows 2, 6, 7 and FIA. Collision probability = d (FACT, WL76). The moment compression is weaker than the known exact integral (DERIVATION). Repaired "deterministic chord d shared with B": standard collision-probability or escape-probability ray quadrature (U) plus ballistic bookkeeping. Not novel; the benefit is noise removal only (Track A checker D1). |
| A2 | Identity-free relational encoder | ALREADY KNOWN | Rows 3 and 6, EGNN-style relative features (V-A), WL78. |
| A2-corridor | Corridor descriptor (repaired) | UNCERTAIN | Pair-path geometry features are a natural extension of row 6's stencils. The record leaves the width and features unspecified (FACT), so novelty cannot be judged. |
| A3 | Pair-conditioned nonnegative low-rank angular factors | ALREADY KNOWN as a mechanism (rows 1, 13, 14, 15, 16). NOT SUPPORTED as a transfer mechanism. | At R ≥ 16 it spans the whole passive cone and w, u are a gauge (DERIVATION, Track A). It targets the non-binding factor (FACT, WL75). Removed in the repaired form. |
| A4 (original) | dρZ/(ε+ΣZ) | ALREADY KNOWN in form (rows 3, 5, 16; WL76 softmax plus ρ). Physically defective. | ρ is redundant with ΣK, produces a receiver-uniform edit artefact and excludes reciprocity (DERIVATION, Track A). |
| A4 (repaired) | ω·d bound, measure m, χ_D, max(1,S) projection, one denominator over disjoint local/far supports | ALREADY KNOWN | Zonal summation rules and smoothing (row 11), passive macromodel enforcement (row 12). The ω·d bound is an elementary derivation. |
| R | Reciprocity condition d_i·z_ij symmetric plus a bin map | ALREADY KNOWN | Exchange-area reciprocity (row 11), COMET adjoint reciprocity (row 10), RENO's architectural reciprocity (row 20). Repaired BGC only states the condition and does not build it, so it is behind RENO here. |
| A5 | Finite-group (D4h) frame averaging | ALREADY KNOWN | Row 17 exactly; maps already built in WL79 (FACT). Prior art (row 18) already handles the gauge and continuous-rotation gaps that BGC leaves open. |
| A6 | Sparse pair latents plus a separate far term | ALREADY KNOWN in structure (rows 14, 22). Feasibility UNCERTAIN. | Cost turns on the unmeasured per-pair rank (Track A CE-8). |
| PR | Pair-class-dependent rank (repaired) | ALREADY KNOWN as a principle (H-matrix admissibility, row 21). PLAUSIBLY DISTINCT only as a measurement. | Measuring the per-pair nonnegative rank for this medium has never been done (FACT, Track A). A number is not a mechanism. |
| Combination | Geometry/pair-conditioned nonnegative low-rank port × bin passive first-exit operator of a finite 3D medium, plus exact ballistic complement, plus D4h, evaluated on held-out boundary sources from 16 one-hot sources | UNCERTAIN, and a combination only | I found no single work with all elements. That is a weak negative from bounded searches (INFERENCE). Under the project's novelty discipline (Track A citing RESEARCH_CENTRIC_TOPIC.md and FNTO contract §8) and this task's rules, absence of an identical implementation does not establish novelty. |

### Where my search changes or adds to Track A's novelty record

All items below are INFERENCE unless labelled.

1. **Two prior-art lineages Track A did not cite already cover the repaired A4 + R pair as textbook practice:**
   - zonal and exchange-factor radiative transfer, which uses summation rules, reciprocity and post-fit smoothing (row 11);
   - passive multiport macromodelling (row 12).

   This strengthens "repaired A4 and R: not novel".
2. **Two neural-operator works are closer than the DeepONet and hypernetwork citations Track A used for M0:**
   - Neural Green's Operators (row 19), a conditioned linear operator with embedded symmetry and conservation;
   - RENO (row 20), architecture-enforced source–receiver reciprocity.

   Repaired BGC's "reciprocity unresolved" is solved in principle in this neighbouring literature. That counts against novelty, not as evidence that it works here.
3. **Gauge-equivariant mesh networks and LDPRT (row 18) handle the per-port frame gauge and the rotation of angular factors under deformation.** These are the A5 gaps Track A left as limitations. BGC's symmetry contribution is therefore a strict subset of known techniques.
4. **Literature-level identity correction (FACT):** "Neural BSSRDF" (arXiv:2312.15711) and NeuPreSS (CGF 2024) carry the same DOI, 10.1111/cgf.15234. They are one per-object work and should not be cited as two predecessors.

### Track B candidates in this dimension

Brief notes only; my task scope is BGC.
- **C1 CP-K:** Track B itself rates its novelty Low (collision-probability, response-matrix and COMET lineage). My rows 10–11 agree (INFERENCE).
- **C2 SBG**, a learned scalar bounded-medium Green's function G∞·τ with τ ∈ [0,1] between exact flights: shares the "Green's function × learned correction" pattern with rows 7 and 19 and the beam-integral structure of row 2. Its domain-monotonicity bound may be PLAUSIBLY DISTINCT as a constraint form, but I did not search it specifically, so it is UNCERTAIN.
- Nothing in this novelty dimension shows that Track B's rejections of C3 CCA or C4 NSM were wrong.

---

## 4. Verdict

**BGC-Transport, original and repaired, has no novel mechanism (INFERENCE, high confidence).**
- Every element maps onto published mechanisms or the project's own WL68–80 history. Rows 1–24 cover this; verification levels are given per row.
- The only element new relative to the project's history (A3) is a known hypernetwork or conditional-low-rank head. It is aimed at the factor the record ranks as non-binding (FACT, WL75 l.195/308).
- The repaired form's corrections (ω·d passivity projection, measure, reciprocity condition, deterministic budget, rest-D4h augmentation, per-class rank) are standard radiative-transfer, macromodelling or H-matrix practice. They make BGC more valid, not more novel.

**The only possible contribution is empirical:** a shared geometry-conditioned constrained operator that measurably transfers to out-of-orbit boundary sources and to deformation from tens of one-hot sources. That is currently NOT SUPPORTED: nothing was measured, and the related learned family failed BND_FAR (FACT, brief §3b).

**Evidence that would be needed before any novelty claim could even be considered.** All of it is unmeasured, none is authorised, and listing it is not a recommendation.
1. Per-pair nonnegative rank by class against the replica noise floor.
2. Held-out INT, BND_NEAR and BND_FAR gates, plus the deformation signed-change gates, against all of the following at matched supervision:
   - learned ATO (A75) and Q78;
   - an unconstrained geometry-conditioned operator (Element-learning or NGO style);
   - a Vicini-style model trained over a shape or chart family;
   - the explicit C1.
3. The A3-specific oracle test that Track A defined.
4. The projection binding fraction and the measured reciprocity defect.
5. Any cost or external comparison under the mandatory baseline gate. There, Neural Full Recompute is 8DNA or RNA rebuilt when the T1 change invalidates the asset, or Native Reuse when it stays valid. Cycles is a separate path-tracing reference, not the Neural Full Recompute baseline.

**Sources:**
- https://rgl.epfl.ch/publications/Vicini2019Learned
- https://dvicini.github.io/learned-subsurface-scattering
- https://github.com/rgl-epfl/learned-subsurface-scattering
- https://arxiv.org/abs/2011.03082
- https://arxiv.org/html/2604.25129v1
- https://arxiv.org/html/2505.23190v4
- https://arxiv.org/abs/2308.02467
- https://arxiv.org/pdf/2209.02624
- https://www.osti.gov/biblio/22039519-comet-solutions-whole-core-candu-benchmark-problems
- https://informahealthcare.com/doi/abs/10.1080/23324309.2016.1227850
- https://www.osti.gov/biblio/6460797
- https://aj.tubitak.gov.tr/engineering/issues/muh-07-31-6/muh-31-6-2-0711-2.pdf
- https://www.sintef.no/globalassets/project/power-systems/publications/fast-passivity-enforcement-for-s-parameter-models-by-perturbation-of-residue-matrix-eigenvaluesjp13.pdf
- https://pixl.cs.princeton.edu/proj/brdf
- https://www.microsoft.com/en-us/research/publication/clustered-principal-components-precomputed-radiance-transfer/
- https://arxiv.org/pdf/2312.15949
- https://neuropara.github.io
- https://arxiv.org/abs/2110.03336
- https://arxiv.org/abs/2003.05425v3
- https://dblp.org/rec/journals/tog/SloanLS05
- https://arxiv.org/abs/2406.01857
- https://arxiv.org/abs/2602.11631
- https://arxiv.org/abs/2102.00491v2
- https://cseweb.ucsd.edu/%7Eravir/mft-sg13.pdf
- https://www.cs.cornell.edu/~srm/publications/EGSR07-vol.html
- https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html
- https://graphics.stanford.edu/papers/bssrdf/
- https://studios.disneyresearch.com/?p=28796
- https://arxiv.org/pdf/2312.09398
- https://arxiv.org/abs/2312.15711
- https://diglib.eg.org/handle/10.1111/cgf15234
- https://arxiv.org/html/2506.09909v1
- https://arxiv.org/abs/2106.12372

**Repository files read (read-only):**
- C:\Projects\ReST-GI\docs\plans\wl80a\neutral_problem_brief.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackA_original_bgc.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackA_critical_audit.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackA_decision_log.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackA_revised_architecture.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackB_selected_candidate.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackB_candidate_records.md
- C:\Projects\ReST-GI\docs\worklogs\75_WORKLOG_RANK16_LEARNED_ATO_TRANSFER.md
- C:\Projects\ReST-GI\docs\worklogs\76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md

---

## C1-CPK

# WL80A cross-review: novelty kill search for Track B C1 CP-K

**Reviewer scope.** I read the sealed C1 specification (`trackB_selected_candidate.md`, the C1 record in `trackB_candidate_records.md`), the neutral brief and the novelty items in `trackB_adversarial_review.md`. I looked at the Track A revised record only for its row on the FIA family. To check internal precedents I read the Worklog 56 setup line and grepped Worklog 53 and Worklogs 76–78. I wrote no files and modified nothing. Literature identities marked "web-verified" were confirmed on an abstract, publisher or author page. Those marked "from memory" were not verified, and no specific claim rests on them. I fabricated no measurements.

## 1. What C1 claims (FACT, from the sealed record)

The amended C1 has four parts:

- **Factorisation:** `q = F_in u`, `ψ = ωW1 q + ωWψ`, `K u = ωΦ_pt q + ωΦout ψ`.
  - F_in is an exact first-flight integral that uses the same rays as the ballistic operator B.
  - Φ_pt and W1 are a point-level first escape and first continuation.
  - W is a cell flight kernel, and the escape Φout is defined as its complement.
  - A body-fitted graded mesh is used. Nothing is learned.
- **Claimed contributions** (§8 of the selection record):
  - (C-a) a port/bin first-exit contract;
  - (C-b) exact ballistic complementarity;
  - (C-c) a passivity identity that is exact at every mesh size h;
  - (C-d) exact D4h permutation equivariance;
  - (C-e) local rebuild on a body-fitted mesh;
  - (C-f) collision densities as joint unknowns with explicit C;
  - (C-g) the reframing that "unseen-source generalisation becomes a discretisation-convergence question that can be checked without references".
- **Track B's own novelty rating:** "Low … system integration".

## 2. Mechanism decomposition used for the search

| ID | Mechanism component in C1 | Where it lives |
|---|---|---|
| M1 | Peierls / collision-density integral equation for isotropic homogeneous scattering. The scalar resolvent is (I − ωW)⁻¹. | Core |
| M2 | Exact uncollided / first-collision source plus a separate ballistic part | F_in, B, S1 |
| M3 | Exact last-flight escape to boundary (port, bin) | Φ_pt, Φout |
| M4 | Flat-source cell collision probabilities computed by ray tracing in 3-D. Conservation fixes the escape as a complement. | W, Φout |
| M5 | Homogeneous kernel applied by FFT convolution with zero padding or a mask, inside an iterative solve or DSA | Solver |
| M6 | Boundary-in → boundary-out per-domain response operator, coupled to other domains through an explicit interface or free-space operator | The K/C role |
| M7 | Per-body volume discretisation that follows deformation without changing topology, with local recomputation | Dent rebuild |
| M8 | Source-agnosticism: a linear operator whose source enters only as the right-hand side, so it is "agnostic to the training functions" | Generalisation argument |
| M9 | One geometric kernel shared across channels; channels differ only by ω (an albedo-power series) | RGB |

## 3. Prior-art table

Legend for "Novel?": KNOWN = ALREADY KNOWN, DISTINCT = PLAUSIBLY DISTINCT, UNCERT = UNCERTAIN, NS = NOT SUPPORTED.

| # | Predecessor (source) | Shared mechanism (exact) | Proposed C1 difference | Why the difference matters | Novel? | Evidence needed to establish the difference |
|---|---|---|---|---|---|---|
| P1 | **Fan, An & Ying, "Fast algorithms for integral formulations of steady-state radiative transfer equation"**, J. Comput. Phys. (DOI 10.1016/j.jcp.2018.12.014); arXiv 1802.03061 (web-verified, abstract and HTML) | M1 and M5. The angle-averaged RTE becomes a second-kind Fredholm equation. In a homogeneous medium the kernel e^{−μ_t\|x−y\|}/(\|S\|·\|x−y\|^{d−1}) is a convolution, evaluated by FFT with zero padding and solved iteratively. Angular radiance is recovered afterwards by a scattering-free transport solve. This is the same algebra as C1's ψ-solve followed by Φout. | (i) Inflow from a boundary packet (F_in) instead of a vacuum boundary with an interior source. The paper says "we only consider the vacuum boundary condition" (FACT, web-verified); its §4 on general BCs was not read. (ii) Output tallied into port/bin first-exit flux. (iii) Used as a stored or re-solved reusable operator. | (i) and (ii) decide whether the contract metrics are reachable. They do not change the numerical method. | **KNOWN** for M1 + M5, including C1 option (U), the masked-FFT lattice. Inflow-as-first-collision-source is a textbook combination (see P4). | None for novelty. For C1's validity, the evidence is the predeclared refinement study. |
| P2 | **Ren, Zhang & Zhong, "A fast algorithm for radiative transport in isotropic media"**, arXiv 1610.00835 (web-verified, abstract only) | M1 and M3. Two steps: (1) an integral equation for the angularly averaged density, solved by FMM, with an FFT preconditioner for nearly homogeneous media (per a search snippet; not checked in the full text); (2) a scattering-free transport solve that recovers the angular solution. | Same as P1. C1 also uses a graded body-fitted mesh instead of FMM or a regular grid. | Graded or boundary-layer meshing is the accuracy lever for BND_FAR. | **KNOWN** (core). Graded meshing for integral transport is a discretisation choice (INFERENCE). | — |
| P3 | **Collision-probability methods in 3-D with ray tracing**: Roy, Hébert & Marleau (1989, ray-tracing CP for arbitrary 3-D geometries, per a search snippet); Marleau & Hébert, NSE 92 (1986), interface-current CP; DRAGON CP / response-matrix driver; "A Numerical Method for Computing Collision, Escape, and Transmission Probabilities in Three Dimensions", NSE (2003) (web-verified titles and abstracts) | M1, M3 and M4. Flat-source region-to-region collision probabilities, escape and transmission probabilities computed by ray tracing in 3-D. Conservation theorems link volume–volume CPs to escape probabilities, and reciprocity relations hold (FACT, patent snippet). The CP literature uses these relations to normalise. | (i) Escape resolved into 2.5 mm ports × 64 bins. Reactor interface currents usually use low-order angular expansions (DP0/DP1; from memory). (ii) Point-level first continuation W1 instead of a cell-level first collision. (iii) Complement-defined escape used deliberately to keep passivity at finite quadrature. | (i) The fine angular escape resolution is what makes con_* reachable. (ii) and (iii) are accuracy and consistency refinements. | (i) **UNCERT**: higher-order DPn interface currents exist (from memory); the port/bin choice is a resolution parameter, not a mechanism. (ii) and (iii) **KNOWN** in substance: they are conservation-normalised CPs plus a first-collision source (P4). | For (i): show that no CP or response-matrix code resolves escape at this angular/spatial density. A literature check of DRAGON's EXCELL and of DPn interface-current papers is still open. |
| P4 | **First-collision source**: Alcouffe (1985), MC first-collision source coupled to S_N (web-verified, OSTI abstract); later deterministic uncollided-flux FCS codes (2019 DOCTORS framework, web-verified snippet) | M2. The uncollided flux is computed exactly or analytically and is used as the source of the collided solve. This removes ray effects and the steep near-source gradient. | Deterministic, using the same rays as B, so Σ F_in = d holds to float precision. Point-level S1 instead of a cell-averaged first source. | Exact budget complementarity with B is required by the brief's contract. | **KNOWN**. Sharing rays to obtain a complementary split is an implementation detail (INFERENCE). | — |
| P5 | **Zonal method**: Rushmeier & Torrance, SIGGRAPH 1987 (web-verified); **Sillion**, "A unified hierarchical algorithm for global illumination with scattering volumes and object clusters", IEEE TVCG 1(3), 1995 (web-verified) | M1, M4 and M6/C-f. Volume zones of a homogeneous isotropic medium exchange energy through volume–volume, volume–surface and surface–surface exchange factors. The zone densities are joint unknowns with surface radiosities in one global solve, which is C1's "joint collision-density unknowns with explicit C". Sillion adds a hierarchy and clustering. | Directional (16/64-bin) boundary ports instead of Lambertian surfaces. The volume is owned as a reusable intrinsic operator, and only first exit belongs to K. | The ownership split is a contract choice (ReST-GI's K/C). Directional boundary resolution is required for con_*. | **KNOWN** for the joint K+C solve and volume-zone exchange (C-f). The directional boundary is **UNCERT**: whether directional zonal or radiosity-with-volumes variants exist was not searched to completion. | Show that no zonal or volumetric radiosity variant uses directional boundary exitance. A focused search is still open. |
| P6 | **Translucent Radiosity**: Sheng, Shi, Wang & Narasimhan, IEEE TVCG 20(7), 2014; I3D 2013 (web-verified) | M6. An intrinsic per-object subsurface scattering matrix is combined with the form-factor (free-space) matrix in one radiosity solve. The matrix may come from analytic, measured or simulation models, including homogeneous and heterogeneous ones. This is the graphics K/C split. | C1's K is exact collided transport with directional port bins, built deterministically, and comes with deformation rebuild. Sheng et al. are diffuse-only. | C1 gives fidelity at the boundary layer and in direction. The split itself is shared. | **KNOWN** for the K/C structure. A **DISTINCT** choice of the intrinsic matrix's construction and resolution, which is not a new mechanism. | Novelty would have to rest on accuracy or cost results, not on the structure. |
| P7 | **Lensch et al., "Interactive Rendering of Translucent Objects"**, PG 2002 / CGF 22 (2003) (web-verified) | M6. A precomputed per-object boundary-to-boundary transport: vertex-to-vertex throughput factors (global part) plus local filter kernels. Rigid object, dynamic lighting. | Exact integral transport, with ports and bins instead of a diffusion BSSRDF. Deformable. | Same as P6. | **KNOWN** for the reusable per-object boundary operator. | — |
| P8 | **Volumetric SSS solves on body-fitted grids**: Wang et al., polygrid diffusion (MSRA, ACM TOG, around 2008; per a web snippet, the exact citation was not confirmed); **Wang, Wang, Holzschuch, Subr, Yong & Guo, CGF 29(2), EG 2010** (QuadGraph from a tetrahedralisation; "handles deformation provided it does not change the topological structure"; web-verified); **Arbree, Walter & Bala, IEEE TVCG 17(7), 2011**, FEM (web-verified) | M7 and M6. Interior transport is solved at runtime from boundary illumination on a volume discretisation fitted to the object. The 2010 paper handles deformation that keeps topology, which is C1's "same topology deformed by a stretch map, no cut cells" (C-e). | Integral transport (Peierls) instead of the diffusion approximation, with point-exact low orders. | Diffusion fails in the 1–3 mfp boundary layer, which is exactly the BND_FAR regime. C1's integral form does not make that approximation (DERIVATION). | C-e **KNOWN**: a topology-preserving body-fitted mesh under deformation. Switching from diffusion to integral transport is a known substitution (P1–P3). | — |
| P9 | **Modular Flux Transfer**: Zhao, Hašan, Ramamoorthi & Bala, SIGGRAPH 2013 / ACM TOG (web-verified) | M3, M4 and M6. Per-exemplar-block voxel-to-voxel, patch-to-patch and patch-to-voxel flux-transfer matrices are precomputed. Low orders are computed accurately at render time, and higher orders go through block transfer. This is structurally F_in / W / Φout, with boundary patches as ports. | One body instead of tiled exemplars. Exact passivity at every h. Directional bins. | MFT targets approximate high-order transport. C1 claims exactness to discretisation. | **KNOWN** structure. C1's guarantees are a property of a sub-stochastic discretisation (see §4, C-c). | — |
| P10 | **Shell transport functions**: Moon, Walter & Marschner, EGSR 2007; Müller, Papas, Gross, Jarosz & Novák, TOG 35(6), 2016; **macro Monte Carlo** (Neuenschwander; Svatos 1998, LLNL thesis) (all web-verified) | M6 and M8. Precomputed local response operators are reused for any source position by construction. | C1 recomputes per body and does not rely on local shells. | They show that the reuse of a source-independent local response is old. | **KNOWN** (M8 in a precomputed form). | — |
| P11 | **Response-matrix / interface-current methods and COMET**: "The Response Matrix Method" (Springer chapter, 1981; the Lindahl & Weiss authorship is unverified); COMET incident-flux response expansion (web-verified via NSE / Georgia Tech records; the original Mosher & Rahnema paper was not retrieved) | M6. Each coarse node has a precomputed boundary-in → boundary-out (plus interior) response, coupled through interface currents and reused when coupling changes. Adjoint COMET states reciprocity relations between forward and adjoint responses (web snippet). | In C1, coupling is through free space (C) rather than shared interfaces, and K is rebuilt deterministically under geometry change. | It changes the coupling operator, not K. | **KNOWN**: a reusable boundary response operator. | — |
| P12 | **Chandrasekhar H-function, invariant embedding, adding–doubling** (van de Hulst) (from memory) | M6 in plane-parallel form. The slab reflection/transmission operators R(μ, μ₀), T(μ, μ₀) are an exact boundary-in/boundary-out K. | A finite 3-D body with edges and corners. | Edges and corners are where BND_FAR lives. | **KNOWN** for the operator concept. The 3-D finite-body extension is covered by P1–P3. | — |
| P13 | **Hašan & Ramamoorthi, "Interactive Albedo Editing in Path-Traced Volumetric Materials"**, ACM TOG 32(2), 2013 (web-verified) | M9. In homogeneous media, per-order contributions scale as ωⁿ. A precomputed albedo-to-intensity mapping is used. | C1 shares W across RGB and changes only ω. | Cost: one geometric kernel for three channels. | **KNOWN**. | — |
| P14 | **DeepRTE**: Zhu, Tang & Ma, arXiv 2505.23190 (web-verified, abstract and summaries) | M8, learned. It learns the RTE Green's function under linearity in the inflow. Pre-training on delta inflow is claimed to give zero-shot transfer to unseen boundary conditions. | C1 uses no learning: the Green's function is solved exactly, not learned. | This is the same structural argument as C1's generalisation argument (source enters only through linearity in the boundary data). DeepRTE stakes it for a learned operator. | M8 **KNOWN** as an argument. C1's non-learned realisation is P1–P3. | — |
| P15 | **Neural Green's Functions**: Yoo, Yeo, Hwang & Sung, NeurIPS 2025, arXiv 2511.01924 (web-verified, abstract). **Boullé & Townsend**: learning Green's functions with randomised SVD and H-matrix structure, FoCM 2022 (web-verified via search) | M8. "Because the Green's function depends only on geometry, the framework is … agnostic to the specific functions used during training" (paraphrase of the abstract). | C1 computes rather than learns. Different PDE class (RTE versus eigendecomposable elliptic PDEs). | Same as P14. It is the published form of C1's claim C-g. | C-g **KNOWN**. | — |
| P16 | **Element learning**: Du & Stechmann, arXiv 2308.02467 (web-verified, abstract) | M6, learned, for the RTE. Per element, a network maps element geometry and parameters to an "in2out" boundary-to-boundary operator and an "in2sol" Green's operator. Elements communicate through the in2out maps. The authors claim it works on any domain geometry without retraining. | C1 is exact and not learned. Its coupling is free space rather than element interfaces. | This is the closest predecessor of **C2 SBG** (a learned bounded Green's function inside a domain split) and of the learned-K question C1 sidesteps. | For C1: not a threat (C1 is not learned). For C2: weakens C2's novelty (INFERENCE). | — |
| P17 | **Neural relightable assets**: RNA (Mullia, Luan, Sun & Hašan, ACM TOG 2024; arXiv 2312.09398); **8DNA** (Wu, Lu, Xu, Hašan & Ramamoorthi, arXiv 2604.25129, SIGGRAPH 2026); **TransGI** (Deng, Han & Fang, arXiv 2506.09909, IEEE TVCG 2025); **NeuPreSS** (Thomson, Frisvad, Ramamoorthi & Jensen, CGF 43(7), 2024); **Neural PRT** (Rainer et al., CGF 41(2), 2022) (all web-verified at abstract level) | M6, learned. A per-asset transport operator: far-field 6-D in RNA, a full near-field 8-D operator in 8DNA, object-centric transfer under rigid motion in TransGI, distant-light SSS in NeuPreSS, PRT in Neural PRT. | C1 is explicit, exact to discretisation, passive by construction, and rebuildable under deformation without retraining. 8DNA's abstract does not address deformation or inter-asset composition (web-verified absence in the abstract only). | This is the relevant comparator axis for ReST-GI's research claims. It is a lifecycle and fidelity difference, not a representational novelty. | **DISTINCT** from these, but only because C1 is classical deterministic transport. The distinction is not a new mechanism. | Any advantage needs a comparison that obeys the AGENTS.md / `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` gate. For 8DNA/RNA under a geometry edit: Native Reuse if the T0 asset stays valid, Neural Full Recompute only on actual invalidation, F/S as a separate comparator, and a separate path-tracing reference. I am not designing that comparison here. |
| P18 | **Learned or geometry-aware SSS samplers**: Vicini, Koltun & Jakob, TOG 38(4), 2019 (local polynomial SDF → learned BSSRDF sampler); Leonard, Höhlein & Westermann, CGF 40(2), 2021 (learned sphere-tracing MS); Kallweit et al., "Deep Scattering", TOG 2017 (geometry descriptor → radiance); NRC, Müller et al., TOG 40(4), 2021 (online cache) (all web-verified) | They condition on local geometry to handle edges and corners: Vicini explicitly covers "curvature, thickness, corners". | C1 resolves corners exactly by the true geometry in W and Φout rather than by a learned descriptor. | Corner and edge contexts are the BND_FAR failure strata. | Not a novelty threat to C1. They show that learned geometry-conditioned SSS is an established, approximate route. | — |
| P19 | **Internal ReST-GI precedents** (FACT, repo): the WL53/55/56 oracle K. It is MC entry → first exit with 2.5 mm cells and 16/64 bins, "2048 photons per input cell", passive, with disjoint K/C ownership. AGENTS.md records it as composing stably. WL76's "first-interaction factorization": exact single-scatter escape E1, "factorisation itself exact", deployable first-collision terms conserve the budget. The Track A revised record lists "(R-a) exact first-collision anchoring with a learned source-independent Q: the FIA family". | M2, M3, M6 and M8 are already in the project. The WL56 oracle is a source-exhaustive, non-learned K under the same contract, so it already has C1's "no source memorisation" property. WL76 already contains C1's F_in and S1 path. | C1 replaces (a) the MC photon oracle with a deterministic CP discretisation, and (b) WL76's learned Q with an explicitly solved resolvent plus escape. | The difference is **cost and noise**. The brief records the oracle at "minutes per configuration (190 s)", against C1's ESTIMATED 1–10 s. It is not a new generalisation mechanism. | C-g **NOT SUPPORTED** as new even within ReST-GI. The deterministic-for-MC substitution is DISTINCT only as an engineering choice. | A measured accuracy/cost comparison of C1 against the existing MC oracle at matched error, and a check of C1 against the WL76 single-scatter path. Neither exists. |

## 4. Classification of each claimed contribution

| Claim | Classification | Rationale |
|---|---|---|
| **Central mechanism**: the exact Peierls / CP factorisation (first flight, scalar resolvent, last flight) used as K | **ALREADY KNOWN** | P1–P4 give the identical algebra and solver family. P5, P9 and P11 give the role of a reusable or jointly solved boundary/volume operator. FACT (sources) plus INFERENCE (identity of mechanism). |
| C-a: port/bin first-exit contract | **NOT a contribution of C1** | The contract is given by the brief (CONTRACT). Resolving CP escape at 16/64 bins is a resolution choice. Any novelty of directional CP escape at this density is **UNCERTAIN** (P3(i), P5 search incomplete). |
| C-b: exact ballistic complementarity, Σ F_in = d | **ALREADY KNOWN** | This is the uncollided/collided split of first-collision-source methods (P4). Sharing rays with B is an implementation detail. |
| C-c: passivity identity exact at every h, via complement escape and consistent W1 | **ALREADY KNOWN** | DERIVATION: the amended form is a sub-stochastic absorbing Markov chain. Every cell or point row partitions unit mass into "collide somewhere" plus "escape", and 1 − ω is killed per collision. So output = ω1ᵀq − (1−ω)1ᵀψ is the standard neutron balance (absorption + leakage = source). Defining escape as a complement is the conservation-relation normalisation of CP practice (P3; the patent snippet states the conservation theorems). Getting it right after the original split form failed is careful engineering, not novelty. |
| C-d: exact D4h permutation equivariance | **ALREADY KNOWN** | A deterministic operator assembled on a symmetric mesh with orbit-symmetrised quadrature inherits the symmetry exactly. Exploiting symmetry is routine in reactor lattice codes (from memory). It contrasts with the learned operators of brief §3b, but that is not prior-art novelty. |
| C-e: local rebuild on a body-fitted topology-preserving mesh | **ALREADY KNOWN** (in substance) | Wang et al. (EG 2010) handle deformation that preserves topology on an object-fitted graph (P8). Arbree et al. use FEM on a body-fitted mesh. Local recomputation of CP rows for the changed region is a direct consequence of locality. Whether it suffices at the 0.35 / 0.90 signed-change gates is HYPOTHESIS (C1's own §9.2). |
| C-f: collision densities as joint unknowns with explicit C | **ALREADY KNOWN** | This is the zonal method (P5) and translucent radiosity (P6). The joint solve is also undesigned in C1 itself (FACT, §6). |
| C-g: "unseen-source generalisation becomes a discretisation-convergence question checkable without references" | **NOT SUPPORTED** as a novel contribution | (i) Any deterministic linear solver has this property. Solution verification by refinement is standard practice (INFERENCE). (ii) It is the stated rationale of DeepRTE and Neural Green's Functions (P14, P15). (iii) Inside ReST-GI, the existing MC oracle K already has it (P19). C1's own record concedes it replaces the learned-K question rather than answering it. |
| Integration of all of the above as a reusable intrinsic K in a dynamic K/C GI split | **UNCERTAIN** | Each component is known, and the task rules exclude novelty from arrangement alone. The specific combination (an exact integral-transport K at directional port/bin resolution, rebuilt on deformation, coupled with an explicit free-space C) was not found as a single published system in my search. That absence is not evidence of novelty. Whatever value it has is **empirical**: accuracy on BND_FAR and dents, and cost below re-rendering. None of that is measured. C1's own estimates make it probably not cheaper than re-rendering under deformation. |

**Overall verdict on C1 novelty: ALREADY KNOWN mechanism.** C1 is a correct and well-amended instance of classical integral-transport / collision-probability numerics (FACT for the cited sources; INFERENCE for the mapping). Its only candidate for a distinct contribution is an engineering result that does not yet exist: a deterministic replacement of the project's MC oracle K that reaches the contract gates on BND_FAR and dent strata at a cost competitive with re-rendering. If that result were ever measured, it would be **evidence for an operating point**, not a new representation (INFERENCE).

## 5. Was any Track B or Track A rejection or rating wrong on novelty grounds?

- **Track B's C1 rating "Low novelty"** is confirmed, and it is somewhat generous on C-g. P1 and P2 (integral RTE with FFT/FMM in isotropic homogeneous media) were missing from Track B's lists, and they are a closer computational match than the CP or zonal citations Track B gave. The internal precedent P19 was also missed; Track B was blind to the repository by design.
- **C2 SBG ("narrow" novelty).** Element learning (P16), DeepRTE (P14), Neural Green's Functions and Boullé–Townsend (P15) cover the learned-Green's-function-inside-a-domain-split idea. What remains of C2 is the domain-monotone factorisation G_Ω = G∞·τ with τ ∈ [0,1] and its geometry-only energy projection. I found no predecessor for that specific constraint, so it stays **UNCERTAIN**. It was not searched exhaustively, and the constraint is an elementary consequence of path-subset monotonicity (DERIVATION, per Track B).
- **Original BGC and Track A revised BGC.** Track A rejected BGC as a source-generalisation mechanism. Nothing in this search changes that. The surviving Track A item, R-a (exact first-collision anchoring with a learned Q, the FIA family), is the learned counterpart of C1. Its first-collision part is P4/P19 prior art.
- **C3 / C4.** Nothing found here reverses their rejection. C4's collision-order split is covered by Modular Flux Transfer and Hašan & Ramamoorthi 2013, as Track B noted. C3's context tables are close to MC-tabulated empirical BSSRDF / response-library practice.

## 6. Limits of this search

- Several identities are **from memory only**: Case & Zweifel, Bell & Glasstone, DSA (Alcouffe 1977, Adams & Larsen 2002), DPn interface currents, van de Hulst adding–doubling, and Lindahl & Weiss authorship.
- I did not read the full texts. In particular, I did not check Fan et al. §4 (general BCs), Ren et al.'s boundary treatment, 8DNA's parameterisation and deformation support, or directional zonal variants.
- The two **UNCERTAIN** items most worth closing with a targeted literature check:
  - CP or interface-current codes with escape resolved at high angular order on fine boundary patches;
  - volumetric zonal or radiosity methods with directional (non-Lambertian) boundary exitance.
- I made no measurement, execution or file modification.

## Sources

**Integral RTE and collision-probability methods**
- [Fan, An & Ying, arXiv 1802.03061](https://arxiv.org/abs/1802.03061), [HTML](https://arxiv.org/html/1802.03061)
- [Ren, Zhang & Zhong, arXiv 1610.00835](https://arxiv.org/abs/1610.00835)
- [3-D collision/escape/transmission probabilities, NSE](https://www.ans.org/pubs/journals/nse/article-2353/), [OSTI record](https://www.osti.gov/biblio/20804864-numerical-method-computing-collision-escape-transmission-probabilities-three-dimensions)
- [Marleau & Hébert interface-current CP, NSE](https://www.ans.org/pubs/journals/nse/article-18171), [DRAGON CP/response-matrix driver](https://proceedings.cns-snc.ca/index.php/pcns/article/view/1036), [Interface current vs MC in 3-D](https://proceedings.cns-snc.ca/index.php/pcns/article/view/1345)
- [CP conservation/reciprocity (USPTO 5267276)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5267276)
- [Alcouffe first-collision source (OSTI)](https://www.osti.gov/biblio/5817627), [DOCTORS framework arXiv 1910.11335](https://arxiv.org/pdf/1910.11335)
- [COMET-related NSE (Zhang & Rahnema)](https://www.ans.org/pubs/journals/nse/article-35439/), [Forget thesis (core.ac.uk)](https://core.ac.uk/download/4692906.pdf), [Response Matrix Method chapter (Scilit)](https://scilit.com/publications/5453aa4aaa6cf3b407603c9f2dac6b30)
- [Macro response MC (OSTI)](https://www.osti.gov/biblio/3847)

**Graphics: volumetric and subsurface transport**
- [Rushmeier & Torrance 1987 (SIGGRAPH history)](https://history.siggraph.org/?p=116086)
- [Sillion 1995 (HAL)](https://hal.archives-ouvertes.fr/inria-00509994)
- [Translucent Radiosity (CMU RI)](https://www.ri.cmu.edu/publications/translucent-radiosity-efficiently-combining-diffuse-inter-reflections-and-subsurface-scattering)
- [Lensch et al. (EG DL)](https://diglib.eg.org/handle/10.2312/8744)
- [Wang et al. EG 2010 (EG DL)](https://diglib.eg.org/handle/10.2312/CGF.v29i2pp497-506), [INRIA page](https://artis.inrialpes.fr/Publications/2010/WWHSYG10/)
- [Arbree, Walter & Bala (Cornell)](https://www.cs.cornell.edu/~kb/projects/heterogeneousSS)
- [Modular Flux Transfer PDF](https://cseweb.ucsd.edu/%7Eravir/mft-sg13.pdf), [Crossref](https://api.crossref.org/works/10.1145%2F2461912.2461938)
- [Moon, Walter & Marschner 2007](https://www.cs.cornell.edu/~srm/publications/EGSR07-vol.html), [Müller et al. 2016](https://cs.dartmouth.edu/wjarosz/publications/muller16efficient.html)
- [Modular Radiance Transfer](https://cs.dartmouth.edu/wjarosz/publications/loos11modular.html)
- [Hašan & Ramamoorthi 2013 (dblp)](https://dblp.org/rec/journals/tog/HasanR13)

**Learned operators and Green's functions**
- [DeepRTE arXiv 2505.23190](https://arxiv.org/abs/2505.23190v2)
- [Neural Green's Functions arXiv 2511.01924](https://arxiv.org/abs/2511.01924)
- [Element learning arXiv 2308.02467](https://arxiv.org/abs/2308.02467)
- [Boullé–Townsend arXiv 2102.00491](https://pdf.arxiv.org/pdf/2102.00491)

**Neural assets, neural SSS and caching**
- [RNA arXiv 2312.09398](https://arxiv.org/pdf/2312.09398)
- [8DNA arXiv 2604.25129](https://arxiv.org/abs/2604.25129)
- [TransGI arXiv 2506.09909](https://arxiv.org/pdf/2506.09909)
- [NeuPreSS (EG DL)](https://diglib.eg.org/handle/10.1111/cgf15234)
- [Neural PRT (EG DL)](https://diglib.eg.org:443/handle/10.1111/cgf14480)
- [Vicini, Koltun & Jakob 2019](https://rgl.epfl.ch/publications/Vicini2019Learned)
- [Leonard et al. 2021](https://diglib.eg.org/handle/10.1111/cgf142623)
- [Deep Scattering arXiv 1709.05418](https://arxiv.org/pdf/1709.05418)
- [NRC (NVIDIA)](https://research.nvidia.com/labs/rtr/publication/muller2021nrc)

**Internal ReST-GI sources (read-only)**
- `C:\Projects\ReST-GI\docs\plans\wl80a\trackB_selected_candidate.md`
- `C:\Projects\ReST-GI\docs\plans\wl80a\trackB_candidate_records.md`
- `C:\Projects\ReST-GI\docs\plans\wl80a\trackA_revised_architecture.md`
- `C:\Projects\ReST-GI\docs\worklogs\56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md`
- `C:\Projects\ReST-GI\docs\worklogs\76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md`

---

## C2-SBG

# WL80A cross-review: novelty kill search for Track B C2 SBG

**Scope and basis.**
- **Sealed records read:** `trackB_candidate_records.md` (C2 spec), `trackB_adversarial_review.md` (both C2 reviews, the developer response and the amended spec), `trackB_selected_candidate.md` (chair matrix and verdict), and the relevant part of `neutral_problem_brief.md`.
- **Repo read-only check:** `docs/plans/wl77a/` (ideation, `coordinator_decision.md`, `review_B_novelty_systems.md`).
- **Web search this session.** Verification labels used below:
  - **[V]**: I checked the source's abstract or page this session.
  - **[V-repo]**: verified with a URL in an earlier sealed project record.
  - **[M]**: from memory, not verified this session.
- Nothing was implemented, run or measured, and no file was written.

**Object under review.** I review the *amended* C2. The original form (Ĝ = vis·G∞·τ with product composition) was conceded wrong by its developer. Amended C2:
- `K_c = ω S1 + ω² Φout Ĝ_c F_in`
- `Ĝ = vis·W∞ + Σ_k ½(G_k^(≥2) D_k + D_k G_k^(≥2))`
- Chart kernels:
  - slab: exact;
  - semi-slab: exact per mode, or a learned ratio;
  - quarter-slab: learned log-ratio τ_edge ∈ [0,1], supervised by MC;
  - dent: nested ratio τ_dent = G_dent^(≥2)/G_rest^(≥2) with pair-frame descriptors.
- The D2 geometry-only column projection.
- Corner composition by product, which was withdrawn and kept only as a secondary hypothesis.

Where it matters, I also point to the original claims.

---

## 0. Headline findings

1. **FACT (repo, `docs/plans/wl77a/ideation.md:123-149`, `coordinator_decision.md:14`). SBG's central mechanism was already proposed in this project and rejected.**
   - WL77A H2 "RBG" was `K_scat = SS_det + g_X⁻¹ Π_Yᵀ Γ_θ Π_X`. It has exact first-collision and escape projections Π at both ends, and a learned, direction-free, source-independent, symmetric scalar resolvent Γ_θ on element pairs within about 15 mm.
   - The coordinator rejected it on objections A4, A5, A7, A12, B6 and B7. One stated reason was "the mechanism is covered by Neural Green's Functions / Element Learning / DeepRTE".
   - Track B was blind to this, and its records never mention it. INFERENCE: SBG is a re-derivation of H2 with four real modifications:
     - an analytic or exact-slab base kernel with a monotone ratio;
     - chart-space supervision instead of source-packet supervision;
     - the D2 projection instead of max-normalisation;
     - graded sub-mfp depth instead of 2.5 mm hat functions.
   - Those modifications answer some of H2's *validity* objections: footprint aliasing, and the near-diagonal Γ at unseen plane configurations being "identified by smoothness only". They do not answer the *novelty* objection B7, which applies to SBG unchanged.

2. **DERIVATION: the D2 "exact energy totals" claim is overstated.**
   - The developer's own identity is `uᵀG = (1 − P_esc)ᵀ` with `u = (1−ω)·1 + ω·P_esc`.
   - That gives **one** linear constraint per column, on a mixture of absorbed and escaped collision weight. The quantity that sets the output total is `P_escᵀ Ĝ[:,x]`, and it is pinned only if Ĝ also has the correct absorption-to-escape split, i.e. the correct `1ᵀĜ[:,x]`.
   - Scaling a column to satisfy D2 therefore guarantees **passivity**, because each term is non-negative. It does **not** guarantee the exact output total.
   - Magnitudes (INFERENCE, using the records' red slab collision count of about 38 and blue of about 8):
     - Red: (1−ω)·1ᵀG ≈ 0.04, so D2 nearly pins the escape total.
     - Blue: (1−ω)·1ᵀG ≈ 0.8 dominates the constraint, so D2 says little about blue output.
   - "Exact per-bin energy" still needs the adjoint solve w* (the original remedy ii). That solve needs a C1-type W, so it does not escape the dependence on C1.
   - The Track B chair's matrix row "Passivity: exact via D2" is correct. The developer's "makes the energy totals exact" (rebuttal to R2-8) is NOT SUPPORTED.

3. **INFERENCE: once the claims SBG has itself withdrawn are removed, every remaining contribution is ALREADY KNOWN as a principle.**
   - The withdrawn claims are corner transfer without corner data, and G∞ as a useful bound.
   - What remains distinct is a parameterisation device (a monotone nested ratio used as a hard output constraint) and an application-level combination.
   - No contribution is a new transfer mechanism. Its own developer reduced the learned part to compression or amortisation (D3, verdict "MARGINAL"), so novelty would have to be argued on a **cost** claim, and that claim is unmeasured.

---

## 1. Prior-art table (mechanism level)

Columns: **Shared** = the exact shared mechanism; **Δ** = the difference SBG proposes; **Novel?** = whether that difference is novel.

| # | Work and source | Shared mechanism | Δ claimed by SBG | Why Δ matters | Novel? | Evidence needed |
|---|---|---|---|---|---|---|
| P1 | **Collision-probability / Peierls integral transport** (reactor physics). Definition, sum rule Σ_j P_ij + P_out = 1 and reciprocity [V, tutorial doc: https://deoliveira-r-orpheus.readthedocs.io/en/latest/theory/collision_probability.html]. "Collision probability methods – an update" [V, title only: https://www.osti.gov/biblio/6529305]. | First flight → scalar collision resolvent → escape. Isotropic re-emission makes the collision density the complete state. Uses the escape complement. | Replaces the solved resolvent with a learned non-iterative Green's function on top of exact flights. | It removes the inner solve, and the error enters once instead of being amplified about 38× in red. | **No** for the sandwich (FACT: classical). The non-iterative learned resolvent is the only change; see P5–P9. | Not applicable: the sandwich is not claimable. |
| P2 | **First-collision-source method** (Gomes & Stevens, NSE 1993; RAY3D / DONTRAN3D) [V-repo: `wl77a/specialist_priorart.md:240`]. | Exact uncollided ray integral, then a first-collision source, then a separate collided solve. | The D1 split `vis·W∞ + G^(≥2)` puts visibility only in the exact single-flight term. | It fixes the original dent visibility error. | **No.** D1 is the first-flight / multi-flight split of the Neumann series, which is standard. | None. |
| P3 | **Zonal method**, Rushmeier & Torrance, SIGGRAPH 1987, pp. 293–302 [V: https://history.siggraph.org/?p=116086]. | Isotropic-scattering volume zones with geometric exchange factors, solved as a radiosity-like linear system. | Learned resolvent instead of a solve; exact angular port/bin ends. | This is the graphics ancestor of the volume-to-volume scalar kernel. | **No** for the volume-zone scalar kernel. | None. |
| P4 | **Photon Beam Diffusion**, Habel, Christensen & Jarosz, CGF 32(4) (EGSR 2013) [V: https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html]. | An exact first-collision beam integrated against a bounded-medium kernel. Exact single scattering is added separately. The paper notes corrections to diffusion inaccuracy are possible. | Transport-level instead of diffusion-level kernel; learned and bounded; exact escape projection at the exit. | The FIA/SBG sandwich structure is PBD's structure with exact end flights. | **No** for "exact beam × kernel + exact SS". The transport-level kernel is the Δ; see P10. | None for structure. |
| P5 | **DeepRTE**, Zhu, Tang & Ma, arXiv 2505.23190 [V: html v4]. | A learned RTE Green's function. The solution is linear in the inflow, I = ∫G_NN·I₋, trained on delta inflows, so the learned object never sees a specific source. | SBG keeps the flights exact and learns only a scalar post-collision kernel. Geometry varies through charts. DeepRTE learns attenuation as well and, per its Remark 3.1, is tied to a fixed geometry. It states no positivity or passivity structure. | Exact flights plus a non-negative, passive structure are exactly what learned operators lacked in this project's history. | **Partly.** "Source-free learned RTE Green's function" is ALREADY KNOWN. Exact flights plus a scalar learned middle under an isotropic phase: PLAUSIBLY DISTINCT, but as a restriction of scope rather than a mechanism. | A head-to-head test on unseen sources and unseen charts against a learning-free chart solve, not against a source-conditioned baseline. |
| P6 | **Neural Green's Functions**, Yoo, Yeo, Hwang & Sung, NeurIPS 2025, arXiv 2511.01924 [V: html v1]. | A geometry-only network (volumetric point cloud → Transolver → per-point features) predicts a factorised Green's matrix G_θ = (KΦ_θ)(KΦ_θ)ᵀ. It never receives f or h, so it is source-agnostic by design. Evaluated on unseen sources and **unseen 3-D shapes**. | Integral transport instead of elliptic Dirichlet PDEs. Exact angular end flights. A monotone ratio to an exact bounded base. Charts instead of a whole-shape encoder. | NGF already establishes the principle SBG relies on for generalisation: learn only a geometry-conditioned Green's function, never the source. | **No** for the principle (this is H2's B7 objection, upheld in WL77A). The domain (RTE, exact flights, passivity) is application-level. | An ablation showing that SBG's specific structure (ratio, charts, D2) beats an NGF-style geometry-only factorised G on the same flights at equal state. |
| P7 | **Element Learning**, Du & Stechmann, arXiv 2308.02467 (v2 2024) [V: html v2]. | Learned **RTE** in2out and in2sol (Green's) operators, trained **only on a reference element** (a canonical local problem). Applied to any domain without retraining and assembled by a global GMRES solve. | SBG trains on canonical *charts* (slab, semi-slab, quarter-slab, dent), but its kernel is global with about 15–20 mm support and needs no outer solve. Flights are exact. Element Learning is 2-D, HG g = 0.8, conditioned on the medium, with no positivity or conservation guarantee. | "Train on canonical local problems, not on global sources" is SBG's chart-space-supervision claim. | **No** for chart- or element-level supervision of a learned transport Green's function. The geometry-varying chart (rather than a varying medium) and the non-iterative global kernel are PLAUSIBLY DISTINCT details. | Show that a chart-supervised kernel reaches the gates on BND_FAR without composition error where an element-composed alternative does not, or is cheaper. |
| P8 | **Learning elliptic PDEs with randomized linear algebra**, Boullé & Townsend, FoCM 2022, DOI 10.1007/s10208-022-09556-w [V]. **GF-Net**, Teng, Zhang, Wang & Ju, MSML / PMLR 190 (2022) [V]. **BI-GreenNet**, Lin et al., CIMS 2023, arXiv 2204.13247 [V; the "fundamental solution + regular part" form is not confirmed from the abstract]. | Learning a Green's function from a modest number of input/output pairs, then reusing it for any source. Uses symmetry and the analytic fundamental solution to handle the singular part. | SBG treats the singular near-diagonal part exactly (vis·W∞) and learns only the smoother remainder. | The "analytic singular part + learned remainder" split is SBG's D1 amendment. | **No** for the split (INFERENCE: standard singularity subtraction; BI-GreenNet's use of the fundamental solution is [V], its exact decomposition is [M]). | None. |
| P9 | **Meta-learning pseudo-differential operators**, Feliu-Fabà, Fan & Ying, arXiv 1906.06782 [V: abstract]. | A network maps a *parameter* to a compressed (wavelet) operator, trained from a few matvecs, and tested on RTE. | SBG's τ_dent maps dent parameters to a Green's-ratio field, and proposes H-matrix compression of G_rest − G_dent. | That is the dent amortisation route, the only place SBG claims learning adds information. | **No** for "learn a parameter → compressed transport operator map". The nested monotone ratio is PLAUSIBLY DISTINCT. | The H-matrix rank of G_rest^(≥2) − G_dent^(≥2) at gate accuracy, and τ_dent transfer to a held-out dent at the signed-change gates. |
| P10 | **Analytic bounded-medium kernels.** Jensen et al. 2001 dipole [M; PDF surfaced: https://www.cs.cornell.edu/courses/cs667/2005sp/readings/jensen01.pdf]; Donner & Jensen 2005 multipole [M]; d'Eon & Irving 2011 quantized diffusion [V-repo: https://naml.us/paper/subsurface/]; Frisvad et al. 2014 directional dipole [M]. | A bounded-medium Green's function built from the infinite-medium kernel plus image corrections for a half-space or slab. | Transport level instead of diffusion level. A learned multiplicative ratio instead of additive images. | Near the boundary (< 2 mfp), which is the BND strata, diffusion images are poor. | **Partly.** The structure "infinite kernel modified by a boundary" is ALREADY KNOWN. The transport-level learned ratio is PLAUSIBLY DISTINCT. SBG's own review showed that product composition is *worse* than additive images at a wedge. | Corner and edge accuracy of the learned ratio against a deterministic quarter-slab solve. |
| P11 | **Exact or heuristic transport-level half-space Green's functions.** Liemert & Kienle, Sci. Rep. 2013, exact semi-infinite RTE [V, abstract: https://pmc.ncbi.nlm.nih.gov/articles/PMC3684809]. Martelli et al., heuristic semi-infinite Green's function from the infinite-medium solution plus an extrapolated boundary [V, title: https://www.semanticscholar.org/paper/a88eba9b63d60bc30a3a6631e3a824277383c363]. Simon, Foschum & Kienle, "hybrid" Green's function: exact infinite-space solution plus an approximate boundary condition [V, title and summary only; full text not readable]. | An exact or near-exact transport-level half-space Green's function, or "exact G∞ + boundary correction". | SBG learns the correction and conditions it on charts and dents. Its exact slab base (D3) is a reduced deterministic solve. | These works supply an exact half-space teacher, so "transport-level rather than diffusion-level bounded kernel" is not new in itself. | **No** for the transport-level G∞-plus-boundary-correction structure. Learning it and extending it to edge and dent charts is PLAUSIBLY DISTINCT. | Show where the learned ratio beats a hybrid analytic G∞ ± image correction at transport level (a learning-free comparator). |
| P12 | **Precomputed canonical-shell transport.** Moon, Walter & Marschner, EGSR 2007 [V: https://www.cs.cornell.edu/~srm/publications/EGSR07-vol.html]. Müller, Papas, Gross, Jarosz & Novák, TOG 35(6) 2016 [V: https://cs.dartmouth.edu/~wjarosz/publications/muller16efficient.html]. | Multiple scattering precomputed in canonical local geometric contexts (spherical shells), reused across dynamic or heterogeneous scenes with no scene-specific precomputation. | SBG's charts are boundary contexts (slab, semi-slab, quarter-slab, dent) and the output is a deterministic linear operator, not a path sampler. | This is the graphics precedent for supervising transport on canonical contexts independently of the scene or source. | **No** for the principle. The boundary-chart linear-operator form is application-level. | None for the principle. |
| P13 | **Learned shape-adaptive transport.** Vicini, Koltun & Jakob, TOG 38(4) 2019 [V: https://rgl.epfl.ch/publications/Vicini2019Learned]. Leonard, Höhlein & Westermann, CGF 40(2) 2021 [V: https://diglib.eg.org/handle/10.1111/cgf142623]. | A learned bounded-medium multiple-scattering response conditioned on **invariant local geometry descriptors**: a polynomial SDF in a canonical frame, or the sphere distance to the boundary. Supervised by brute-force MC on canonical or synthetic geometry rather than on the target scene. Absorption or survival is predicted separately. | SBG learns a deterministic, linear, non-negative scalar kernel between exact flights rather than a stochastic exit sampler, with a monotone bound and a passivity projection. | This is the closest graphics precedent for chart-conditioned learning of bounded-medium transport. | **Partly.** Geometry-descriptor conditioning with MC chart supervision is ALREADY KNOWN. The passive linear-operator form is PLAUSIBLY DISTINCT. | Same as P6: a structure ablation. |
| P14 | **Per-asset learned transport.** RNA, Mullia et al., TOG 2024 [V: https://arxiv.org/abs/2312.09398]. 8DNA, Wu, Lu & Xu, arXiv 2604.25129 [V, abstract summary]. TransGI, Deng, Han & Fang, TVCG 2025 [V: https://arxiv.org/abs/2506.09909]. Neural SSS, Tg et al., CGF 43(4) 2024 [V: https://diglib.eg.org/handle/10.1111/cgf15158]. NeuPreSS, CGF 43(7) [V, title]. NRC, Müller et al., TOG 40(4) 2021 [V: https://arxiv.org/abs/2106.12372]. | Learned light transport of a medium or asset. 8DNA factors a learned distribution × a predicted survival probability (albedo), which is a "shape × separately determined total" analogue of the D2 projection. | These are source- or light-conditioned and baked per asset (NRC adapts online). SBG is source-free and chart-conditioned. | These are the external baselines the project positions against. SBG differs fundamentally in what is learned. | The difference **is real** but is the same difference P6/P7 already established. The "shape × total" normalisation is ALREADY KNOWN as a pattern. | Under AGENTS.md's baseline gate: an actual neural rebuild on invalidation (Neural Full Recompute) versus SBG under the same edit. That has not been specified. |
| P15 | **Precomputed translucent transfer.** Wang, Tran & Luebke, TOG 24(3) 2005 [V]. COMET response functions (Mosher & Rahnema; Forget thesis) [V, secondary: response functions computed once per *unique local problem*]. | A precomputed transfer operator with separate single scattering (Wang 2005). Response functions per unique local configuration, reused across a global problem (COMET). | SBG's "finite chart set at rest" claim. | The idea that the rest box decomposes into finitely many local problems, each solved once, is COMET's premise. | **No.** | None. |
| P16 | **Domain monotonicity of Green's functions.** Dirichlet Green's functions satisfy G_Ω1 ≤ G_Ω2 for Ω1 ⊂ Ω2, cited as standard [V: arXiv 2402.03571 cites it as standard; arXiv 2608.19356 proves it by the maximum principle]. For transport with a vacuum boundary the analogue follows from the path-subset argument (DERIVATION, re-derived; consistent with both Track B reviews). | The bound 0 ≤ G_Ω ≤ G_Ω′ for nested domains with an absorbing or escaping boundary. | Uses it as the **hard output parameterisation** of a learned transport kernel: a log-ratio τ ∈ [0,1] relative to an exact larger-domain base, nested for dents. | It is the only structural guarantee SBG adds beyond H2 or NGF. | **Principle: No. Use as a constraint on a learned RTE kernel: PLAUSIBLY DISTINCT** (no instance found; that absence is weak evidence). Its value with G∞ as the base: NOT SUPPORTED (∫G∞ = 1/(1−ω) = 1000 in red, conceded). With the slab base: UNCERTAIN. | Ablation: an unconstrained log-kernel versus the monotone ratio on unseen charts and dents. The constraint must reduce error or violations, not just exist. |
| P17 | **Equivariance by group averaging** (Reynolds operator, frame averaging; Puny et al. 2022 [V-repo, cited in WL77A coordinator decision]). | Exact invariance by averaging over the stabiliser and the x↔y swap. | Applied at tabulation time. | It fixes the tie-discontinuity objection. | **No.** | None. |
| P18 | **Internal precedent: WL77A H2 RBG** [FACT, repo: `docs/plans/wl77a/ideation.md:123-162`, `coordinator_decision.md:14`, `review_B_novelty_systems.md:197-212`]. | Exact Π ends, a learned scalar, source-free, symmetric Γ on element pairs within 15 mm, passive by construction, exact SS, K/C split by in-medium visibility. | (a) Chart solves and MC instead of S16-R source packets. (b) A monotone ratio to an exact base. (c) D2 instead of max-normalisation. (d) Graded depth of about 0.05 mm instead of 2.5 mm hats. (e) A non-iterative chart-sum application. | (a) addresses H2's "near-diagonal Γ at unseen plane configurations identified by smoothness only". (d) addresses the footprint-aliasing objection. | **Not a new mechanism.** (a)–(e) are corrections to H2's realisation. | Any SBG test must say that it is a revised H2. It must also use a learning-free deployable control (exact chart solves, or C1 on the same flights) as the primary comparator, as WL77A B4 already required. |

---

## 2. Classification of the central mechanism and each claimed contribution

**Central mechanism: a learned, non-iterative, source-independent scalar bounded-medium Green's function between exact flights, conditioned on geometry.**

**ALREADY KNOWN** as a mechanism.
- FACT and INFERENCE: each part is published.
  - The flight–resolvent–flight sandwich: P1, P3, P4.
  - Source-free learned Green's functions that generalise to unseen sources by construction: P5, P6, P8.
  - A geometry- or element-conditioned learned RTE Green's operator trained on canonical local problems: P7.
  - A learned, geometry-descriptor-conditioned bounded-medium multiple-scattering response: P13.
- FACT: the same combination was proposed in this project (WL77A H2) and rejected, in part for novelty (P18).
- An identical *published* implementation (exact angular flights + learned scalar collision resolvent for a port/bin contract) was not found. Under the review rules that absence is not evidence of novelty.

| Claimed contribution (amended SBG unless noted) | Classification | Rationale |
|---|---|---|
| C2-a: unseen-source generalisation "by construction" because τ never sees a source | **ALREADY KNOWN** | This is NGF's stated design principle (P6), shared by DeepRTE (P5) and WL77A H2. DERIVATION: it holds for any source-free linear Green's representation, so it does not distinguish SBG from C1, H2 or NGF. |
| C2-b: exact single-flight / multi-flight split (D1) | **ALREADY KNOWN** | First-collision-source and singular-part subtraction (P2, P8). |
| C2-c: monotone bound 0 ≤ G_Ω ≤ G_base, τ ∈ [0,1] | **Principle ALREADY KNOWN (P16). As a hard constraint on a learned transport kernel: PLAUSIBLY DISTINCT** | DERIVATION: valid for an escaping, index-matched boundary, and it fails under Fresnel/TIR (conceded). With G∞ as the base it is near-vacuous in red: **NOT SUPPORTED** as a useful constraint. With the exact slab base its usefulness is UNCERTAIN. Spatial-subset caveat (INFERENCE): the brief's dent pushes the top down with the bottom fixed, so Ω_dent ⊂ Ω_rest in spatial coordinates and the nesting holds for collision sites. It is not a statement about materially indexed ports, which move. |
| C2-d: exact in-plane-invariant slab base (3-variable G_slab, Fourier → 1-D depth Fredholm) | **ALREADY KNOWN** | INFERENCE [M]: transverse-Fourier or Hankel reduction of slab transport is standard. Transport-level half-space and slab Green's functions exist (P11). |
| C2-e: chart-space supervision (canonical boundary charts, supervised exhaustively, independent of source sampling) | **ALREADY KNOWN** as a principle | Reference-element training (P7), unique-local-problem response functions (P15), canonical-shell precomputation (P12), MC on canonical or synthetic shapes (P13). The specific charts (slab, semi-slab, quarter-slab, dent) are application-level. DERIVATION (developer's D3, conceded): on the box every chart is exactly computable by reduced solves, so learning there adds **compression, not information**. |
| C2-f (original): corners from edge charts by product composition, with no corner data | **NOT SUPPORTED** | DERIVATION (Track B Review 2, checked by the developer): already in the steady-state Dirichlet wedge, r1²r4² − r2²r3² < 0, so the product is biased low. No transport boundary-layer composition rule is known. The claim was withdrawn. |
| C2-g: D2 geometry-only energy projection | **Identity ALREADY KNOWN; the projection device is PLAUSIBLY DISTINCT; "exact output totals" is NOT SUPPORTED** | DERIVATION: uᵀG = (1−P_esc)ᵀ is the resolvent form of the classical collision-probability conservation sum rule (P1). It gives one constraint per column. That bounds passivity, but the escape total is exact only if Ĝ's absorption/escape split is exact. This is tight in red and loose in blue (§0 item 2). Normalising collision-probability matrices to conservation is, from memory [M], standard practice in lattice codes; not verified this session. "Learned shape × separately fixed total" is a known pattern (8DNA survival term, P14; Vicini, Leonard absorption prediction, P13). Track B reviews also show D2 column scaling breaks the symmetric-τ reciprocity unless the scaling is symmetric (D·Ĝ·D), whose existence is unverified. |
| C2-h: "errors are not iteration-amplified" (resolvent-level approximation) | **ALREADY KNOWN** as a trade-off | DERIVATION: true. But the conditioning reappears as the global geometric sensitivity of G that τ must learn (conceded, R2-5). "Learn the Green's function" versus "learn the generator and solve" are the two classical organisations; WL77A ideation:342 makes the same distinction. |
| C2-i: chart-wise partition-of-unity sum ½(G_k D_k + D_k G_k) | **ALREADY KNOWN** | Partition-of-unity blending and domain-decomposition assembly; the symmetric form is a minor algebraic device. Its accuracy at chart overlaps is UNCERTAIN, with no derivation offered. |
| C2-j: nested dent ratio τ_dent with pair-frame descriptors, plus H-matrix compression of G_rest^(≥2) − G_dent^(≥2) | **Learned parameter → operator: ALREADY KNOWN (P9, P6). Monotone nesting: PLAUSIBLY DISTINCT. H-matrix rank at gate accuracy: UNCERTAIN** | This is the only route on which learning could add information (amortisation over a dent family). Its compressibility, supervision count (thousands of MC fields) and rebuild cost are unproven and conceded as the weakest link. |
| C2-k: exact D4h equivariance by Reynolds averaging | **ALREADY KNOWN** | P17. |
| C2-l: the combination as a whole | **UNCERTAIN (narrow)** | Transport-level, monotone-ratio-constrained, chart-supervised scalar Green's function between exact port/bin flights, with a geometry-only passivity projection. No identical work was found. Every element is known, and the combination is a corrected WL77A H2. Novelty at most at the application or engineering level, and only meaningful if a cost advantage is shown. |

---

## 3. What empirical evidence would be required to support any residual novelty claim

Each item is a HYPOTHESIS until measured. All must be predeclared.

1. **A learning-free deployable control as the primary comparator.**
   - The control is the exact chart kernels from reduced solves, or C1 on the *same* flights and lattice.
   - SBG may claim value only if, at equal accuracy on every stratum (INT, BND_NEAR, each BND_FAR sub-stratum, against MC references), its apply plus per-dent rebuild beats both:
     - that control inside the GI loop;
     - the few-second path-traced re-render.
   - This is SBG's own revised step 4. WL77A B4 required the same for H2. Without it, a pass belongs to the classical solver, not to learning.

2. **A structure ablation against a published-style learned Green's function.**
   - Use the same exact flights and the same chart supervision, with an NGF-style unconstrained, geometry-only factorised G (P6) at matched state.
   - SBG's specific devices (monotone ratio, nesting, D2) must show a measurable gain in error or violation rate on unseen charts and dents. Merely guaranteeing a property the baseline violates rarely is not enough.

3. **Corrected D2 claim test.**
   - Report the per-column error of the escape total after D2 projection versus after the exact adjoint (w*) recalibration, per channel.
   - Expected from the DERIVATION: small in red, large in blue. If blue errors are material, "geometry-only exact energy" must be dropped.

4. **Dent amortisation test, the only information-adding route.**
   - Measure the H-matrix rank of G_rest^(≥2) − G_dent^(≥2) at gate accuracy.
   - τ_dent transfer to a held-out dent (relative signed-change error ≤ 0.35, correlation ≥ 0.90).
   - τ_dent rebuild time versus a local deterministic re-solve.
   - If the deterministic re-solve is as fast, the learned component has no claim (SBG's own rule).

5. **Baseline-terminology compliance** (AGENTS.md gate, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`).
   - Any external comparison must state the actual invalidation, and RNA/8DNA's native rebuild lifecycle (Neural Full Recompute versus Native Reuse).
   - It must keep the separate path-traced reference.
   - C2's records compare cost only to path tracing and C1. No such comparison is specified.

6. **Provenance disclosure.** Any successor write-up must state that SBG is a revision of WL77A H2 RBG and address H2's upheld objections individually.

---

## 4. Did the Track B rejections of C3 / C4 look wrong from this dimension?

No novelty evidence found here overturns them. INFERENCE: C3's context-atlas premise matches COMET-style unique-local-problem response functions (P15). C4's modal and collision-order tail is covered by the prior art Track B already cites (PRT compression; Hašan & Ramamoorthi 2013). Both rejections were made on validity and cost, which this search does not bear on.

---

## 5. Verdict

- **Central mechanism: ALREADY KNOWN** (FACT for the parts, P1–P8 and P12–P13; FACT that the same combination was proposed and rejected internally as WL77A H2).
- **Residual distinctness: UNCERTAIN, narrow, engineering or application level.** It consists of the monotone nested-ratio output constraint (PLAUSIBLY DISTINCT), the D2 projection device (PLAUSIBLY DISTINCT, but its "exact energy" claim is NOT SUPPORTED), and the specific chart set.
- **Generalisation contributions:** the source-side claim is ALREADY KNOWN. Chart-side transfer without chart data is NOT SUPPORTED (product composition withdrawn).
- **Consequence (INFERENCE):** SBG's case rests on a cost claim (learned compression or amortisation versus deterministic chart solves). That claim is ESTIMATED only, at tens of ms against tens of ms, and is unmeasured. This agrees with Track B's deferral of C2 as a conditional variant of C1. It adds two points Track B lacked: the internal WL77A precedent, and the D2 over-statement.

**Sources (web, checked this session):**
- [DeepRTE, arXiv 2505.23190](https://arxiv.org/html/2505.23190v4)
- [Neural Green's Functions, arXiv 2511.01924](https://arxiv.org/html/2511.01924v1)
- [Element Learning, arXiv 2308.02467](https://arxiv.org/html/2308.02467v2)
- [Boullé & Townsend, arXiv 2102.00491](https://arxiv.org/abs/2102.00491v2)
- [GF-Net, PMLR 190](https://proceedings.mlr.press/v190/teng22a.html)
- [BI-GreenNet, arXiv 2204.13247](https://arxiv.org/abs/2204.13247v1)
- [Meta-learning pseudo-differential operators, arXiv 1906.06782](https://arxiv.org/abs/1906.06782)
- [Structure-preserving RTE surrogates, arXiv 2511.04991](https://arxiv.org/abs/2511.04991)
- [Collision probability method (tutorial)](https://deoliveira-r-orpheus.readthedocs.io/en/latest/theory/collision_probability.html)
- [Collision probability methods – an update (OSTI)](https://www.osti.gov/biblio/6529305)
- [Rushmeier & Torrance 1987](https://history.siggraph.org/?p=116086)
- [Photon Beam Diffusion](https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html)
- [Christensen & Burley 2015](https://graphics.pixar.com/library/ApproxBSSRDF)
- [Moon, Walter & Marschner 2007](https://www.cs.cornell.edu/~srm/publications/EGSR07-vol.html)
- [Müller et al. 2016](https://cs.dartmouth.edu/~wjarosz/publications/muller16efficient.html)
- [Vicini, Koltun & Jakob 2019](https://rgl.epfl.ch/publications/Vicini2019Learned)
- [Leonard, Höhlein & Westermann 2021](https://diglib.eg.org/handle/10.1111/cgf142623)
- [Deep Scattering, arXiv 1709.05418](https://arxiv.org/pdf/1709.05418)
- [Liemert & Kienle 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3684809)
- [Martelli et al., heuristic semi-infinite Green's function](https://www.semanticscholar.org/paper/Heuristic-Green's-function-of-the-time-dependent-a-Martelli-Sassaroli/a88eba9b63d60bc30a3a6631e3a824277383c363)
- [Simon, Foschum & Kienle, hybrid Green's function (title only)](https://biomedicaloptics.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-18/issue-01/015001/Hybrid-Greens-function-of-the-time-dependent-radiative-transfer-equation/10.1117/1.JBO.18.1.015001.full)
- [Domain monotonicity, arXiv 2608.19356](https://arxiv.org/pdf/2608.19356)
- [Domain monotonicity cited as standard, arXiv 2402.03571](https://arxiv.org/pdf/2402.03571)
- [Wang, Tran & Luebke 2005 (bibtex)](https://researchr.org/publication/WangTL05/bibtex)
- [COMET (ANS)](https://www.ans.org/pubs/journals/nse/article-53536/)
- [RNA](https://arxiv.org/pdf/2312.09398)
- [8DNA](https://arxiv.org/pdf/2604.25129)
- [TransGI](https://arxiv.org/pdf/2506.09909)
- [Neural SSS](https://diglib.eg.org/handle/10.1111/cgf15158)
- [NeuPreSS](https://diglib.eg.org/handle/10.1111/cgf15234)
- [NRC](https://arxiv.org/abs/2106.12372)
- [GINO](https://arxiv.org/pdf/2309.00583)

**Repo files cited (read-only):**
- C:\Projects\ReST-GI\docs\plans\wl80a\trackB_candidate_records.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackB_adversarial_review.md
- C:\Projects\ReST-GI\docs\plans\wl80a\trackB_selected_candidate.md
- C:\Projects\ReST-GI\docs\plans\wl80a\neutral_problem_brief.md
- C:\Projects\ReST-GI\docs\plans\wl77a\ideation.md
- C:\Projects\ReST-GI\docs\plans\wl77a\coordinator_decision.md
- C:\Projects\ReST-GI\docs\plans\wl77a\review_B_novelty_systems.md
- C:\Projects\ReST-GI\docs\plans\wl77a\specialist_priorart.md
