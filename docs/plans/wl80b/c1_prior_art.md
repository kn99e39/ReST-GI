# C1 novelty and admissibility (Part F)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part F: novelty and admissibility of C1 (CP-K)

This is design-only. I read only the allowed inputs, ran nothing and wrote no file.

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement or record (file:line) |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

**How each source was checked.**

| Mark | Meaning |
|---|---|
| V | Abstract or text read this session |
| V-s | Snippet or bibliographic record only |
| V-repo | Taken from the WL80A or WL77A record; not re-opened by me |

## F0. Verdict

- **Novelty.** C1 contains no new method [ND, from F2–F3].
  - Its equation, its computational mechanism and every listed contribution (C-a to C-g, plus shared-W RGB) are ALREADY KNOWN, or are not contributions.
  - The combination "exact integral-transport intrinsic K at port/bin resolution, rebuilt under deformation, coupled with an explicit free-space C" is UNCERTAIN, and only at the engineering or application level. Not finding an identical system is not evidence of novelty.
  - Any value C1 has would be an empirical operating point. Nothing about C1 has been measured.
- **WL77A H3.** C1 escapes none of the four H3 rejection grounds (`docs/plans/wl77a/coordinator_decision.md` l.15) [HM].
  - Removing H3's learned closure does not answer the "vacuous learning" ground. It makes that ground true by definition.
  - The cost and scope grounds carry over unchanged or stronger.
  - In two respects C1 is less exact than H3: it gives up exact reciprocity, and its order-2 escape is cell-averaged rather than exact.
- **Admissibility.**
  - Under the FNTO v0.1 contract text, C1 is **not admissible as the FNTO learned K**.
  - Under the current `docs/Architecture.md`, the K definition does not say "learned", so the text is ambiguous rather than prohibitive.
  - One role is already documented for a non-learned K: the WL56 Monte Carlo oracle K, used as a baseline or oracle.
  - Every other role needs an explicit human decision (F5).

## F1. WL77A H3 and its rejection grounds, applied to C1

### What H3 was [HM: `candidate_H3.md`]

- **Operator** (l.69–113): 𝒦 = 4πσ_t[ω·D1 + ω²·D2 + ω³·b̄ᵀ L_θ⁻¹ b̄].
  - D1 (single scatter) and D2 (double scatter) are computed deterministically and exactly.
  - L_θ = V − ω·τ̂_θ is a collision-probability (CPM) Galerkin generator on a grid with 0.5 mm boundary bands and 1.25 mm interior cells (l.78–80).
  - Its symmetric form gives exact reciprocity (l.123).
  - Passivity comes from scaling by s_X·s_Y (l.109).
- **Learned part:** a stencil closure of 6–10k parameters, bounded by c_max = 0.5 (l.93–99, l.138).
- **θ = 0** is stated to be "a learning-free integral-transport solver" (l.113).

### The four grounds (`coordinator_decision.md` l.15), checked against amended C1

| # | H3 ground | Does it apply to C1? | Basis |
|---|---|---|---|
| G1 | Transfer comes from a published classical CPM / first-collision-source solve; the learned closure is likely vacuous (A8 Horn A, B4) | **Yes, more strongly.** C1 has no learned part, so Horn A holds by definition, not by prediction. Horn B (a closure cannot be both source-independent and useful) disappears only because nothing is learned. | [HM] `trackB_selected_candidate.md` l.135 ("nothing to learn"; conceded "a substitution, not an answer"); `responses.md` l.205–221 |
| G2 | Admissibility as a learned FNTO K is an untaken human decision | **Yes, unchanged, and starker.** H3 at least had a learned θ; C1's θ is empty. | [HM] FD l.191, l.403; CD l.17 |
| G3 | The apply meets the cap only with an unbuilt kernel (B5) | **Yes, plausibly worse.** See the derivation below. | [ND]; [HM] CD l.21 |
| G4 | Needs a known scalar kernel: g = 0, homogeneous, index-matched; does not extend to Burley, Fresnel or g ≠ 0 (A9) | **Yes, unchanged.** All incoming-direction dependence enters through F_in only under these assumptions. | [EP]; [HM] SC l.218 |

**G3 derivation [ND].** This is a roofline lower bound. It is unmeasured, and the bandwidth (912 GB/s vendor peak) is the one review B5 assumed.
- The amended C1 near-field W has about 1e8 nonzeros (AR l.338–353, per Part A).
- In CSR with fp32 values and int32 indices that is about 0.8 GB read per W application, so at least about 0.9 ms per application. The read is shared across RGB only if channels are batched.
- Applications per solve:
  - plain iteration needs about 524 sweeps (CD l.21), so at least about 0.46 s per light;
  - the accelerated 10–40 applications give at least about 9–35 ms, before the far field, F_in and Φout.
- H3's near generator was 1.6–2.3e7 nonzeros (`candidate_H3.md` l.240), and B5 already put H3 at or above the 5 ms single-light cap.
- The measured single-light anchor is FIA's 0.0039 s [HM, `76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.255].

### What changed from H3 to C1 [ND, comparing the two records]

| Element | H3 (θ = 0) | Amended C1 | Net effect |
|---|---|---|---|
| Order-2 escape | Exact D2 on a fine 0.25 mm layer (l.90) | Cell-averaged through Φout acting on ψ (SC l.95–97) | C1 is less exact at order 2 |
| Second-collision density | Exact arrivals integrated per element (b̄, l.91) | Point-to-cell W1 (SC l.95) | Equivalent class |
| Depth grading | 0.5 mm bands | Graded down to 0.05 mm (AR l.338–353) | C1 is finer in depth. Laterally both use 1.25 mm, and C1 has no lateral boundary band |
| Passivity | s_X·s_Y scaling certificate | Escape defined as the complement, plus sub-stochastic W (CD l.20) | Different devices; both standard |
| Reciprocity | Exact by construction | Given up, and its defect reported (AR l.304) | C1 regresses |
| Learned part | Closure, bounded by c_max | None | Removes the learning question rather than answering it |

- **Identity statement.** CD l.18 ("the H3 θ = 0 operator class with a revised discretisation") is accurate. FD l.191 ("C1 is H3 at θ = 0") overstates the sameness, and CD l.18 supersedes it.
- **Precision note on Part A, ambiguity 15.** Both operators hold the order-2 collision density at element resolution. "H3 grids k ≥ 3, C1 grids k ≥ 2" holds for the order of *escape*, not for collision density.

### Additional ground from WL77A that also carries over

- **B4, information fairness.** C1 receives σ_t, ω_c, g = 0, isotropy and index matching. The learned families did not (A75 received only B and β; FIA received σ_t and ω).
- So "C1 beats the learned family" would not show that the learned family lacked an attainable mechanism.
- CD l.16 already restricts what a C1 pass would mean [HM].

## F2. Targeted prior-art search

**Column key.**

| Column | Question |
|---|---|
| Eq | Is the underlying equation identical? |
| Mech | Is the computational mechanism identical? |
| Repr | Does the representation differ? |
| Upd | Does the update mechanism differ? |
| Work | Does the workload differ? |
| Contrib | Is there a potentially defensible new contribution for C1? |

| # | Predecessor (check level) | Eq | Mech | Repr / Upd / Work | Contrib | Class |
|---|---|---|---|---|---|---|
| P1 | Peierls / collision-probability (CP) method. Ray-traced 3-D CP: Roy–Hébert–Marleau 1989; Marleau–Hébert NSE 1986; the 3-D CP / escape / transmission NSE 2003 paper; DRAGON EXCELT 3-D CP module (V-s, this session); CP conservation and reciprocity relations (US 5267276, V-s) | Yes | Yes for W (flat-source CP by ray tracing) and for escape as the conservation complement | Repr: point-level W1 and Φ_pt at the first collision. Upd: none new. Work: reactor lattice cells versus an optical slab | No | ALREADY KNOWN |
| P2 | First-collision source: Alcouffe 1985 (V-repo); DOCTORS (V-repo); Fan–An–Ying JCP 2019 §4, which treats the boundary condition as a boundary source, with an auxiliary uncollided transport solve with Φ = 0 on the inflow boundary Γ₋ (V, this session; this closes WL80A open check C.3 for Fan) | Yes | Yes (F_in, Σ F_in = d) | Rays shared with B is an implementation detail | No | ALREADY KNOWN |
| P3 | Fast integral RTE solvers: Fan–An–Ying (FFT with zero padding); Ren–Zhang–Zhong (FMM; scalar density first, then a scattering-free solve for the angular field) (V) | Yes | Yes: scalar resolvent followed by an angular sweep, which is ψ then Φout | Repr: graded body-fitted mesh versus a regular grid or FMM | No (a discretisation choice) | ALREADY KNOWN |
| P4 | Response matrix / interface current: Weiss–Lindahl high-order response-matrix equations (NSE 58; Legendre interface currents, V-s); Roberts–Forget eigenvalue response matrix (V-s); TEP with DP1/DP2 angular interface expansion (V-s); domain decomposition exchanging interface *angular fluxes* (parallel block Jacobi; TITAN coarse-mesh ray tracing with angular and spatial interface projections; characteristic direction probabilities coupling through interface angular flux, V-s) | Yes (a boundary-in → boundary-out response) | Yes for the K-as-response-matrix role | Repr: 2.5 mm × 64 discrete bins versus Legendre or DPn orders, or per-track angular fluxes. Work: coupling through free space (C) instead of shared interfaces | No. Direction-resolved interface fluxes exist, so 16/64 is a resolution parameter. This closes WL80A open check C.1 at class level | ALREADY KNOWN (class); exact resolution UNCERTAIN, and not a mechanism |
| P5 | Zonal and exchange-factor family: Hottel / exchange-factor method (OSTI 6460797, V-s: "for gray media with isotropic scattering, exchange factors are functions of system geometry and total extinction only"); generalized zonal method with nondiffuse and anisotropic exchange (V-s); vector-form exchange-area method, ASME J. Heat Transfer 131(1) 2009, with weighted *directional* distributions (V-s; full text blocked, 403, authors not confirmed); discrete transfer method, Lockwood & Shah 1981, with per-element solid-angle beams (V-s); Rushmeier–Torrance 1987 and Sillion 1995 (V-repo) | Yes | Yes for joint volume–surface exchange (C-f) and for **channel-shared geometry (M9)** | Repr: directional boundary bins. Directional variants exist (GZM, vector form, discrete transfer) | No. This closes WL80A open check C.2 at class level | ALREADY KNOWN |
| P6 | Directional radiosity in graphics: Immel, Cohen & Greenberg, SIGGRAPH 1986, with a discretised hemisphere of directions per patch (V-s); Max, EGWR 1994, with direction bins for anisotropic volume scattering (V-s); Languénou, Bouatouch & Chelle, EGWR 1994 (title only, V-s) | Same family (surface or volume finite elements) | Directional patch exitance is the same idea as port × bin | Work: surfaces or volumes in scenes versus an intrinsic slab K | No | ALREADY KNOWN |
| P7 | Plane-parallel boundary operators: matrix-operator method, Plass, Kattawar & Catchings, Appl. Opt. 12(2) 1973 (V-s); adding–doubling and invariant embedding (V-repo) | Yes (1-D) | Discrete-angle R/T boundary operators | Work: a finite 3-D body with edges and corners, which P1–P4 cover | No | ALREADY KNOWN (concept) |
| P8 | Green's-function source-agnosticism (C-g): Neural Green's Functions (arXiv 2511.01924); DeepRTE (arXiv 2505.23190, V: pre-trained with delta-function inflows, "zero-shot" claim; per V-repo it needs retraining for new geometry); Boullé–Townsend | Yes (linear, with the source only on the right-hand side) | C1 computes; these learn | Upd: C1 rebuilds deterministically | No | ALREADY KNOWN (as an argument) |
| P9 | Precomputed and compressed transport: CPCA and LDPRT PRT (V-repo); Wang–Tran–Luebke TOG 24(3) 2005, translucent PRT with single and multiple scatter (V-s); Hašan–Pellacini–Bala TOG 25(3) 2006, direct-to-indirect transfer as wavelet-sparse matrices (V-s); Lensch 2002/03; Translucent Radiosity; Modular Flux Transfer; shell transport functions (V-repo) | Different approximations of the same linear operator | Precomputed linear transfer, reused across lighting | Repr: C1 is explicit and uncompressed | No (reuse for lighting only at fixed geometry is old) | ALREADY KNOWN |
| P10 | Dynamic and incremental updates: Drettakis & Sillion, SIGGRAPH 1997, line-space hierarchy that finds the links affected by an object move and re-solves only those (V-s); Wang et al. EG 2010 (topology-preserving deformation) and Arbree 2011 (V-repo); generalized perturbation theory for CP-matrix variations in lattice codes (ANS, V-s) | n/a | Recompute the affected rows, then re-solve globally | Upd: C1's dent rebuild has no specified dependency certificate (B6) | No | ALREADY KNOWN (in substance) |
| P11 | Neural transport: RNA, 8DNA, TransGI, NeuPreSS (= Neural BSSRDF, one DOI), NRC, Vicini 2019, Element Learning (V-repo) | Different | Learned per asset or per shape | C1 is non-learned and exact to its discretisation | Different only because C1 is classical; not a mechanism | NOT a novelty basis (the difference is the absence of learning) |
| P12 | Project-internal precedents: the WL56 MC oracle K uses "uniform 2.5 mm … 64 outgoing / 16 incoming … 2048 photons per input cell", with K = entry → first exit [HM, `56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md` l.30]; WL77A H3; WL76 FIA (F_in, single scatter) | Yes | The same contract, solved by Monte Carlo instead of deterministic CP | C1 is a deterministic, low-noise replacement for the MC oracle. Cost anchor: oracle S 190 s per update in WL67 (as the roadmap records it) | Engineering only | ALREADY KNOWN internally |

### Per-claim classification

| Claim | Class | Basis |
|---|---|---|
| Central mechanism (first-flight source, then scalar CP resolvent, then last-flight escape) | ALREADY KNOWN | P1–P3 |
| C-a: port/bin first-exit contract | NOT a contribution | Given by the brief; P4–P6 for directional interfaces |
| C-b: exact ballistic complement, Σ F_in = d | ALREADY KNOWN | P2 |
| C-c: passivity through complement escape | ALREADY KNOWN | P1 conservation normalisation. It holds only if W ≥ 0 and 1ᵀW ≤ 1 (CD l.20) |
| C-d: exact D4h symmetry | ALREADY KNOWN | Inherited from a symmetric mesh and quadrature |
| C-e: body-fitted local rebuild | ALREADY KNOWN | P10 |
| C-f: joint K + C collision densities | ALREADY KNOWN | P5, P6 |
| C-g: "unseen-source generalisation becomes a convergence question" | NOT SUPPORTED as new | P8 and P12. It holds for any deterministic linear solver, and for the MC oracle |
| M9: RGB shares W | ALREADY KNOWN | P5 (exchange factors depend only on geometry and extinction); Hašan–Ramamoorthi 2013 (V-repo) |
| Point-exact W1 / Φ_pt combined with cell-level complement escape | ALREADY KNOWN in substance | First-collision source plus conservation normalisation. I found no element at PLAUSIBLY DISTINCT |
| Whole system as a dynamic-GI intrinsic K | UNCERTAIN (application or engineering) | No identical system found; absence is not evidence |

## F3. Better implementation versus new method

**Implementation or engineering (I).** None of these changes the transport method.
- graded body-fitted mesh;
- point-level first continuation;
- complement escape;
- positivity-preserving far field;
- rays shared with B;
- orbit-symmetrised quadrature;
- precorrected FFT or a matrix-free W;
- warm-started accelerated solves;
- a deterministic replacement for the WL56 MC oracle.

**Methodological contribution (M).** None is supported.

**What could be claimed if it were ever measured [UH].** All of these are unmeasured.
- C1 reaches the contract gates on BND_FAR, corner and dent strata at an affordable resolution, with lower noise and cost than the MC oracle. That would be an operating-point result.
- Or C1 plateaus (order < 0.5), which CD l.15 classes as a convention, reference or lateral-bias finding.

**What neither outcome would show.**
- It would not answer whether a *learned* K can transfer.
- It would not attribute the learned-family failures (CD l.16).
- It would not establish value against a neural baseline.

## F4. Admissibility against the contract text

### FNTO v0.1 contract

`docs/Architecture.md` l.9 marks it as the historical design contract dated 2026-10-23. Its text:

| Line | Text | Bearing on C1 |
|---|---|---|
| l.6 | Aim: "reusable learned source-to-receiver transport response" | C1 has no learned part |
| l.26 | "K_{θa}: reusable learned intrinsic optical-domain scattering operator" | Same |
| l.66–72 | K^sharp / K^internal are learned families | Same |
| l.103 | Persistent: "trainable K_a parameters" | Same |
| l.180 | Completion condition: "learned intrinsic-domain … kernel" | Same |
| l.17 | "do not replace learned complex shading with a limited explicit radiosity solver" | Applies by analogy: C1 is explicit and limited to g = 0, homogeneous, index-matched media |
| l.86 | "not an exhaustive explicit per-surface full path tracer hidden under a neural name" | This text is about the learned **C**. WL77A A8 and H3 l.290 applied it to K by analogy only [ND]. It is not a direct bar on K |
| l.161 | Nonrigid edits need "persistent intrinsic scattering rules and configuration-dependent within-object relations" | Read below |

- **Reading of l.161 [ND].** C1 can be cast in this form: the rule is a set of declared physical constants (σ_t, ω_c, g = 0, index matching), and the relations are F_in, W1, W, Φ_pt and Φout.
- **Verdict: not admissible as the FNTO K under v0.1.**

### Current living contract, `docs/Architecture.md`

| Line | What it says | Bearing on C1 |
|---|---|---|
| l.17 | Defines K physically, without "learned" | Does not exclude C1 |
| l.36 | K must be linear, non-negative and passive | C1 meets this by design at convergence, conditional on W ≥ 0 and 1ᵀW ≤ 1 [WD] |
| l.42 | "Reusable rule parameters: source-independent intrinsic material/medium response" | Silent on whether these must be learned |
| l.50 | Runtime inputs exclude target K, reference images and identities | C1 complies [WD] |
| l.94 | Excludes a "free-space radiosity solver" | Names a C-side solver; an intrinsic CPM solver is not named |
| l.11 and l.98 | The open question is "learned K adequacy"; "no learned representation is adopted" | — |
| l.87 | Already accepts a non-learned K in one role: the "exact optical-domain selective oracle — baseline for avoided work" | See next point |

- `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.12 limits that oracle role: MC rebuild savings "are not a deployable learned contribution".
- **Verdict: ambiguous.** The text does not forbid an explicit K, but the research question it owns is about a learned K.

### WL80A brief and baseline gate

- **Brief.** It asks for "a learned K" (`neutral_problem_brief.md` l.23). It frames value as "cheaper than re-rendering" (l.152).
- **Baseline gate.** Under `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.21–30, a path-traced re-render is comparison class 3 only.
- Any value claim must also name:
  - Neural Full Recompute, for an 8DNA/RNA asset that is actually invalidated (for example by a dent changing the medium's internal geometry);
  - or Native Reuse, where the asset stays valid (for example a rigid move);
  - and the internal F/S comparator.
- None of these is measured for C1 [HM: FD l.199 amendment 6].

## F5. Alternatives that need an explicit human decision

1. **Whether an FNTO K must contain learned parameters.** Either keep v0.1 l.6/26/103/180, or amend the contract so that, where the kernel is known, a K may be an explicit known-physics operator. That amendment would put C1 in scope for g = 0, homogeneous, index-matched media.
2. **If (1) is amended:** whether declared physical constants (σ_t, ω_c, g, index matching) count as Architecture §4 "reusable rule parameters". Also whether this privileged physics is acceptable when comparing with learned arms (WL77A A9 and B4).
3. **If (1) is not amended, which role C1 has:**
   - (a) a diagnostic instrument, or a deterministic analogue of the WL56 MC oracle K, in the existing baseline/oracle role (CD l.17);
   - (b) no role.
4. **Whether C1 outputs may be used as labels or a teacher for a learned K.** [ND] They would be model outputs, not "genuinely independent physical labels" in the sense the roadmap requires (`RESEARCH_ROADMAP.md` l.42). So this needs its own decision.
5. **Whether to reopen a physics-sandwiched learned K.** These are separate decisions. Both are covered by WL77A open question 3 (`coordinator_decision.md` l.100) and FD l.205.
   - WL77A H3, a learned closure on the CPM generator: rejected in WL77A.
   - WL80A C2 SBG, a learned scalar resolvent between exact flights (a revised WL77A H2): conditional.
6. **Which comparator defines value for any C1 run:**
   - Neural Full Recompute, or Native Reuse, under the gate;
   - internal F/S;
   - the class-3 path-traced re-render.

   The brief's "cheaper than re-rendering" covers class 3 only.
7. **Clamp and positivity policy.** This is protocol-level, not admissibility. Any far-field clamp must be predeclared and its binding fraction reported (FD l.195; AGENTS.md forbids post-hoc clamps).

Nothing here is authorised, and none of these items is a recommendation.

## Sources

**Web (this session).**
- [Fan, An & Ying, arXiv 1802.03061 (HTML)](https://arxiv.org/html/1802.03061); [OSTI record](https://www.osti.gov/biblio/1547885)
- [Ren, Zhang & Zhong, arXiv 1610.00835](https://arxiv.org/abs/1610.00835)
- [DeepRTE, arXiv 2505.23190](https://arxiv.org/abs/2505.23190)
- Exchange-factor and zonal methods:
  - [Exchange factor method (OSTI 6460797)](https://www.osti.gov/scitech/biblio/6460797)
  - [Generalized zonal method (OSTI 7310648)](https://www.osti.gov/biblio/7310648)
  - [n-bounce method (OSTI 5568544)](https://www.osti.gov/biblio/5568544)
  - [Vector-form exchange-area method, ASME J. Heat Transfer 131(1)](https://asmedc.silverchair.com/heattransfer/article/131/1/012701/467241/A-Vector-Form-Exchange-Area-Based-Method-for)
- Discrete transfer method:
  - [IFRF handbook](https://ifrf.net/research/handbook/what-is-the-discrete-transfer-method/)
  - [Heriot-Watt research portal](https://researchportal.hw.ac.uk/en/publications/improvements-to-the-discrete-transfer-method-of-calculating-radia/)
- [Immel, Cohen & Greenberg 1986 (SIGGRAPH history)](https://history.siggraph.org/?p=116032)
- Max 1994: [Scilit](https://scilit.net/publications/99721abbc5ae702fef4f7f0eed05d939); [Springer](https://link.springer.com/doi/10.1007/978-3-7091-7484-5_10)
- [Pérez, Pueyo & Sillion survey (lists Languénou et al. 1994)](https://rd.springer.com/chapter/10.1007/978-3-7091-6858-5_28)
- [Plass, Kattawar & Catchings 1973 (Scilit)](https://scilit.com/publications/27041e81eb7621b584b1b5523fdae29b)
- Response-matrix and interface-current methods:
  - [Roberts thesis (MIT DSpace)](https://dspace.mit.edu/handle/1721.1/87490)
  - [Response-matrix background (core.ac.uk)](https://core.ac.uk/download/33353029.pdf)
  - [ANS NSE a_28220](https://wx1.ans.org/pubs/journals/nse/a_28220)
  - [TEP DP1/DP2 extension (GTNEUT)](https://frc.gatech.edu/files/2016/02/2004-extensionsofthetepneutraltransport.pdf)
  - [Marleau & Hébert, ANS NSE](https://www.ans.org/pubs/journals/nse/article-18171/)
  - [DRAGON5 module listing (Polytechnique Montréal course page)](https://moodle.polymtl.ca/mod/resource/view.php?id=215826)
- Domain decomposition with interface angular fluxes:
  - [Globally conservative hybrid method, arXiv 1605.05388](https://arxiv.org/pdf/1605.05388)
  - [TITAN manual](https://me.vt.edu/content/dam/me_vt_edu/labs/transport-theory/titan.manual-v1.05.pdf)
  - [ANS NSE article 2049](https://www.ans.org/pubs/journals/nse/article-2049/)
- CP perturbation theory: [ANS NSE article 23431](https://www.ans.org/pubs/journals/nse/article-23431/); [US patent 5267276](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5267276)
- [Wang, Tran & Luebke 2005 (SIGGRAPH history)](https://history.siggraph.org/?p=113966); [researchr](https://researchr.org/publication/WangTL05/related)
- [Hašan, Pellacini & Bala 2006 (Cornell)](https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting)
- [Drettakis & Sillion 1997 (SIGGRAPH history)](https://history.siggraph.org/?p=117251); [dblp](https://dblp.org/rec/conf/siggraph/DrettakisS97.html)

All other citations are V-repo, from `docs/plans/wl80a/prior_art_matrix.md` l.316–409.

**Repository files read (read-only), all under `C:\Projects\ReST-GI\`.**
- `docs\plans\wl80a\`: `prior_art_matrix.md`, `coordinator_disposition.md`, `trackB_selected_candidate.md`, `final_architecture_decision.md`, `neutral_problem_brief.md`
- `docs\plans\wl77a\`: `coordinator_decision.md`, `candidate_H3.md`, `review_A_physics.md`, `review_B_novelty_systems.md`, `responses.md`, `adjudication_draft.md`, `specialist_physics.md`
- `docs\`: `FNTO_ARCHITECTURE_CONTRACT_V0_1.md`, `Architecture.md`, `RESEARCH_ROADMAP.md`, `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`
- `docs\worklogs\`: `56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md`, `76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md`
