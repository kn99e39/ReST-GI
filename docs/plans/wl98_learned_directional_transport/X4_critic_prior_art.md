# WL98 — Adversarial critic X4: prior art and novelty

**Date.** 2026-11-16. **Role.** Adversarial critic X4 of the design-only batch WL98. Lens: prior art and novelty.

**Scope.** Critique of the four WL98 design documents in this directory, all read in full:
- `DESIGN_REVIEW.md` (coordinator draft: primary FLS, alternative ECRO-FC);
- `D1_physics_measure_first.md` (XQT, LLT);
- `D2_learning_generalization_first.md` (LIEM-S, LIEM-B);
- `D3_systems_cost_first.md` (RB-DTO, LER-Δ).

Nothing was implemented, trained or rendered. No existing file was changed. No scratch calculation was run; this critique rests on literature, the four designs, and a check of the ECRO generator code (`experiments/ecro86/model86.py:83-104`).

**Labels.**
- [MEAS] a measured value with its source.
- [AUDIT] a WL97 interpretation.
- [PRED] my own prediction or argument.
- [HUM] a human decision.
- [LIT] a literature fact. The URL was checked by web search in this session unless it is marked *(URL from memory, unverified)*.

**Bottom line.**
1. **Every candidate's core mechanism already exists in the literature under another name.** What each design adds for ReST-GI is a *contract*: passivity, flux measure, K / C ownership, rigid certificate. None adds a new transport mechanism.
2. **The coordinator's primary, FLS, does not survive.** In GM1 its learned part is a learned fit of element-local ER × incident field into lobes. Its delivery is an existing explicit method: virtual spherical lights / SG lights with explicit visibility.
3. **XQT, LLT, ECRO-FC and LER-Δ do not survive as learned-transport candidates.**
   - XQT is a learned Monte Carlo kernel denoiser.
   - LLT is learned visibility interpolation, in the WL52 class.
   - ECRO-FC keeps the visibility-coupled ECRO generator.
   - LER-Δ is a per-scene neural direct-to-indirect transfer plus negative-energy radiosity.
4. **Two candidates survive only CONDITIONALLY, and each needs a human decision that changes scope or canon.**
   - RB-DTO is close to PRT self-transfer and to object-centric scattering functions (OSF, TransGI), and it needs a canon change for self-path ownership.
   - LIEM-S is close to the learned shape-adaptive BSSRDF of Vicini et al. 2019. It sits in a class the project's novelty rule names explicitly, and it answers a different question from the brief's.
5. **Proposed verdict: NO_DEFENSIBLE_CANDIDATE** under the current canon and the brief's exclusions (§7). If the owner grants the self-coupling ownership refinement (D3 decision D1), RB-DTO becomes the one candidate worth a design-ready review, and its novelty must then be established against OSF, PRT and TransGI as measured external baselines.

---

## 1. Prior-art map

The table covers the works closest to the candidates. Nothing in it is ReST-GI evidence.

| ID | work | what it does (as relevant here) | URL |
|---|---|---|---|
| P1 | Immel, Cohen, Greenberg 1986, *A radiosity method for non-diffuse environments* | directional radiosity: per-patch outgoing radiance on a discretised hemisphere, glossy secondary sources (mirror → wall) | https://history.siggraph.org/?p=116032 |
| P2 | Christensen, Stollnitz, Salesin, DeRose 1996, *Global illumination of glossy environments using wavelets and importance* | multi-resolution (wavelet) directional basis for glossy radiance and the transport operator | https://graphics.pixar.com/people/derose/publications/Glossy |
| P3 | Sloan, Kautz, Snyder 2002, *PRT* | per-object precomputed transfer incl. self-shadowing and **self-interreflection**, applied to incident lighting at runtime; reused under rotation | https://jankautz.com/publications/prtSIG02.pdf |
| P4 | Sloan, Liu, Shum, Snyder 2003, *Bi-scale radiance transfer* | macro / meso transfer incl. self-interreflection | https://www.microsoft.com/en-us/research/publication/bi-scale-radiance-transfer/ |
| P5 | Ng, Ramamoorthi, Hanrahan 2003, *All-frequency shadows (wavelet PRT)* | non-linear wavelet approximation for all-frequency transfer | https://cseweb.ucsd.edu/~ravir/papers/allfreq/ |
| P6 | Zhou et al. 2005, *Precomputed shadow fields* | scene-independent per-object source radiance and occlusion fields, composed at runtime for moving objects | https://history.siggraph.org/?p=113965 |
| P7 | Kristensen, Akenine-Möller, Jensen 2005, *Precomputed local radiance transfer* | PRT for local lights, glossy, full GI | https://fileadmin.cs.lth.se/graphics/research/papers/plrt2005 |
| P8 | Pan et al. 2007, *Precomputed radiance transfer field for rendering interreflections in dynamic scenes* | per-object **radiance transfer fields**, independent of scene configuration, give interreflections between moving rigid objects | https://diglib.eg.org/handle/10.2312/CGF.v26i3pp485-493 |
| P9 | Iwasaki et al. 2007, *PRT for dynamic scenes taking into account light interreflection* | rigid objects as secondary sources in dynamic scenes | https://diglib.eg.org/handle/10.2312/EGWR.EGSR07.035-044 |
| P10 | Sun et al. 2007, *Interactive relighting with dynamic BRDFs* | multi-bounce precomputed transfer tensors (indirect light non-linear in the BRDF) | https://history.siggraph.org/?p=115336 |
| P11 | Loos et al. 2011, *Modular radiance transfer* | transport precomputed once on simple shapes and warped / combined at runtime: a *shared* operator over shapes, no per-scene precompute | https://cs.dartmouth.edu/wjarosz/publications/loos11modular.html |
| P12 | Hašan, Pellacini, Bala 2006, *Direct-to-indirect transfer* | static-scene precomputed multi-bounce linear transfer (sparse wavelet) from direct to indirect, glossy | https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting |
| P13 | George, Sillion, Greenberg 1990, *Radiosity redistribution for dynamic environments* | edits corrected by shooting **positive and negative energy** over the existing solution | https://en-cg-web.coecis.cornell.edu/pubs/1990/GSG90.html |
| P14 | Dachsbacher et al. 2007, *Implicit visibility and antiradiance* | negative (anti-)radiance to cancel unblocked transport; directional bins (64–512) per sample | https://history.siggraph.org/?p=115370 |
| P15 | Hašan et al. 2009, *Virtual spherical lights (VSL)* | glossy secondary senders as lobes spread over a solid angle, delivered with explicit visibility | https://www.cs.cornell.edu/~kb/projects/VSL |
| P16 | Tokuyoshi 2015, *Virtual spherical Gaussian lights* | SG-lobe secondary lights with analytic one-bounce glossy evaluation, real time | https://diglib.eg.org/handle/10.1111/cgf12748 |
| P17 | Huo et al. 2020, *SG-based lightcuts for glossy interreflections* | hierarchical clustering of SG glossy virtual lights | https://diglib.eg.org/handle/10.1111/cgf14011 |
| P18 | Xu et al. 2014, *Interreflections with all-frequency BRDFs* | SG-represented BRDFs, analytic triangle-to-point one-bounce glossy transfer | https://cg.cs.tsinghua.edu.cn/people/~kun/interreflection |
| P19 | Wang et al. 2009, *All-frequency rendering of dynamic, spatially-varying reflectance* | SG lobes fitted to the NDF (fitting, not learning) | https://www.microsoft.com/en-us/research/?p=159362 |
| P20 | Currius et al. 2020, *SG light-field textures* | **a CNN predicts per-point SG lobe parameters** of the precomputed local light field (static scene) | https://www.cse.chalmers.se/~ramon/sg-lightfields.html |
| P21 | Heitz et al. 2016, *Linearly transformed cosines* | lobe family with closed-form integration over polygons | https://history.siggraph.org/?p=102764 |
| P22 | Kaplanyan, Dachsbacher 2010, *Cascaded light propagation volumes* | directional (SH) state propagated explicitly between cells | https://informatics.tuwien.ac.at/news/329 |
| P23 | Vorba et al. 2014, *On-line parametric mixture models* | per-region vMF / Gaussian mixtures of the incident distribution, fitted from samples | https://cgg.mff.cuni.cz/~jaroslav/papers/2014-onlineis/index.htm |
| P24 | Ruppert, Herholz, Lensch 2020, *Parallax-aware mixtures* | spatially displaced (parallax-aware) lobe mixtures fitted robustly from samples | https://uni-tuebingen.de/en/177303 |
| P25 | Dong, Wang, Li 2023, *Neural parametric mixtures for path guiding* | a neural field decoded into parametric lobe mixtures, spatio-directional, trained from noisy MC | https://neuropara.github.io |
| P26 | Bako et al. 2017, *Kernel-predicting convolutional networks (KPCN)* | a learned per-pixel normalised kernel over noisy MC estimates | https://graphics.pixar.com/library/MLDenoisingB |
| P27 | Gharbi et al. 2019, *Kernel-splatting network* | a learned **per-sample splat kernel**: each sample is distributed onto neighbouring pixels | https://groups.csail.mit.edu/graphics/rendernet |
| P28 | Zhu et al. 2020, *Deep photon mapping / deep KDE* | learned per-photon kernels for density estimation; same quality with about 10× fewer photons | https://arxiv.org/abs/2004.12069v1 |
| P29 | Vicini, Koltun, Jakob 2019, *Learned shape-adaptive SSS* | shared CVAE conditioned on local shape features and medium properties; predicts absorption and the exit-position distribution; trained on a volumetric path tracer; one model for all shapes | https://rgl.epfl.ch/publications/Vicini2019Learned |
| P30 | Leonard, Höhlein, Westermann 2021, *Learning multiple-scattering solutions for sphere-tracing* | learned CVAE macro-steps that replace internal random walks inside a sphere | https://diglib.eg.org/handle/10.1111/cgf142623 |
| P31 | Hadadan, Chen, Zwicker 2021, *Neural radiosity* | per-scene neural 4D radiance solving the rendering-equation residual; non-diffuse | https://saeedhd96.github.io/neural-radiosity |
| P32 | Rainer et al. 2022, *Neural PRT* | PRT-inspired network architectures for static scenes under dynamic lighting | https://diglib.eg.org/handle/10.1111/cgf14480 |
| P33 | Gao, Mu, Xu 2022, *Neural global illumination* | per-scene network: direct → indirect under dynamic area lights, static scene | https://cg.cs.tsinghua.edu.cn/people/~kun/2022neuralGI/NeuralGlobalIllumination.pdf |
| P34 | Müller et al. 2021, *Neural radiance caching* | online-trained cache (retrains while rendering) | https://arxiv.org/abs/2106.12372 |
| P35 | Guo et al. 2020, *Object-centric neural scene rendering (OSF)* | **per-object learned scattering function, intrinsic to the object, plus non-learned path tracing between objects; rearrangement needs no retraining** | https://arxiv.org/abs/2012.08503v1 |
| P36 | Deng, Han, Fang 2025, *TransGI* | object-centric neural transfer per object, probe-based inter-object bounces, object movement / addition without retraining | https://arxiv.org/abs/2506.09909 |
| P37 | Xu et al. 2025, *A generalizable light-transport 3D embedding for GI* (arXiv 2510.18189) | scene-generalizable transformer transport embedding, from geometry + materials + area lights | https://arxiv.org/abs/2510.18189v1 |
| P38 | Verbin et al. 2022, *Ref-NeRF IDE* | continuous directional encoding of a roughness-dependent reflected-direction distribution | https://dorverbin.github.io/refnerf |

**Prior art the coordinator's E12 omits that bears directly on the candidates:**
- P15–P18 for FLS (glossy lobe senders with explicit visibility are standard many-light practice);
- P35 OSF and P11 MRT for the ownership composition and shape-shared operators;
- P26–P28 for XQT;
- P29–P30 for LIEM;
- P12–P14 for LER-Δ.

E12 groups "Wang 2009" under learned lobe prediction, but it is SG *fitting*; Currius 2020 (P20), which E12 also lists, is the learned case.

E12's description of P37 as "mainly diffuse" is correct (checked on re-run): the paper demonstrates diffuse GI (irradiance) across about 14k indoor scenes and reports only *preliminary* spatial-directional radiance estimation for glossy materials, used to guide an unbiased path tracer. Its adaptation to new rendering tasks uses limited fine-tuning, i.e. retraining of the head. It is the closest precedent for "one shared model across unseen scenes", and it does not cover sharp glossy interreflection.

**The novelty rule in force.** `docs/RESEARCH_CENTRIC_TOPIC.md:48`: "Novelty cannot rest on a hierarchy, selective updating, per-region neural models, BSSRDF approximation or operator compression alone. A research claim needs correct physical ownership, demonstrated validity under edits and measured quality/cost relative to appropriate prior art."
- Consequence: **no plan in WL98 includes the closest prior-art method as a measured comparator.** Under this rule, novelty cannot be judged by any of the plans as written (§6, defect X-1).

---

## 2. Cross-cutting finding: the K / C ownership composition itself is prior art

The ReST-GI reuse story combines three pieces:
- a learned, object-intrinsic response;
- explicit, current-geometry coupling between objects;
- reuse of that response under rigid motion without retraining.

That combination is published:
- **OSF (P35):** a per-object learned scattering function, intrinsic to the object, with non-learned path tracing between objects and no retraining on rearrangement.
- **TransGI (P36):** object-centric neural transfer, with object movement handled without retraining.
- **Non-neural precursors, all with interreflection between or within rigid objects:** PRT self-transfer (P3), radiance transfer fields (P8), shadow fields (P6) and Iwasaki 2007 (P9).

So "reuse after a rigid edit without retraining because ownership is split" is **not** a novelty claim available to any WL98 candidate. What can be new is narrower:
- **(a)** high-frequency (glossy, α ≈ 0.03) transport, where PRT (low-order SH / wavelet), OSF and TransGI are weak. TransGI states glossy↔glossy as its weakness; the coordinator's E12 notes this.
- **(b)** a *passive, measure-defined* operator contract;
- **(c)** one shared model across unseen assets, i.e. no per-asset fit. P11 (shapes) and P29 / P37 (shared across shapes / scenes) are partial precedents.

Any candidate whose learned part is not a high-frequency, non-local, rigid-invariant transport quantity has nothing left to claim once these prior works are taken into account.

---

## 3. Coordinator draft (`DESIGN_REVIEW.md`)

### 3.1 FLS — flux-allocated spatio-angular lobe state (coordinator PRIMARY)

**Closest prior art.**
- Delivery of per-element glossy lobes with explicit visibility: virtual spherical lights (P15), virtual SG lights (P16), SG lightcuts (P17), SG triangle-to-point glossy transfer (P18).
- Directional state iterated with explicit coupling: directional radiosity (P1, P2) and LPV (P22).
- A learned predictor of per-point SG lobe parameters: Currius 2020 (P20).
- Lobe mixtures fitted from incident samples, including spatially displaced (parallax-aware) lobes: P23–P25.
- Lobe density with an exact projected integral over polygons: LTC (P21).

**What is new relative to that work.** Three things:
1. the albedo-bounded π-softmax-with-NULL allocation, which makes the lobe state passive by construction;
2. lobes normalised over the *clipped* hemisphere;
3. training of the lobe *closure* against downstream delivery error rather than against the lobe fit.

Item 1 is a constraint pattern, not a transport mechanism. The prior art above uses normalised lobes times a flux weight, which is the same measure discipline. Item 2 fixes a convention. Item 3 is task-loss training of a mixture reducer.

**Relabel flag: YES.** In GM1, FLS is a VSL / SG-light glossy many-light or directional-radiosity solver whose lobe-fitting step is learned (P20-style).

**Fatal objections.**
- **F-FLS-1. The brief excludes this form of learned component.**
  - In GM1, every element is a planar or convex surface with pointwise frozen ER (E10). The physical outgoing field of an element given its delivered incident field is exactly ∫ f_ER(ω_i, ω_o) L_in(u, ω_i) dω_i⊥: ER applied to the incident deposits.
  - The coordinator's own list of what Λ_θ owns shows this. Item (1) is "element-local scattering, of which ER is the physical model". Item (2), sub-element structure, is exact once deposit positions are known, which explicit C records. Item (3) is fitting K lobes to that exact field.
  - The learned object is therefore a **compression of element-local ER × incident field into lobes**. That is "local BRDF compression" with a spatial wrapper, which the brief forbids. D1 (§4.6, C1), D2 (§6.5, R-C) and D3 (§3.5) each independently reject exactly this candidate for this reason.
  - The coordinator's own HD-1 asks the human whether to admit it, which concedes the point. A design that needs the owner to relax a stated brief prohibition is not a defensible candidate under that brief.
- **F-FLS-2. The learning-free twin is the published method.**
  - ELT (moment-matched lobes from exact ER, delivered by the same cubature) is a VSL / VSGL-type explicit glossy lobe sender (P15, P16).
  - Gate A (WL95 canon) treats the learning-free explicit-sample class as a comparator, not the method. If ELT ≈ FLS, FLS is decorative.
  - If FLS > ELT, the gain is better lobe *fitting* (P20 / P25 class). That is not a new transport representation, and the novelty rule excludes "operator compression alone".

**Serious objections.**
- **S-FLS-1. Superposition is lost, which regresses from ECRO and from all PRT prior art.**
  - The coordinator admits that the closure is not additive across sources. PRT, shadow fields, direct-to-indirect transfer and ECRO all keep exact linearity. Light transport is linear, and dropping linearity forfeits relighting and source-update reuse, the one measured learned advantage (WL87, source update 3.5–3.9×).
  - The suggested remedy, per-source states, makes cost scale with the number of lights. That is the explicit comparator's cost structure.
- **S-FLS-2. The lobe family fails where FLS claims its advantage.**
  - D2's scratch fit (§5.1) on the frozen S1 ER lobe shows generic Gaussian / Student-t mixtures with K ≤ 4 at 0.35–0.6 lobe L1 at grazing, with 2–7× peak overshoot. A flux-conserving adaptive quadtree reaches about 0.11 at 1,024 leaves.
  - FLS claims its advantage exactly at grazing and below the 8×32 cell width. Its own proposed STOP rule (optimal-fit K ≤ 8 vs FC 8×32) is likely to fire on WG [PRED]. That is good for the plan but bad for the candidate.
- **S-FLS-3. Rigid reuse is weight reuse only.** All lobe state is regenerated after any edit, because the incident field changes. This is the ECRO "rule reuse" form, which D3 §2.3 shows has no geometry-edit cost lever on GM1. Nothing rigid-invariant and non-local is cached.
- **S-FLS-4. Cubature at occlusion edges (F2) means occluded lobe delivery is explicit.** This is the VSL / VSGL visibility problem, with the same remedy: more rays.

**Strengths.**
- It has a clean flux / passivity contract.
- Deterministic delivery addresses the X1 single-draw variance problem (E5).
- Its contract tests (T-a … T-r) are good.
- **As an explicit comparator (ELT), the design is valuable**: the VSL-type arm the plans otherwise lack.

**Survives: NO** as a learned-transport candidate. Its explicit form (ELT) should stay as a comparator.

### 3.2 ECRO-FC — independent alternative

**Closest prior art.**
- Flux-conserving per-patch directional tables: P1, P2, P14 (antiradiance uses 64–512 directional bins per sample).
- The learned geometry-conditioned generator is neural-PRT / neural-radiosity class (P31–P33), applied to a scene family.

**What is new.** Nothing beyond ECRO v0. The changes are a carrier convention, a coverage change and a passivity constraint.

**Relabel flag:** not applicable. It is a lineage control, not a new method.

**Fatal objection.**
- **F-ECRO-FC-1. Dynamic visibility stays coupled into the learned "local" factors.**
  - I verified the code: `factors()` builds the dense form-factor matrix `Ft` from the current state rays, and its row sums give the openness input `opn` (`model86.py:83-95`). Message passing `Ft @ msg(t)` and a scene-global mean token `glb(t).mean(0)` (`:97-99`) feed every per-patch factor c and a. The inputs include world centre, normal and tangent (`:95`).
  - The per-patch "local" response is therefore a function of current scene visibility and of the whole scene. That is the brief's unsuitability criterion 3, and it also breaks rigid invariance (D1 §4.7; D3 §3.3).
  - ECRO-FC changes only the carrier, coverage and passivity, so it keeps this coupling. As an *alternative candidate* it is dead on a stop condition. As an H1 / H2 attribution arm it is legitimate.

**Serious objections.**
- **S-ECRO-FC-1. The arm confounds three treatments.** It changes the carrier (FC, adaptive 4×16 → 8×32), coverage and passivity together. The plan reads "ECRO-FC vs ECRO + coverage" as H2, but that comparison also changes passivity. A factorial design is needed, or at least a carrier-only arm (D1's C1f does this for the convention).
- **S-ECRO-FC-2. It cannot carry a utility result.** The generator is O(N²) dense and, on the kernel basis, costs more per edit than the explicit work it replaces (D3 §2.3: 0.91–0.97) [ARITH, D3].

**Strengths.** It is the cleanest within-lineage H1 / H2 attribution arm.

**Survives: NO** as a candidate. Keep it as a diagnostic arm.

### 3.3 Coordinator rejections of A and B

I agree with both rejections.
- The local form of A is BRDF compression. Ref-NeRF's IDE (P38) is an appearance encoding and adds no transport content.
- The non-local form of A is learned C (WL52).
- The learned form of B is dominated by its learning-free twin (E4; D2 §5.1).

---

## 4. D1 (`D1_physics_measure_first.md`)

### 4.1 XQT — exact-sample quadrature transport (D1 PRIMARY)

**Closest prior art.**
- Learned per-sample splat kernels: Gharbi 2019 (P27).
- Learned per-pixel normalised kernels: KPCN (P26).
- Learned kernel density estimation of particle flux: deep photon mapping (P28), which reports 10× fewer photons for equal quality. That is exactly XQT's claimed lever (fewer relation rays for equal error).

**What is new.**
- The kernel acts in (receiver patch × projected-disk direction cell) space instead of image space.
- Its normalisation is per sample (column-stochastic), so flux is conserved. P27 / P28 normalise per output, i.e. at the gather.
- It masks pooling by explicit visibility records.

These are a domain transfer of a learned MC estimator plus a conservation constraint.

**Relabel flag: YES.** XQT is a learned Monte Carlo denoiser / kernel density estimator for the flight integral. D1 says so itself ("the defensible learned transport is an estimator").

**Fatal objections.**
- **F-XQT-1. It is not a learned transport representation.**
  - It holds no transport state. ER is exact, visibility is explicit, and the learned part redistributes already-computed explicit samples.
  - The brief asks whether *learned* transport can be reused instead of explicit transport. XQT keeps all explicit transport and learns its variance reduction: "explicit transport plus a learned filter".
  - It meets unsuitability criterion 1 ("does not substantially replace or reuse explicit transport"). Gate A (WL95) classifies the explicit exact-sample class as a comparator. A learned smoother on that class does not change the class.
- **F-XQT-2. Its own scratch result predicts no learning margin.** D1 §9: twin / oracle error ratio 1.00–1.33 for an oracle per-element kernel. The learning-free twin captures almost all of the 8–50× pooling gain. D1 itself predicts a Stage-1 stop. The isotropic limitation is acknowledged, but edge-aware learning-free filters (cross-bilateral, à-trous with geometry guides) are the standard non-learned answer to that, and they are not in the twin set (defect D1-P1).

**Serious objections.**
- **S-XQT-1. The explicit comparator is the weakest one available.** XQT's η is measured against the unpooled receiver gather X1. D3 §2.4 notes that sender-side deposition (C3 / C3UR / photons) removes about 90 % of the receiver-placement deficit at roughly R0 cost [MEAS WL87 / WL89]. For glints, a receiver-anchored gather is the wrong explicit baseline.
- **S-XQT-2.** "Reuse without retraining" is trivially satisfied by every pretrained denoiser (P26–P28 generalise across scenes). It is not a ReST-GI contribution.

**Strengths.**
- It has the strongest physical contract in the batch: linear, flux-exact, passive, rigid-invariant, and no visibility inference.
- Its learning-free S1-c gate is decisive and cheap.

**Survives: NO** as a learned-transport candidate. It is an admissible variance-reduction comparator.

### 4.2 LLT — learned lobe transfer from sparse probes (D1 ALTERNATIVE)

**Closest prior art.**
- Lobe footprint delivery: VSL (P15) and SG lights (P16, P18).
- Interpolating visibility from sparse probes: cone tracing / voxel cone tracing (Crassin et al. 2011, *URL from memory, unverified*: https://research.nvidia.com/publication/2011-09_interactive-indirect-illumination-using-voxel-cone-tracing).
- Learned delivery and visibility: the WL52 learned C (in-project negative).
- Learned per-photon footprint kernels: P28.

**What is new.** A normalised sender-anchored learned form factor with a miss class, conditioned on stratified probes.

**Relabel flag: partly.** It is learned C restricted to sharp senders.

**Fatal objection.**
- **F-LLT-1. To beat its twin it must infer unqueried visibility.** Canon gives visibility to explicit C, and WL52 failed exactly at contacts and transitions.
  - The S1-d information test is circular: collision pairs are drawn from the *training generator*. It certifies only that the generator lacks occluders smaller than the probe spacing, a distribution prior that the held-out set can violate at will.
  - This is the brief's criterion 3 in its learned-C form. D1 marks it partial.

**Serious objections.**
- **S-LLT-1.** Its cost is per (node × light), which is the explicit lobe tracer's cost structure.
- **S-LLT-2.** It has no reuse beyond probe certificates.

**Strengths.**
- S1-d is a sound learning-free information test in form.
- It is the only design that directly attacks "learned delivery beyond queries".

**Survives: NO.**

### 4.3 D1 rejections (A1, A2, B1, C1, E0, O1)

I agree with all six. One addition on O1, the asset-level reflectance-field K: it is the OSF (P35) / radiance-transfer-field (P8) / 8DNA class. Under the baseline gate, its rigid case is Native Reuse, as D1 says.

---

## 5. D2 (`D2_learning_generalization_first.md`)

### 5.1 LIEM-S — Learned Intrinsic Exit Measure (D2 PRIMARY)

**Closest prior art.**
- **Vicini, Koltun, Jakob 2019 (P29)** is a shared conditional generative model:
  - conditioned on local shape features and medium properties near the entry point;
  - predicts absorption and the exit-position distribution;
  - trained on brute-force volumetric path tracing;
  - one model for all shapes, with code public.
- **Leonard, Höhlein, Westermann 2021 (P30)**: learned macro-steps that replace internal random walks. This is LIEM's "amortise Twin-W" lever.
- Classical twins: dipole, Burley and directional dipole (Frisvad et al. 2014, *URL from memory, unverified*: https://doi.org/10.1145/2682629).

**What is new relative to P29 / P30.**
1. The internal-direction density ν passes through an exact rough-dielectric closure. P29 handles the outgoing direction separately.
2. Analytic through-object and TIR anchors.
3. A passivity bound by construction (m_c ≤ 1 − m_bal).
4. A designed impulse measure with a grazing quota.
5. Integration into FNTO's K / C ownership.

Items 1 and 2 are real but incremental. Items 3 and 4 are contract hygiene. D2 flags the prior art itself but lists it as unverified; it is now verified, and the match is close.

**Relabel flag: YES, mostly.** LIEM-S is a learned shape-adaptive BSSRDF with refinements. `RESEARCH_CENTRIC_TOPIC.md:48` names "BSSRDF approximation" as insufficient for novelty.

**Fatal objection: none on the brief's stop list.** The contract holds within its scope.

**Serious objections.**
- **S-LIEM-1. It does not test the central hypothesis.**
  - The brief asks whether *learned high-frequency, spatially distributed* transport can be reused. In LIEM-S the high-frequency part, the sharp exit lobe, is produced by the **exact** interface closure. The learned part is the collided exit measure.
  - D2's own toy (§5.2) shows that this measure is angularly smooth wherever amortisation pays (27–50 events per exit). It is high-frequency only where walks are short (2.5–12 events).
  - The candidate therefore learns low-frequency transport plus anchors and leaves sharpness to exact physics. That is close to the pattern the brief warns against, "sharp to explicit, learned residual", relocated inside K.
- **S-LIEM-2. It changes the workload family to GM2 media** (HD2). It also needs a refractive-interface medium reference that does not exist: WL53 validated only index-matched media; Cycles BSSRDF has path-history semantics (WL51); Mitsuba float32 skips about 89 µm at boundaries.
- **S-LIEM-3. The learned-K lineage's shape-generalisation record is negative** (WL68, WL71–WL80), and LIEM inherits that risk.
- **S-LIEM-4. The twins exclude the closest prior art.** Twin-D (dipole / Burley) is a weak twin. The honest external comparator is P29 itself, which has public code. Without it, "learning beats Twin-D" says nothing about novelty (defect D2-P1).

**Strengths.**
- It is the most rigorous learning design: impulse coverage as a treatment, NLL on unbiased walk events, and exact rigid equivariance.
- Its headroom gate (I0.2) can stop it before training.

**Survives: CONDITIONAL**, on HD2 (GM2 scope) and HD10 (novelty acceptance). Even then it answers "learned SSS reuse", not the brief's sharp-interreflection question.

### 5.2 LIEM-B — hierarchical categorical exit distribution (D2 ALTERNATIVE)

**Closest prior art.**
- Hierarchical spatio-directional trees fitted from samples: practical path guiding SD-trees (Müller et al. 2017, *URL from memory, unverified*: https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf).
- Tabulated BSSRDF (Twin-T).

**New.** A shared network predicts the hierarchy.

**Relabel flag:** representation variant of LIEM-S.

**Fatal objection.** D2 admits it is not mechanism-independent. The brief asks for an *independent* alternative, so D2 has none.

**Serious objection.** Its sharp exit lobe needs α-resolution cells (the α⁻² growth) unless the internal-cone variant is used. That variant is LIEM-S again.

**Survives: CONDITIONAL**, only as LIEM-S's H2 control. It is not a valid alternative.

### 5.3 D2 rejections (R-A … R-E)

I agree with all of them.
- R-D (rigid-cluster self-transport) is the same class as D3's RB-DTO. D2 rejects it for canon conflict and weak learned value, which is a direct disagreement with D3 that the owner must resolve (§7).

---

## 6. D3 (`D3_systems_cost_first.md`)

### 6.1 RB-DTO — Rigid-Body Directional Transfer Operator (D3 PRIMARY)

**Closest prior art.**
- PRT self-transfer, incl. self-interreflection, reused under rigid rotation: P3 and P4. All-frequency extensions: P5 and P18.
- Per-object transfer fields for dynamic rigid scenes with interreflection: P8 and P9.
- **OSF (P35)**: a per-object learned scattering function intrinsic to the object, with explicit path tracing between objects and no retraining on rearrangement. This is RB-DTO's ownership composition, with per-object training instead of a shared compiler.
- TransGI (P36), object-centric neural transfer.
- A shared operator over shapes: MRT (P11), low-frequency.
- A shared model across scenes: P37.

**What is new relative to that work.**
1. *Sharp* glossy self-transport (α ≈ 0.03–0.15), carried as a passive, measure-valued lobe pushforward per incident sample. PRT is low-frequency or wavelet-compressed, and OSF and TransGI are weak on glossy↔glossy.
2. One *shared* compiler for unseen bodies, with no per-asset path tracing.
3. A self-path-region certificate with explicit fallback. Shadow fields and PRT ignore intrusion, and OSF does not certify it.
4. A path-tagged two-channel (sharp / broad) supervision.

Items 1 and 2 together are the only novelty niche I found in WL98 that is not already occupied. Each alone has a precedent (P5 / P18 for all-frequency transfer; P11 / P29 / P37 for shared models).

**Relabel flag: PARTIAL.** The core is learned PRT / OSF self-transfer. The novelty would have to come from sharpness plus sharing, demonstrated against those baselines.

**Fatal objections, conditional on canon.**
- **F-RB-1. It needs a canon change.**
  - Architecture §2 assigns exposed same-asset free-space paths to C: "Object identity must not freeze exposed interreflection". RB-DTO moves C_self into a body operator. D1 (O1) and D2 (R-D) reject the same class on this ground. D3 makes it HUM D1 and says that without it RB-DTO is inadmissible.
  - It is therefore not a defensible candidate *under the current canon*. It becomes one only if the owner rules for it. This is not softened: the brief says canonical decisions belong to the user, and the batch may not adopt a new architecture.

**Serious objections.**
- **S-RB-1. The learning-free twin is a straw man.**
  - Twin 2 (precomputed per-body transfer) is costed with **uniform** α⁻² binning (0.17–13.7 GB). The prior-art twin is compressed all-frequency precomputed transfer: Haar-wavelet or SG transfer (P5, P18, P10), or the CPCA-compressed transfer fields of P8.
  - A per-body compressed explicit transfer is equally rigid-invariant and equally reusable, and needs no learning. Against it, learning buys only (2) *no per-asset precompute*.
  - D3's regime-map utilities (4.8–48×) use placeholder constants against uniform binning. D3 says so, but the claims are not supported by them.
- **S-RB-2. The workload is chosen for the claimed lever.** The admission clauses (self-transport share ≥ s_min; a cavity edit) select for the candidate. That is legitimate only with a declared adversarial workload and scoped generality (defect D3-P2).
- **S-RB-3. The output is multimodal.** Sharp multi-bounce glossy self-transport in a groove or corner reflector produces several separated exit lobes per incident sample. K_s lobes must cover that multimodality. The D2 §5.1 grazing mixture failure applies at horizon-truncated exits.
- **S-RB-4. Its benefit is zero on GM1.** Every GM1 body is planar or convex, so N_b ≡ 0 (D3's own contract test). The candidate cannot touch the WL86–WL96A sharp failure, which was inter-body.

**Strengths.**
- It is the only candidate in the batch whose learned part is (i) non-local, (ii) rigid-invariant transport content rather than weights only, (iii) high-frequency and (iv) a replacement for explicit iterated sharp work. That matches the brief's central hypothesis better than anything else in WL98.
- Its energy contract is explicit, with a defined measure.
- It has a convex-body null test.
- Cost anatomy is gated before training.

**Survives: CONDITIONAL**, on the HUM self-coupling canon refinement. It also needs compressed precomputed transfer as Twin 2, and OSF / PRT / TransGI-class external baselines measured or explicitly UNMEASURED.

### 6.2 LER-Δ — learned environment resolvent with explicit edit-delta (D3 ALTERNATIVE)

**Closest prior art.**
- Static-scene precomputed multi-bounce transfer: Hašan 2006 (P12).
- Per-scene neural indirect transfer: P32 and P33.
- Neural radiosity, per scene (P31).
- Edit correction by **negative energy**: George, Sillion, Greenberg 1990 (P13) and incremental radiosity (Chen 1990, *URL from memory, unverified*: https://doi.org/10.1145/97880.97905).
- Negative radiance for blocked paths: antiradiance (P14).

**What is new.** Directional, glossy lobes in the resolvent. The composition "precomputed static resolvent + explicit ±Δ over dirty paths" is P12 + P13.

**Relabel flag: YES.**

**Fatal objections.**
- **F-LER-1. It is per-environment learned state.** That is per-scene neural precompute, the Native Reuse / RNA / 8DNA class under the baseline gate, not a shared learned rule. D3 itself flags it as HUM D4.
- **F-LER-2. The Δ-iteration carries negative flux.** −B makes ρ(H_E(M − B)) < 1 unguaranteed, and AGENTS.md forbids clamps.
- **F-LER-3. H_E is applied to Δ fields outside its training span.** Passivity is by construction, but fidelity is not, which is the WL41 / WL44 failure mode.

**Strengths.** An honest identification of the second rigid-invariant amortisation target.

**Survives: NO.**

### 6.3 D3 rejections

I agree with all of them, including NRC (P34: it retrains online, which violates "without retraining").

---

## 7. Proposed verdict

**NO_DEFENSIBLE_CANDIDATE**, under the current canon (Architecture §2 ownership of exposed same-asset paths) and the brief's exclusions. This is not a class-level falsification.

**Rationale.**
- FLS, XQT, LLT, ECRO-FC and LER-Δ each fail a brief stop condition or exclusion:
  - BRDF compression (FLS);
  - no learned replacement of explicit transport (XQT);
  - learned visibility (LLT);
  - visibility coupled into the learned factors (ECRO-FC);
  - per-scene state with unguaranteed convergence (LER-Δ).
  Each is also an existing method relabelled.
- The two survivors need a human decision before they are candidates:
  - **RB-DTO** needs a canon refinement (certified rigid self-coupling). It is the one candidate aimed at the brief's hypothesis, and its novelty niche (sharp plus shared) is narrow and must be measured against OSF / PRT / TransGI.
  - **LIEM-S** needs a GM2 scope change and novelty acceptance in an excluded class (BSSRDF approximation), and it tests learned *low-frequency* intrinsic transport with exact sharp closure.
- Three designers independently found no defensible learned candidate in the GM1 surface-only family: D1 §1.2 (Proposition I), D2 §3 and D3 §2.5. The coordinator's FLS lives in that family.

**If the owner grants D3's decision D1:** the batch's defensible content becomes RB-DTO as the primary candidate, with LIEM-S as the only independent alternative, which in turn needs HD2. At that point **DESIGN_READY_FOR_HUMAN_REVIEW** would be appropriate, once the plan defects in §8 are fixed. That conversion is a human decision, not one this critique can make.

---

## 8. Experiment-plan defects

**Cross-design (X).**
- **X-1. No plan measures the closest prior art as a comparator**, although the project's novelty rule requires "measured quality/cost relative to appropriate prior art". The comparators should be:
  - FLS / ELT: VSL / VSGL (P15, P16) or SG triangle transfer (P18);
  - XQT: deep KDE / kernel splatting (P27, P28) and a non-learned edge-aware filter;
  - LIEM: Vicini 2019 (P29), which has public code;
  - RB-DTO: compressed per-body precomputed transfer (P5 / P8 / P18) and OSF / TransGI;
  - LER-Δ: direct-to-indirect transfer (P12).
  Without these, a "beats its twin" result cannot support a novelty or advantage claim.
- **X-2. Twins are weaker than the state of the art in each case.** Moment-matched ELT (FLS), an isotropic fixed kernel (XQT), dipole Twin-D (LIEM) and uniform-binned Twin 2 (RB-DTO) all inflate the learning margin. Each twin should be the strongest learning-free method at the same interface.
- **X-3. Reuse after rigid edit is untested where it matters.** For FLS, XQT, ECRO-FC and LLT, "fixed weights at T1" equals rule reuse: all state is regenerated. A T0 → T1 test cannot distinguish them from any pretrained network, so it is not evidence for the hypothesis's "reuse". Only RB-DTO and LIEM-S cache rigid-invariant transport state that a T1 test actually exercises.
- **X-4. The external-baseline gate is not met.** The mandatory terminology gate requires native lifecycle behaviour for each external baseline. OSF / TransGI-type object assets moved rigidly are **Native Reuse** under that gate, and only D3 (B7) and D2 (HD9) mention Native Reuse at all. All plans must list them as measured or UNMEASURED with a reason.

**Coordinator (DR).**
- **DR-P1.** ECRO-FC confounds carrier, coverage and passivity (§3.2).
- **DR-P2.** The WG / WN workloads cannot exercise FLS's claimed iterated-sharp advantage. GM1 has one sharp panel; WGG is "optional". The plan admits that the higher-order sharp regime "barely" exists in GM1, so a positive H3 for FLS is untestable as planned.
- **DR-P3.** §10 refers to `CRITIQUE.md`, which does not exist in the directory, and says the responses are "recorded in the WL98 worklog". The critique-and-response trail is not verifiable yet.
- **DR-P4.** E12 files Wang 2009 under learned lobe prediction (it is fitting) and omits P15–P18, P26–P30 and P35. Its description of P37 is correct.

**D1.**
- **D1-P1.** The XQT twin family has no edge-aware learning-free filter.
- **D1-P2.** XQT's η is measured against the receiver gather X1 rather than against sender-side deposition (C3 / C3UR / photons).
- **D1-P3.** The S1-d collision test samples the training generator, so it is circular for out-of-distribution occluders.

**D2.**
- **D2-P1.** Vicini 2019 is not an arm.
- **D2-P2.** I0.1 depends on a refractive-medium reference that does not exist yet, so the whole of Track I is gated on reference engineering. That should be stated as the first stop.
- **D2-P3.** Track G's G2 can be skipped by HD8, which leaves H1 unattributed for ECRO.

**D3.**
- **D3-P1.** Twin 2 is uniform-binned; it should be compressed (Haar / SG / CPCA).
- **D3-P2.** Track-B admission selects for the claimed lever. A declared adversarial workload is needed (moderate self-transport, frequent intrusion).
- **D3-P3.** The regime-map utilities use placeholder t_q and must not appear as evidence.
- **D3-P4.** Certificate cost and false-positive rate are not budgeted in Stage 0.

**Respected everywhere.** M64 is never used as truth; CK2 is diagnostic only; seeds ≥ 3 and draws ≥ 16 [HUM]; thresholds are left to the human.

---

## 9. Missed alternatives

- **MA-1. All-frequency compressed precomputed per-body transfer** (Haar or SG, with explicit current C between bodies; P5, P8, P18). It is learning-free, rigid-reusable and high-frequency. It is the necessary twin for RB-DTO and a possible answer to the hypothesis's "reuse" half without learning.
- **MA-2. Analytic lobe-to-polygon delivery with LTC or SG** (P21, P18) as a deterministic, variance-free twin for FLS / ELT, which uses cubature. Visibility still has to be handled explicitly.
- **MA-3. A learned sampler (path guiding) driving the explicit unbiased estimator** (P23–P25). It addresses the X1 heavy tail (E5) without owning transport or introducing bias. Its learned part is reusable after edits only partially, and it is not a transport representation, but it is the strongest learned competitor to XQT and should be named.
- **MA-4. OSF / TransGI-style per-object neural assets as the external Native Reuse baseline** (P35, P36) for any object-level candidate.
- **MA-5. Shared-shape modular transfer (MRT-style, P11) extended to glossy.** This is a precedent RB-DTO's "shared compiler" must beat to claim novelty.

---

## 10. Sources

All URLs are in §1. P37 was re-checked on the re-run of this role (2026-11-16): https://arxiv.org/abs/2510.18189v1 and https://research.nvidia.com/labs/rtr/publication/xu2026generalizable/ . Four entries were not checked in this session and are marked *(URL from memory, unverified)*: Crassin 2011, Frisvad 2014, Müller 2017 and Chen 1990. Code checked: `experiments/ecro86/model86.py:83-124`. Canon checked: `docs/RESEARCH_CENTRIC_TOPIC.md:48` (novelty rule) and the WL95 Gate A entry.
