# WL77A Specialist Report: Agent 4, Prior Art and Novelty (2026-11-07)

**Assigned role:** Agent 4, prior art and novelty kill search, in the WL77A multi-agent architecture-discovery workflow ([Worklog 77A](../../worklogs/77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md)).

**Status: COMPLETED.** This is a read-only literature review. No code, protocol, result or historical document was changed, and nothing ran on a GPU or a server. Writing this file was the only repository change.

**Research question.** A learned intrinsic light-transport operator K should stay accurate when the source position changes, including independent, previously unseen source ports near boundaries and far from supervised sources. Which information must be represented explicitly, and what can be learned as a shared operator? Only K is in scope. C, the dynamic free-space coupling, is not studied.

**Scope of this report:**
- which mechanism classes prior art already covers, so that a candidate built only from them is a combination and not a new mechanism;
- where open space plausibly remains;
- what the literature implies for the explicit-versus-learned split.

The report does not select a candidate, and it does not recommend a research direction.

---

## 1. Input sources read

### 1.1 Repository (read-only)

| Source | What was used |
|---|---|
| `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` (l. 1–40) | Baseline roles for RNA, 8DNA and TransGI. Mandatory Neural Full Recompute terminology. TransGI blocked (no code, arXiv 2506.09909). WL69 correction. |
| `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` §8 (l. 160–172) | The project's existing novelty discipline and prior-art list: RNA, Neural Light Transport, Neural SSS, Garment-fold interreflection, Gaussian-surfel radiosity. |
| `docs/RESEARCH_CENTRIC_TOPIC.md` l. 35–41 | "Novelty cannot rest on a hierarchy, selective updating, per-region neural models, BSSRDF approximation or operator compression alone." |
| `docs/worklogs/25_WORKLOG_RNA_8DNA_CODE_LEVEL_OWNERSHIP_AUDIT.md` l. 1–20, 182, 197 | Pinned checkouts (RNA `66b5b09`, 8DNA `4a2157c`). Paper IDs (RNA arXiv:2312.09398; 8DNA arXiv:2604.25129). 8DNA notation F(x_o, ω_o, x_i, ω_i) with a code direction swap. |
| `docs/worklogs/27_WORKLOG_NEURAL_FULL_RECOMPUTE_COST_RNA_8DNA.md` l. 1–40 | Neural Full Recompute: 8DNA 38.9 min to first recovery and a 1.76 h schedule; RNA 3.24 h. |
| `docs/worklogs/67_WORKLOG_BASELINE_SUPERIORITY_EVIDENCE.md` l. 20, 32–33, 124, 138, 151–165 | TransGI scope (rigid only, transparent objects removed, scene-specific training). 8DNA's Mitsuba boundary-layer bias. RNA and 8DNA cannot run the medium workload W1 natively. |
| `docs/worklogs/69_WORKLOG_COMMON_WORKLOAD_EXTERNAL_BASELINE.md` (via the strategy-document summary) | Released 8DNA runs natively on the floor-medium fixture and fails 3–7/12. Its CY arm is a Cycles re-render, not Neural Full Recompute. |
| `docs/worklogs/74A_WORKLOG_ANGULAR_MAPPING_READONLY_ANALYSIS.md` l. 31–130 | The per-block cone and outgoing hull of the CP/ATO factorisation. Exact local D4 equivariance of the bin grid; frames not equivariant across faces. |
| `docs/worklogs/75_*.md` l. 128, 186, 195, 264, 270 | A75: TRAIN_NEW con_coarse 0.131 (= G2 oracle 0.129). BND4_FAR 0.803 / 0.824 vs AR 0.793 / 0.792. Late-training drift: 0.73–0.77 at 2k steps. |
| `docs/worklogs/77_WORKLOG_FIA_ANGULAR_EXPRESSIVITY.md` l. 103–105, 137 | Projection floors: ring 0.453, FIA-AZ (order 2 relative azimuth) 0.385, free phase order ≤ 2 0.352, noise 0.0016–0.0019. STOP at Stage 1. |
| `experiments/fnto_repr68/family68.py`, `experiments/fnto_fia76/fia76.py` (function and class index only) | Confirms the code structure the task describes: ATO predict(T, d, A, O); FIA quadrature + `response` + `Runtime.apply`. Not re-audited numerically. |

### 1.2 External sources

Each source below was retrieved through WebSearch or WebFetch on 2026-11-07. Verification status is marked per entry:
- **V**: title, authors and venue confirmed on an official or primary page (arXiv abstract, publisher, author page, PMLR/NeurIPS/ICML page);
- **V-sec**: confirmed only through secondary indexes (SIGGRAPH history archive, dblp, library catalogues, aggregators);
- **S**: mechanism details come only from search snippets or abstracts, not the full text;
- **U**: unverified, from background knowledge, included only to flag a lead.

---

## 2. Mechanism map: what ReST-GI's K currently is

The map below is in mechanism-class terms, so that prior art can be placed against it. It is a paraphrase of the task brief plus the files above.

| ReST-GI element | Mechanism class |
|---|---|
| Port × bin boundary discretisation, P = 3840 ports × 16 in-bins → 3840 × 64 out-bins | Discrete boundary-to-boundary transport matrix (a BSSRDF / scattering operator of a finite medium in a piecewise-constant spatio-angular basis) |
| Ballistic B, exact and geometry-only | Uncollided flux, computed analytically or by ray tracing |
| W59/W+UV ATO: K_scat = d Σ_r A[r,a] T[i,j,r] O[r,o]; T from a geometry GNN (local edges W + low-rank UV tail), mass-normalised | Low-rank (CP) separable transport. Spatial transfer conditioned on geometry, plus a global fixed angular dictionary. Passive through mass normalisation. |
| AR / RSK (WL72) | Symmetry regularisation; relational shared kernel |
| DASQ (WL49), CAK (WL55) | Distance-adaptive hierarchical quadrature; contact-adaptive co-refinement |
| FIA (WL76) | First-collision factorisation: (1 − β) E_{x∼p1}[albedo(E1 + M)], with a learned source-independent emitter response R(anchor → receiver) and ring lobes |
| FIA-AZ (WL77) | Low-order relative-azimuth Fourier lobes |

---

## 3. Prior-art catalogue (findings with evidence)

### 3.1 Per-asset baked neural transport: the external neural baselines and their family

**RNA: Relightable Neural Assets.**
- **Citation:** K. Mullia, F. Luan, X. Sun, M. Hašan. arXiv:2312.09398 (Dec 2023, v2 Sep 2024). ACM TOG 2024, presented at SIGGRAPH 2025. **V** (arXiv / Adobe).
  - The repository cites it inconsistently: FNTO contract "TOG 2024; DOI 10.1145/3695866", WL25 "TOG 44(1) 2025". arXiv lists "TOG 2024 (Presented at SIGGRAPH 2025)". Not material to this question.
- **Link:** https://arxiv.org/abs/2312.09398
- **Mechanism:** Explicit geometry, a neural feature grid and an MLP precompute shading and scattering of one asset under far-field (directional) illumination. The asset is inserted into a path tracer.
- **Relevance:** Per-asset. Lighting is far-field directional, so there is no source-position input at the asset boundary in the 8D sense. It cannot express the source-port transfer question. Its learned state is scene-baked, which is the ownership model ReST-GI avoids (WL25).

**8DNA: 8D Neural Asset Light Transport by Distribution Learning.**
- **Citation:** L. Wu, H. Lu, B. Xu, M. Hašan, R. Ramamoorthi. arXiv:2604.25129, submitted 28 Apr 2026. **V** (arXiv abstract). The repository says SIGGRAPH 2026 (WL25 l. 18), but the arXiv page shows no venue field, so the venue is **V-sec** at best.
- **Link:** https://arxiv.org/abs/2604.25129
- **Mechanism** (arXiv HTML, read directly):
  - Learns the full 8D boundary transport F(x_o, ω_o, x_i, ω_i) of an asset "bounded by a 2D surface".
  - Factorisation F = α(x_o, ω_o) · p(x_i, ω_i | x_o, ω_o): an albedo MLP times a conditional exit distribution. The distribution is an autoregressive normalising flow with rational-quadratic splines, 32 knots per RGB channel.
  - Exit points are reparameterised onto the axis-aligned bounding box in cylindrical coordinates.
  - Encodings: triplane for x_o, cubemap for ω_o.
  - Trained "for each asset in isolation" on forward path-traced samples from random rays on a bounding sphere, with an online buffer.
  - Valid only if "no other objects can intersect with the asset's convex hull". Direct specular can be separated analytically.
- **Relevance:** This is the closest external analogue of an intrinsic K: a full near-field boundary-to-boundary operator, passive by construction (albedo × normalised distribution). Two points matter here:
  1. **Factorisation precedent.** The albedo × normalised-exit-distribution factorisation is exactly the mass-normalised passive pattern of ReST-GI's T (W+UV mass-normalised). A candidate claiming novelty for "passive via mass × simplex distribution" would collide with 8DNA, and with Vicini 2019 (§3.3).
  2. **Different question.** 8DNA has no sparse-source generalisation problem. It is supervised densely over all entry points and directions of one fixed geometry, with the incident location an explicit network input, and it does not generalise across geometry. The ReST-GI question (16 supervised source ports, held-out boundary sources, geometry-conditioned and deformable) is not tested by 8DNA.
- **Role here:** a Native Reuse / Neural Full Recompute baseline, not a mechanism source for source transfer.

**TransGI.**
- **Citation:** Y. Deng, L. Han, L. Fang. "TransGI: Real-Time Dynamic Global Illumination With Object-Centric Neural Transfer Model." arXiv:2506.09909 (11 Jun 2025). **V** (arXiv). A catalogue record lists IEEE TVCG 31(12):10457–10472, 2025; that venue is **V-sec**.
- **Link:** https://arxiv.org/abs/2506.09909
- **Mechanism:** An object-centric neural transfer model (MLP decoder + vertex-attached latent features) combined with local light probes and inter-probe radiance sharing for dynamic lighting. Under 10 ms per frame.
- **Relevance:** Object-centric precomputed transfer with runtime probe lighting is conceptually close to a K/C split, but at the surface-appearance level. No participating-media boundary operator.
- **Repository status:** blocked (no released code; rigid motion only; transparent objects removed; WL67 l. 33).

**Neural Light Transport for Relighting and View Synthesis.**
- **Citation:** X. Zhang, S. Fanello, Y.-T. Tsai, T. Sun et al. ACM TOG 40 (2021), DOI 10.1145/3446328. arXiv:2008.03806. **V-sec** (arXiv listing, MIT DSpace).
- **Link:** https://arxiv.org/abs/2008.03806
- **Mechanism:** A semi-parametric network in a texture atlas. A physically based diffuse base plus learned residuals for non-diffuse and global effects.
- **Relevance:** Precedent for "explicit physical base + learned residual", the pattern behind the "ballistic + collided" split. Per-subject, with no source-position transfer inside a medium.

**Neural Precomputed Radiance Transfer.**
- **Citation:** G. Rainer, A. Bousseau, T. Ritschel, G. Drettakis. CGF 41(2), Eurographics 2022, DOI 10.1111/cgf.14480. **V** (EG digital library).
- **Link:** https://diglib.eg.org/handle/10.1111/cgf14480
- **Mechanism:** PRT-inspired neural architectures for static scenes under dynamic environment lighting.
- **Relevance:** Neural PRT is prior art for "learned transfer operator applied linearly to the lighting". It covers far-field lighting only.

**Neural SSS: Lightweight Object Appearance Representation.**
- **Citation:** T. TG, D. M. Tran, H. W. Jensen, R. Ramamoorthi, J. R. Frisvad. CGF 43(4), EGSR 2024, DOI 10.1111/cgf.15158. **V-sec** (EG digital library, DTU orbit). The full text was not readable (HTTP 403 / 500).
- **Link:** https://diglib.eg.org/handle/10.1111/cgf15158
- **Mechanism (S):**
  - A compact network encodes the full light transport inside an object, including heterogeneous scattering, with an importance-sampling renderer under arbitrary lighting.
  - "Under a planar half-space assumption, one BSSRDF model can serve many geometries."
- **Relevance:** A direct precedent for a shared learned SSS operator reused across geometries. It obtains that sharing through the planar half-space assumption, which is exactly what fails near boundaries, edges and corners: the regime where ReST-GI's held-out far-boundary sources fail (WL75 BND4_FAR 0.80–0.82). Its per-object, non-half-space mode is baked.

**NeuPreSS / "Neural BSSRDF".**
- **Citations:**
  - NeuPreSS: "Compact Neural Precomputed Subsurface Scattering for Distant Lighting of Heterogeneous Translucent Objects", Pacific Graphics 2024, CGF, DOI 10.1111/cgf.15234. **V-sec**.
  - arXiv:2312.15711, "Neural BSSRDF: Object Appearance Representation Including Heterogeneous Subsurface Scattering" (T. TG, J. R. Frisvad, R. Ramamoorthi, H. W. Jensen). Its arXiv page links the same DOI 10.1111/cgf.15234, so it is probably the NeuPreSS preprint. That link is an inference, not confirmed.
- **Links:** https://diglib.eg.org/handle/10.1111/cgf15234, https://arxiv.org/abs/2312.15711
- **Mechanism:** An MLP over (surface position, view direction, incident direction) under distant directional light, per object.
- **Relevance:** Per-object, far-field. Not source-position transferable.

**Real-Time Neural Appearance Models.**
- **Citation:** T. Zeltner, F. Rousselle, A. Weidlich et al. ACM TOG 43(3), 2024, DOI 10.1145/3659577. arXiv:2305.02678. **V** (NVIDIA / arXiv).
- **Link:** https://arxiv.org/abs/2305.02678
- **Mechanism:** Learned hierarchical textures plus neural decoders, with two graphics priors: a transformation into learned shading frames, and microfacet importance sampling.
- **Relevance:** Precedent for learned or canonicalised local frames. This bears on WL74A's finding that per-face bin frames are not equivariant across faces.

### 3.2 Scene-level and online neural GI

| Work | Verification and link | Mechanism | Relevance |
|---|---|---|---|
| Real-time Neural Radiance Caching for Path Tracing (Müller, Rousselle, Novák, Keller). SIGGRAPH 2021, TOG 40(4), DOI 10.1145/3450626.3459812 | **V**. https://arxiv.org/abs/2106.12372 | Online self-trained radiance cache. Generalisation "via adaptation" rather than pretraining. | Explicitly avoids the pretrained-generalisation problem, the problem this research is asking about. Not an operator; not linear in the sources by construction. |
| Neural Radiosity (Hadadan, Chen, Zwicker). SIGGRAPH Asia 2021, TOG 40(6), DOI 10.1145/3478513.3480569 | **V**. https://arxiv.org/abs/2105.12319 | Network minimises the rendering-equation residual; learnable geometric features. | Solution of one scene; not a reusable operator. |
| Dynamic Neural Radiosity with Multi-grid Decomposition (Su, Dong, Ren, Jin, Chen, Wang, Li). SIGGRAPH Asia 2024 conf., DOI 10.1145/3680528.3687685 | **V-sec** (author page). https://woaixuexisr.github.io/papers/dynamic-neural-radiosity | Neural radiosity over variable-scene parameters; feature grid decomposed into 3D grids, 2D planes and MLPs. | Parametric, not source-transferable at medium boundaries. |
| Vertex Features for Neural Global Illumination. arXiv:2508.07852, SIGGRAPH Asia 2025 | **V-sec**; author order inconsistent across sources. https://arxiv.org/abs/2508.07852 | Learnable features at mesh vertices with loss-driven refinement. | A storage representation, not a transfer mechanism. |
| Neural Global Illumination: Interactive Indirect Illumination Prediction under Dynamic Area Lights (D. Gao, H. Mu, K. Xu). IEEE TVCG, 2022 or 2023 (sources disagree) | **V-sec**. https://cg.cs.tsinghua.edu.cn/people/~kun/2022neuralGI/NeuralGlobalIllumination.pdf | Screen-space end-to-end network from direct lighting to GI; static scenes, moving area lights. | Source-position generalisation exists in screen space for surfaces. Not an intrinsic operator. |
| A Generalizable Light Transport 3D Embedding for Global Illumination (B. Xu, M. Varma T, C. Wang, T.-M. Li, L. Wu, B. Wronski, R. Ramamoorthi, M. Salvi). arXiv:2510.18189 (v2 Jun 2026); comment says SIGGRAPH 2026 | **V** (arXiv abstract) | Scene point cloud → transformer of global point-to-point interactions → neural primitives; queries cross-attend to nearby primitives. Outputs diffuse irradiance; glossy radiance is preliminary. | **The closest "geometry-conditioned, cross-scene shared learned transport" precedent.** Surface GI, not a boundary-port operator of a medium, and not linear in the source. A geometry-conditioned shared transformer for transport is therefore covered as a class. |
| ReNeRF: Relightable Neural Radiance Fields with Nearfield Lighting. ICCV 2023 | **V-sec** (CVF open access). https://openaccess.thecvf.com/content/ICCV2023/html/Xu_ReNeRF_Relightable_Neural_Radiance_Fields_with_Nearfield_Lighting_ICCV_2023_paper.html | Uses explicit point-to-light distance and direction to interpolate and extrapolate near-field OLAT light positions. | Precedent for explicit source-relative geometry as the input that enables near-field source generalisation. Capture and relighting, not a medium. |

### 3.3 Geometry-conditioned shared learned scattering operators (most relevant class)

**A Learned Shape-Adaptive Subsurface Scattering Model.**
- **Citation:** D. Vicini, V. Koltun, W. Jakob. ACM TOG 38(4), SIGGRAPH 2019, art. 127, DOI 10.1145/3306346.3322974. **V** (EPFL RGL).
- **Links:** https://rgl.epfl.ch/publications/Vicini2019Learned, code https://github.com/rgl-epfl/learned-subsurface-scattering
- **Mechanism:**
  - A conditional VAE, trained on brute-force volumetric path-tracer data, samples outgoing surface locations directly.
  - It is conditioned on material and on "a set of features characterizing geometric variation in a neighborhood of the incident location": a low-order polynomial implicit surface capturing "curvature, thickness, corners, cylindrical and toroidal regions".
  - The code has a separate `AbsorptionModel` config, consistent with a scalar absorption/albedo model next to the position sampler.
  - Not read (U): polynomial degree, fitting kernel radius, frame canonicalisation and the outgoing-direction treatment.
- **Relevance — the strongest prior art for the class "one shared learned operator conditioned on local geometry around the source, generalising to unseen shapes and source positions":**
  - Its explicit information is source-centred local geometry (a polynomial fit), in a source-relative frame.
  - Its learned part is a material- and geometry-conditional exit distribution plus an absorption scalar. This is again the passive mass × normalised-distribution pattern.
  - A ReST-GI candidate that conditions a shared kernel on a source-centred geometry descriptor is therefore a combination with Vicini 2019, unless it adds something structurally different. Candidates for such a difference:
    - full exit-direction resolution with port/bin correspondence;
    - non-local or far-boundary geometry beyond the polynomial support;
    - linear multi-source application with exact passivity.
  - Vicini's descriptor is local by construction. It is not designed to represent geometry far from the incident point, which is the regime where ReST-GI's far-boundary error lives (WL74A: learned far-boundary error is about 70–80 % spatial).

**Deep Scattering: Rendering Atmospheric Clouds with Radiance-Predicting Neural Networks.**
- **Citation:** S. Kallweit, T. Müller, B. McWilliams, M. Gross, J. Novák. ACM TOG 36(6), SIGGRAPH Asia 2017, art. 231, DOI 10.1145/3130800.3130880. arXiv:1709.05418. **V** (arXiv / Disney / author page).
- **Link:** https://arxiv.org/abs/1709.05418
- **Mechanism:** A hierarchical 3D descriptor of the medium "with respect to the shading location and the light source" is fed progressively into a network that predicts radiance. It is pre-learned from "tens of cloud exemplars" and used to render new clouds. Stencil orientation and the single/multiple-scattering split are **U** (full text not readable).
- **Relevance:** Prior art for a shared, geometry-conditioned learned multiple-scattering predictor, with a hierarchical (coarse far, fine near) descriptor expressed relative to both receiver and light. Hierarchical relative-geometry descriptors for learned transport are covered.

**Learning Multiple-Scattering Solutions for Sphere-Tracing of Volumetric Subsurface Effects.**
- **Citation:** L. Leonard, K. Höhlein, R. Westermann. CGF 40(2), Eurographics 2021, DOI 10.1111/cgf.142623. arXiv:2011.03082. **V-sec**.
- **Link:** https://arxiv.org/abs/2011.03082
- **Mechanism:** A sequence of small CVAEs predicts the number of scattering events, the exit position and direction, and in-scattered direct light within a sphere, used as a sphere-tracing step.
- **Relevance:** A shared learned local transport block composed by an explicit outer walk. It is precedent for "learn a canonical local Green's function, handle the geometry explicitly by composition". It is a stochastic sampler, not a deterministic linear operator.

**Neural Local Inter-reflection Modeling for Garment Fold Rendering.**
- **Citation:** J. Son, N. Ryu, G. Kim, J. H. Lee, S. Lee. CGF 45(2), Eurographics 2026, DOI 10.1111/cgf.70348. **V-sec** (EG digital library, project page).
- **Link:** https://spock-the-wizard.github.io/garment-fold-rendering
- **Mechanism:** Geometry-conditioned local interreflection with an intensity term × a directional SH term. The paper reports generalisation to unseen geometry without retraining.
- **Relevance:** Precedent for geometry-conditioned local transport with an intensity × direction factorisation. SH directional, so low-order angular content.

**Neural Layered BRDFs.**
- **Citation:** J. Fan, B. Wang, M. Hašan, J. Yang, L.-Q. Yan. SIGGRAPH 2022. arXiv:2111.03797 (preprint title "Neural BRDFs: Representation and Operations"). **V-sec**.
- **Link:** https://wangningbei.github.io/2022/NLBRDF.html
- **Mechanism:** BRDFs are encoded as latent codes, and a learned layering network replaces the random-walk adding operation.
- **Relevance:** Precedent for a learned operator that composes transport operators in a latent space. Not spatial.

### 3.4 Analytic and semi-analytic subsurface transport: what is explicit in classical models

| Work | Verification | Mechanism | Relevance |
|---|---|---|---|
| A Practical Model for Subsurface Light Transport (Jensen, Marschner, Levoy, Hanrahan). SIGGRAPH 2001 | **V-sec**. https://graphics.stanford.edu/papers/bssrdf/ | Exact single scattering + dipole point-source diffusion for multiple scattering, on a half-space | Classical precedent for splitting a closed-form low-order term from a smooth multiple-scattering kernel, and for the method of images as the boundary treatment |
| A Rapid Hierarchical Rendering Technique for Translucent Materials (Jensen, Buhler). SIGGRAPH 2002, DOI 10.1145/566654.566619 | **V-sec**. https://graphics.ucsd.edu/~henrik/papers/fast_bssrdf/ | Irradiance samples in an octree, aggregated hierarchically with an error-controlled far field | **Prior art for DASQ (WL49):** distance-adaptive hierarchical quadrature of a diffusion kernel |
| A Quantized-Diffusion Model (d'Eon, Irving). SIGGRAPH 2011, TOG 30(4) | **V-sec**. https://naml.us/paper/subsurface/ | Quantised diffusion Green's function; multilayer searchlight problem; better near the source | Near-source accuracy of analytic kernels is a known weakness; the paper also improves hierarchical point-cloud methods |
| Photon Beam Diffusion (Habel, Christensen, Jarosz). CGF 32(4), EGSR 2013 | **V**. https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html | Incident light is a beam of sources along the refracted ray, integrated numerically (3–5 samples); directional incidence; exact single-scatter term; can correct diffusion inaccuracies | **Direct precedent for the FIA structure (WL76):** integrate a canonical kernel over the first-collision density along the incident ray. Erratum to Eq. 13 noted. |
| Directional Dipole Model (Frisvad, Hachisuka, Kjeldsen). ACM TOG 34(1), 2014 | **V-sec**. https://people.compute.dtu.dk/jerf/code/dirsss/ | Ray sources (positive and negative, mirrored) replace point sources; incidence direction enters analytically | Analytic precedent for incidence-dependent lateral structure, related to WL74A's incidence-dependent footprint shift |
| Approximate Reflectance Profiles (Christensen, Burley). Pixar TM 15-04, 2015 | **V** (Pixar library). https://graphics.pixar.com/library/ApproxBSSRDF | Radial profile as a sum of two exponentials; 3 parameterisations | The Burley profile behind the WL44–51 Yellow history; radial (azimuth-free) by construction |
| A Hybrid Monte Carlo Method for Accurate and Efficient Subsurface Scattering (H. Li, F. Pellacini, K. Torrance). EGSR 2005 | **V-sec**. https://diglib.eg.org/handle/10.2312/EGWR.EGSR05.283-290 | Monte Carlo for the first scattering events, dipole after deep penetration | Precedent for "explicit few-collision + smooth remainder" splits |
| A Comprehensive Framework for Rendering Layered Materials (W. Jakob, E. d'Eon, O. Jakob, S. Marschner). SIGGRAPH 2014 | **V** (Cornell / EPFL). https://www.cs.cornell.edu/projects/layered-sg14/ | Slab scattering in a directional basis: Fourier in azimuth × elevation discretisation; adding-doubling composition; "accurate for narrowly peaked functions" through sparse Fourier coefficients | **Directly relevant to WL77.** In plane-parallel slab transport the azimuthal Fourier expansion is the standard explicit angular representation. Its use of many orders for narrowly peaked functions is consistent with WL77's measured finding that relative-azimuth order ≤ 2 leaves 0.352–0.385 of the projection error. An inference: WL77 measured a 3D finite slab, not this paper's setting. |

### 3.5 Precomputed transport, operator compression and hierarchical kernels

| Work | Verification and link | Mechanism | Relevance |
|---|---|---|---|
| Precomputed Radiance Transfer (Sloan, Kautz, Snyder). SIGGRAPH 2002 | **V-sec**. https://jankautz.com/publications/prtSIG02.pdf | Per-point transfer vectors / matrices in low-order SH, applied linearly to lighting | Linear "operator × lighting" with a low-rank angular basis: the root class of ATO |
| Clustered Principal Components for PRT (Sloan, Hall, Hart, Snyder). SIGGRAPH 2003 | **V-sec** (title only, SIGGRAPH archive). https://history.siggraph.org/?p=115879 | Transfer matrices compressed by clustered PCA | Shared bases with per-point coefficients; ATO's global A/O plus spatial T is a CP variant of this class |
| Local, Deformable PRT (Sloan, Luna, Snyder). SIGGRAPH 2005 | **V-sec**. https://history.siggraph.org/?p=113968 | Zonal harmonics: cheap rotation of transfer into the deformed local frame; includes SSS on deforming models | **Precedent for explicit, cheaply rotatable frames:** equivariance through a rotatable basis. Relevant to WL74A's frame finding. |
| Spherical Harmonic Gradients for Mid-Range Illumination (Annen, Kautz, Durand, Seidel). EGSR 2004 | **V** (EG digital library / author PDF). https://jankautz.com/publications/gradientSH_RS04.pdf | First-order Taylor expansion of the lighting coefficients in position for nearby lights | A classical answer to "source position changes" in PRT: an explicit spatial expansion of the incident field |
| Precomputed Shadow Fields for Dynamic Scenes (K. Zhou, Y. Hu, S. Lin, B. Guo, H.-Y. Shum). SIGGRAPH 2005 | **V-sec**. https://history.siggraph.org/?p=113965 | Per-object precomputed occlusion and source fields, composed at runtime for moving objects and local lights | Precedent for the intrinsic precomputation + runtime composition split (the C side) |
| All-Frequency Interactive Relighting of Translucent Objects with Single and Multiple Scattering (R. Wang, J. Tran, D. Luebke). TOG 24(3), 2005 | **V-sec** (dblp). https://history.siggraph.org/?p=113966 | Precomputed translucent transport (Jensen BSSRDF incl. single scatter), wavelet-compressed | Precedent for a compressed linear boundary-to-boundary translucent transport matrix |
| Direct-to-Indirect Transfer for Cinematic Relighting (Hašan, Pellacini, Bala). SIGGRAPH 2006 | **V**. https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting | Huge transfer matrix, sparse in wavelet space, applied to the direct-light vector | A source-agnostic linear operator on the first-bounce field. Structurally close to "first-collision field → collided response" (FIA). |
| Kernel Nyström Method for Light Transport (J. Wang, Y. Dong, X. Tong, Z. Lin, B. Guo). SIGGRAPH 2009 | **V-sec**. https://history.siggraph.org/?p=108787 | Low-rank reconstruction of the transport matrix from sparse rows and columns with a nonlinear kernel | **Precedent for reconstructing a transport matrix from a few source columns.** The ReST-GI source-transfer problem is a structured column-completion problem (16 supervised columns). |
| Modular Radiance Transfer (Loos, Antani, Mitchell, Nowrouzezahrai, Jarosz, Sloan). SIGGRAPH Asia 2011, TOG 30(6) | **V** (Jarosz page). https://cs.dartmouth.edu/wjarosz/publications/loos11modular.html | Compact transport precomputed on canonical simple shapes, warped and combined at runtime | Precedent for scene-independent intrinsic transport on canonical blocks, warped to geometry |
| A Rapid Hierarchical Radiosity Algorithm (Hanrahan, Salzman, Aupperle). SIGGRAPH 1991. Wavelet Radiosity (Gortler, Schröder, Cohen, Hanrahan). SIGGRAPH 1993 | **V-sec**. https://graphics.stanford.edu/papers/rad/ | Adaptive hierarchical form factors, O(n) blocks; wavelet generalisation | Prior art for CAK / CASP adaptive refinement and hierarchical coupling |
| H-matrices, Part I (Hackbusch). Computing 62(2), 1999 | **V-sec**. https://doi.org/10.1007/s006070050015 | Data-sparse hierarchical low-rank blocks for integral operators | The near-field-dense / far-field-low-rank block structure is classical. A "local W + low-rank UV tail" (W+UV) is a one-level instance. |
| Lightweight Neural Basis Functions for All-Frequency Shading (Z. Xu, Z. Zeng, L. Wu, L. Wang, L.-Q. Yan). SIGGRAPH Asia 2022 | **V** (NVIDIA). https://research.nvidia.com/labs/rtr/publication/xu2022lightweight | Learned spherical bases with learned rotation and product networks | Precedent for learned angular dictionaries with a learned rotation (an equivariance surrogate) |
| Differentiable Light Transport with Gaussian Surfels via Adapted Radiosity (K. Jiang, J.-M. Sun, Z. Li, D. Wang, T.-M. Li, R. Ramamoorthi). arXiv:2509.18497 (Sep 2025) | **V** (arXiv) | Radiosity in SH coefficient space over Gaussian-surfel elements | Coefficient-space transport precedent (already in the FNTO contract §8) |

### 3.6 Neural operators and Green's-function learning

| Work | Verification and link | Mechanism | Relevance |
|---|---|---|---|
| Neural Operator: Graph Kernel Network for PDEs (Z. Li et al.). arXiv:2003.03485 (2020) | **V** (arXiv) | Kernel integral operators computed by message passing | Class of "geometry-graph-conditioned learned kernel integration": T from a GNN belongs here |
| Multipole Graph Neural Operator (Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, A. Stuart, K. Bhattacharya, A. Anandkumar). NeurIPS 2020 | **V** (NeurIPS). https://arxiv.org/abs/2006.09535 | Multi-level graph mimicking the fast multipole method; equivalent to multi-resolution kernel-matrix factorisation | **Prior art for "local edges + hierarchical far-field" learned kernels** (W+UV, DASQ in learned form) |
| Fourier Neural Operator (Li et al.). ICLR 2021 | **V** (ICLR). https://arxiv.org/abs/2010.08895 | Kernel parameterised in Fourier space on regular grids | Translation-equivariant on periodic regular domains; a bounded 3D slab with boundary sources is not its native setting. DeepRTE's authors say DeepONet and FNO "face fundamental limitations when directly applied to the RTE" (**S**, secondary). |
| DeepONet (L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis). Nat. Mach. Intell. 3:218–229 (2021) | **V-sec**. https://arxiv.org/abs/1910.03193 | Branch net (input function) × trunk net (query location): a learned low-rank expansion | Structurally the same separable "Σ_r branch_r · trunk_r" form as ATO's rank-r factorisation; the low-rank operator class is covered |
| Neural Operator: Learning Maps Between Function Spaces (Kovachki et al.). JMLR 24(89), 2023 | **V** (JMLR). https://www.jmlr.org/papers/v24/21-1524.html | Umbrella framework with graph, multipole, low-rank and Fourier parameterisations, and universal approximation | Every ATO / W+UV / GNN variant falls under this taxonomy |
| Geometry-Informed Neural Operator (GINO; Z. Li et al.). NeurIPS 2023 | **V** (NeurIPS). https://arxiv.org/abs/2309.00583 | SDF + point-cloud geometry encoder (GNO) → latent FNO → decoder; discretisation-convergent | Geometry-conditioned operators across shapes are covered as a class |
| Transolver (H. Wu, H. Luo, H. Wang, J. Wang, M. Long). ICML 2024, PMLR 235 | **V** (PMLR). https://arxiv.org/abs/2402.02366 | Physics-attention over learned slices; linear complexity on general geometries | Same class (geometry-general learned operators) |
| Group Equivariant FNO (G-FNO; J. Helwig et al.). ICML 2023, PMLR 202 | **V** (PMLR). https://arxiv.org/abs/2306.05697 | Rotation- and reflection-equivariant Fourier layers; reported to outperform data-augmentation baselines | Equivariance by construction beats augmentation in that study. Relevant to WL72's failed exact-symmetry loss (a loss, not a construction). Transfer to this problem is an inference. |
| E(n) Equivariant GNNs (V. G. Satorras, E. Hoogeboom, M. Welling). ICML 2021, PMLR 139 | **V** (PMLR). https://arxiv.org/abs/2102.09844 | Message passing equivariant to rotations, translations, reflections and permutations | Off-the-shelf equivariant geometry encoder |
| Learning Elliptic PDEs with Randomized Linear Algebra (N. Boullé, A. Townsend). Found. Comput. Math., 2022, DOI 10.1007/s10208-022-09556-w | **V-sec** (NSF PAR / arXiv). https://arxiv.org/abs/2102.00491 | Provable recovery of a Green's function from input–output pairs, using its hierarchical low-rank structure | **Theory relevant to sample efficiency:** recovery uses randomly drawn forcings and exploits hierarchical low-rank structure, with a data requirement (O(ε⁻⁶ log⁴(1/ε)) pairs) far above 16 one-hot sources. Elliptic, not transport. The extrapolation is an inference. |
| GreenNet (arXiv:2204.13247); Deep Generalized Green's Functions (arXiv:2306.02925); Variational Green's Functions (arXiv:2602.12349); Neural Green's Functions (NeurIPS 2025, arXiv:2511.01924); deep surrogate for reaction-diffusion Green's functions with the point-source location as parameter (arXiv:2310.03642) | **S** (search snippets only; titles and IDs seen, full texts not read) | Explicit analytic singular / free-space part + learned smooth (boundary-correction) part; or a point-source location as the operator parameter | **Precedent for "explicit free-space singular kernel + learned boundary correction"**, all for elliptic PDEs. No transport-equation (RTE) instance found. |

### 3.7 Learned radiative transfer (outside graphics)

| Work | Verification and link | Mechanism | Relevance |
|---|---|---|---|
| Physics Informed Neural Networks for Simulating Radiative Transfer (S. Mishra, R. Molinaro). arXiv:2009.13291; JQSRT venue not confirmed | **V** (arXiv) | PINN solving the RTE per problem, plus inverse problems | Per-instance solver, not a reusable operator |
| DeepRTE: Pre-trained Attention-based Neural Network for Radiative Transfer (Y. Zhu, M. Tang, Z. Ma). arXiv:2505.23190 (v1 May 2025, v4 Oct 2025) | **V** (arXiv abstract); architecture details **S** | Mesh-free neural operator that maps boundary conditions, cross-sections and scattering kernel to the solution. Green's-function theory in the architecture; **pre-training with delta-function inflow boundary conditions**; zero-shot claims. | **Closest non-graphics precedent for "learn the RTE Green's function from one-hot boundary inflows and superpose linearly".** It conditions on the medium (cross-sections) rather than on 3D boundary geometry, as far as the abstract shows; dimension and geometry scope are unverified. A candidate built on "delta-inflow pretraining + linear superposition" is not new as a class. |
| Element Learning, with applications to radiative transfer (S. Du, S. N. Stechmann). arXiv:2308.02467 (v2 Oct 2024) | **V** (arXiv) | One learned map per element (geometry + parameters → in2out boundary operator and in2sol Green's function); HDG-like assembly; "any domain geometry … without retraining" | **The closest structural precedent to an intrinsic boundary-to-boundary K conditioned on geometry:** a learned in2out operator per element, composed by an explicit global solver. Elements are small and canonical (it is a finite-element method), not a whole medium domain with far-boundary sources. |
| Filtered MgNet for RTE (arXiv:2604.23265) | **S** | DOM angles + learnable multigrid sub-operators; reported to degrade in transport regimes | Indicates that learned multigrid RTE solvers struggle in the transport (non-diffusive) regime, which applies here (mfp 1 mm vs a 10 mm slab, near-boundary sources) |
| Lattice-Boltzmann Lighting (R. Geist, K. Rasche, J. Westall, R. Schalkoff). EGSR 2004 | **V** (EG digital library) | Lattice photon transport tending to diffusion; clouds | A discrete-ordinate / lattice transport precedent; not learned |
| First-collision source method (classical neutron transport, e.g. Gomes and Stevens, NSE 1993; RAY3D / DONTRAN3D) | **V-sec** (OSTI / ANS records) | Uncollided flux by ray tracing → first-collision source → collided S_N solve; mitigates ray effects | **Mature precedent for the FIA factorisation** (exact uncollided + first-collision density + collided response). The search found no learned-collided-response variant; that absence may reflect search depth. |

### 3.8 Equivariance in illumination and transport learning

| Work | Verification and link | Relevance |
|---|---|---|
| RENI: Rotation-Equivariant Conditional Spherical Neural Fields (Gardner et al.). NeurIPS 2022 | **V-sec** (NeurIPS). https://arxiv.org/abs/2206.03858 | SO(2) (vertical-axis) equivariant illumination prior, compared against SO(3) and non-equivariant variants. An illumination prior, not a transport operator. |
| Local, deformable PRT (ZH rotation, §3.5); G-FNO (§3.6); EGNN (§3.6); Zeltner et al. learned shading frames (§3.1) | as above | Equivariance tools exist. The search found **no work applying a steerable or equivariant angular basis to a learned boundary-to-boundary transport operator of a bounded medium.** This is a weak negative (search depth, §8). |

---

## 4. Mapping of ReST-GI's historical mechanisms to prior art

| ReST-GI mechanism (worklog) | Closest prior art | Assessment |
|---|---|---|
| Passive mass × normalised exit distribution (W+UV, T mass-normalised) | 8DNA (albedo α × flow p); Vicini 2019 (absorption model + CVAE exit sampler) | Covered |
| Global rank-r angular dictionary A/O × spatial T (ATO, WL59/68/74/75) | PRT / CPCA (2002–2003); DeepONet branch × trunk; low-rank neural operators; Kernel Nyström | Covered as a class |
| Local W (128 neighbours) + low-rank UV tail | H-matrices (1999); Multipole GNO (2020); hierarchical radiosity | Covered |
| DASQ distance-adaptive quadrature (WL49) | Jensen and Buhler 2002 hierarchical octree; FMM / H-matrix; hierarchical radiosity | Covered |
| CAK / CASP contact-adaptive co-refinement (WL54/55) | Hierarchical / wavelet radiosity adaptive subdivision | Covered as a class |
| Exact symmetry *loss* (AR, WL72/73) | G-FNO shows construction > augmentation (general ML) | A loss-based route is weaker than construction in that study; consistent with WL72's null result |
| Relational shared kernel (RSK, WL72) | Graph kernel networks; Deep Scattering relative descriptors; 2510.18189 transformer embedding | Covered as a class |
| FIA first-collision factorisation (WL76) | First-collision source method (neutron transport); Jensen 2001 single + multiple split; Photon Beam Diffusion (beam of sources along the refracted ray); Direct-to-Indirect transfer | **Covered.** The exact factorisation is classical physics. The specific learned source-independent emitter response R(anchor → receiver) with ring lobes inside FIA has no exact precedent found, and WL76 measured it insufficient (training contrast 0.69–0.75). |
| Ring / low-order relative-azimuth lobes (WL76/77) | Burley / dipole radial profiles (azimuth-free); directional dipole (analytic incidence dependence); Fourier-azimuthal layered framework (Jakob et al. 2014) | Covered. Prior art in slab transport routinely uses higher azimuthal orders, consistent with WL77's measured order ≤ 2 shortfall (0.352–0.385 vs gate 0.226). |

---

## 5. Which mechanism classes are already well covered

A candidate built only from the elements below is a combination, not a new mechanism. Novelty would then have to come from the measured result (source-transfer validity under the ReST-GI contract), not from the parts.

1. **Per-asset baked boundary transport, including near-field 8D:** RNA, 8DNA, Neural SSS (non-half-space mode), NeuPreSS, NLT, Neural PRT, TransGI.
2. **Passivity by mass × normalised distribution:** 8DNA, Vicini 2019.
3. **Low-rank / separable linear transport operators:**
   - PRT family (2002–2005), Kernel Nyström, Direct-to-Indirect;
   - DeepONet and low-rank neural operators;
   - global A/O × T (ATO) belongs here.
4. **Hierarchical near-dense / far-low-rank kernels:**
   - hierarchical and wavelet radiosity, H-matrices, FMM;
   - Jensen and Buhler 2002 for SSS;
   - Multipole GNO for learned kernels.
5. **Geometry-conditioned shared learned operators across shapes:**
   - Vicini 2019 (local polynomial at the source);
   - Deep Scattering 2017 (hierarchical relative descriptor);
   - garment-fold interreflection 2026;
   - generalizable LT embedding (arXiv 2510.18189);
   - GINO, Transolver, graph kernel networks.
6. **Explicit uncollided / first-collision + learned or smooth remainder:**
   - first-collision source method;
   - Jensen 2001, PBD 2013, Li–Pellacini–Torrance 2005;
   - NLT (physical base + learned residual).
7. **Learn the Green's function from delta / point sources, then superpose linearly:**
   - DeepRTE (delta-inflow pretraining for the RTE);
   - the point-source-parameterised Green's-function surrogate (arXiv:2310.03642);
   - Boullé–Townsend (theory, elliptic).
8. **Analytic singular / free-space part + learned boundary correction:** GreenNet, Deep Generalized Green's Functions, VGF (elliptic). For transport, the method of images in the dipole family plays the same role analytically.
9. **Learned boundary-to-boundary element operators composed by an explicit global solver:** Element Learning (in2out / in2sol, RTE); Leonard et al. 2021 (learned sphere-step).
10. **Equivariance tooling:**
    - G-FNO, EGNN, RENI;
    - ZH rotation in LDPRT;
    - learned shading frames (Zeltner 2024);
    - azimuthal Fourier bases in layered transport.
11. **Source-relative explicit geometry for near-field light generalisation:** ReNeRF; SH gradients (Annen 2004).

---

## 6. Where open space plausibly exists

These are hypotheses about uncovered territory, not claims of novelty. Each depends on search depth (§8).

**O1. Held-out source-port transfer as the evaluated property.**
- No work found evaluates a geometry-conditioned learned boundary-to-boundary operator of a bounded scattering medium on held-out source positions with port × bin correspondence on both sides, trained from a small fixed set of one-hot boundary sources (16 here).
  - Vicini, Neural SSS and Deep Scattering train densely over source configurations.
  - 8DNA trains densely per asset.
  - DeepRTE uses delta inflows but, per its abstract, conditions on medium coefficients. Its scope with respect to 3D boundary geometry, near-boundary/corner sources and sparse source counts is unverified.
- The question "which information must be explicit so that sparse-source supervision transfers to far-boundary sources" appears unaddressed as a stated research question.

**O2. Explicit boundary-relative structure in a transport (RTE) boundary operator.**
- For elliptic PDEs, "explicit singular part + learned boundary correction" exists (GreenNet / VGF).
- For subsurface rendering, the explicit boundary treatment is the method of images on a half-space (dipole family). The learned geometric correction is local (Vicini's polynomial) or planar (Neural SSS).
- No work found makes the explicit part a source-relative, boundary-aware transport kernel and learns only a geometry-conditioned correction with full angular output for sources near edges and corners of a finite 3D medium. Examples of such a kernel: an image-source construction for the faces of a box, or distance-to-boundary and the incidence-dependent footprint shift WL74A identified.
- Risk: a candidate here would be a combination of classes 6–8 in §5 unless its explicit part is structurally new for the RTE geometry.

**O3. Equivariant high-order angular representation inside a learned spatial operator.**
- Layered-material transport uses many azimuthal Fourier orders in plane-parallel slabs.
- Equivariant neural operators exist (G-FNO, EGNN).
- No work found combines receiver-local, source-relative azimuthal harmonics of order above 2 (WL77 measured that orders ≤ 2 are insufficient) with a learned, geometry-conditioned spatial transfer in a bounded 3D medium under exact local D4 bin equivariance (WL74A).
- Whether this is a research gap or simply unpublished engineering is uncertain.

**O4. Validated F/S state reuse of such an operator under deformation.**
- Out of scope for WL77A Gate 4, but noted: the per-asset baselines (8DNA, RNA) cannot reuse under internal geometry change. 8DNA's convex-hull constraint is explicit in its paper.
- Geometry-conditioned shared operators (Vicini, GINO-type) can in principle reuse, but none found reports a deformation-validity protocol against an independent reference.

**Not open space:**
- the K/C split itself (shadow fields, modular radiance transfer, 8DNA asset + path tracer, TransGI object model + probes);
- operator compression, hierarchy and passivity parameterisations, taken alone.

---

## 7. Implications for explicit versus learned

This section is an interpretation of prior art against the ReST-GI evidence. It is not a selection or a recommendation.

1. **What prior art keeps explicit when sources move.** Successful source-transferable models keep explicit:
   - the source-relative geometry: Vicini (local polynomial around the incident point), Deep Scattering (descriptor relative to shading point and light), ReNeRF (point-to-light distance and direction), SH gradients (explicit spatial expansion);
   - the uncollided / first-collision structure: classical FCS, PBD, Jensen 2001;
   - the boundary treatment: method of images, half-space assumption.

   What they learn is a canonical, low-dimensional conditional response in a source-centred frame. ReST-GI's ATO instead uses source-independent global atoms A/O. Its spatial transfer T is angle-independent, and per-face frames are not equivariant (WL74A). Relative to prior art, the incidence-dependent sub-cell footprint shift and the source-relative frame are therefore implicit rather than explicit.
2. **Data efficiency.** Green's-function learning theory (Boullé–Townsend) and classical Kernel Nyström both reconstruct transport from samples, using hierarchical low-rank structure and many probing samples. With 16 one-hot sources, hierarchical low-rank structure alone is unlikely to determine the near-source and near-boundary blocks, which are dense and source-specific. Prior art handles those blocks explicitly: analytically, by quadrature, or by local geometry conditioning. This is an inference from theory for elliptic operators. It is consistent with, but not proven by, WL75: interior transfer improved, far-boundary transfer did not (0.79 → 0.80–0.82).
3. **Exact factorisation ≠ learnable approximation.** FIA mirrors the first-collision source method and PBD. In those methods the collided remainder is solved by S_N or diffusion, or integrated with an analytic kernel; it is never a learned source-independent emitter with restricted angular lobes. WL76 and WL77 showed that the learned remainder's angular restriction is binding. Prior art indicates the remainder's angular structure needs a rich basis (Fourier-azimuthal, many orders) or a solver, not a ring / order-2 lobe.
4. **Equivariance by construction.** G-FNO reports that construction outperforms augmentation; the general ML evidence favours equivariance built into the architecture over losses. WL72's exact-symmetry loss gave no gain on independent sources, which is consistent with this (not proof).

---

## 8. Rejected claims and ideas (with reasons)

| Claim or idea | Reason rejected |
|---|---|
| "FIA's first-collision factorisation is a novel mechanism" | Classical first-collision source method; Jensen 2001 single + multiple split; Photon Beam Diffusion integrates a kernel along the incident beam. Only the learned emitter response inside it lacks an exact precedent, and WL76 measured it insufficient. |
| "Rank-r A/O × T (ATO) or W + UV is a new operator form" | Covered by PRT / CPCA, DeepONet, low-rank and multipole neural operators, and H-matrices. |
| "DASQ / CAK hierarchical adaptivity is novel" | Jensen–Buhler 2002, hierarchical / wavelet radiosity, FMM / H-matrices, Multipole GNO. The project's own novelty discipline already excludes this (RESEARCH_CENTRIC_TOPIC l. 41). |
| "Passivity via mass × normalised distribution is novel" | 8DNA α · p and Vicini absorption + exit sampler. |
| "A geometry-conditioned shared transformer or GNN transport model is new" | arXiv 2510.18189 (generalizable LT embedding, SIGGRAPH 2026 per its comment), GINO, Transolver, graph kernel networks, Deep Scattering, Vicini 2019. |
| "Learning a Green's function from delta sources and superposing is new for RTE" | DeepRTE pretrains with delta-function inflow boundary conditions (abstract). |
| "8DNA generalises across geometry or could serve as a source-transfer mechanism" | Its paper says it is trained "for each asset in isolation" and is valid only when nothing intersects the convex hull. Per-asset and dense-source, so not a source-transfer test. Its role is a baseline (BASELINE_ROLES l. 9, 36). |
| "Neural SSS's half-space sharing solves boundary sources" | Its geometry reuse relies on the planar half-space assumption, which is exactly what breaks near edges, corners and finite thickness, the failing regime here. |
| "Use FNO directly on the port grid" | FNO's native setting is regular, periodic grids. The port boundary of a box with face seams and corners is neither, and DeepRTE's authors report FNO and DeepONet limitations for the RTE (secondary). It can be applied via a latent grid (GINO), but that is then the GINO class. |
| "TransGI as a mechanism or a measured comparator" | No released code; rigid, surface-only, transparent objects excluded (WL67 l. 33). Not applicable to a medium K. |
| "Order-2 relative azimuth suffices if learned well" | WL77 projection floor 0.385 / 0.352 vs gate 0.226 with noise 0.002. Layered-transport prior art uses more Fourier orders for peaked functions. |
| "An exact-symmetry loss gives equivariance" | WL72 found no independent-source gain. General ML evidence (G-FNO) favours construction over augmentation and losses. |

---

## 9. Requirements a candidate should meet to avoid being a pure recombination

These are derived from prior art. They are not new gates.

1. The candidate must state which element is not covered by §5 classes 1–11. Otherwise its claim is the empirical result only.
2. If it uses source-relative geometry conditioning, it must cite Vicini 2019 and Deep Scattering, and say what differs. Candidate differences:
   - non-local or far-boundary geometry;
   - full outgoing-bin output with port correspondence;
   - exact linear multi-source application;
   - exact passivity.
3. If it uses an explicit singular or near-field term plus a learned correction, it must cite GreenNet / VGF (elliptic) and the dipole method of images, and specify the explicit RTE boundary construction (faces, edges, corners of the slab).
4. If it uses delta-source pretraining or superposition, it must cite DeepRTE and Element Learning.
5. If it uses an equivariant angular basis, it must cite G-FNO / EGNN and Jakob et al. 2014 (Fourier-azimuthal), and state the order and how it respects the exact local D4 of the bin grid (WL74A).
6. Any baseline comparison must keep the BASELINE_ROLES terminology: Neural Full Recompute vs Native Reuse vs path-traced re-render.

---

## 10. Remaining uncertainty and search limitations

**Tools and coverage**
- Only WebSearch (US-indexed) and WebFetch were used. About 45 queries were run, with roughly 12 primary pages fetched.
- Several publisher PDFs were inaccessible: EG digital library HTTP 403, NSF PAR HTTP 500, and the Vicini 2019 PDF exceeded the 10 MB fetch limit. Deep Scattering's ar5iv page carried no text.
- As a result, these mechanism details are abstract- or snippet-level:
  - Neural SSS: the full half-space formulation;
  - Vicini 2019: polynomial degree, support, frame and outgoing-direction treatment are **U**;
  - Deep Scattering: stencil orientation and the scattering-order split are **U**;
  - DeepRTE: architecture, dimension and geometry scope are **S**.

**Venue and author discrepancies**
- Neural GI under dynamic area lights: TVCG 2022 vs 2023.
- Vertex Features: author order.
- RNA: TOG 2024 vs "44(1) 2025".
- 8DNA: no venue on its arXiv page; the repository says SIGGRAPH 2026.
- arXiv 2312.15711 is probably the NeuPreSS preprint (inferred from the shared DOI).

**Possible blind spots**
- Neutron-transport and atmospheric-RT ML literature behind paywalls: learned response matrices or learned collided-flux operators may exist there (Element Learning is one hit).
- Recent (2025–2026) SIGGRAPH / EG / EGSR papers not yet well indexed.
- Patent literature.
- Non-English literature.
- The "no work found" statements in §6 (O1–O3) and §3.8 are weak negatives limited by this search, not proofs of absence.

**Inferences that are not measured**
- The applicability of Boullé–Townsend sample-complexity reasoning (elliptic) to RTE boundary operators.
- The link between Jakob et al. 2014's high-order azimuthal needs and WL77's finding.

**Repository evidence only spot-checked:** WL75 / WL77 / WL74A numbers were confirmed at the cited lines. Nothing was re-computed.

---

## 11. Source list (external)

**Baselines and per-asset neural transport**
- https://arxiv.org/abs/2312.09398 (RNA)
- https://arxiv.org/abs/2604.25129 and https://arxiv.org/html/2604.25129 (8DNA)
- https://arxiv.org/abs/2506.09909 (TransGI)
- https://arxiv.org/abs/2008.03806 (NLT)
- https://diglib.eg.org/handle/10.1111/cgf14480 (Neural PRT)
- https://diglib.eg.org/handle/10.1111/cgf15158 (Neural SSS)
- https://diglib.eg.org/handle/10.1111/cgf15234 (NeuPreSS) and https://arxiv.org/abs/2312.15711 (Neural BSSRDF)
- https://arxiv.org/abs/2305.02678 (Real-Time Neural Appearance Models)

**Scene-level and online neural GI**
- https://arxiv.org/abs/2106.12372 (NRC)
- https://arxiv.org/abs/2105.12319 (Neural Radiosity)
- https://woaixuexisr.github.io/papers/dynamic-neural-radiosity (Dynamic Neural Radiosity)
- https://arxiv.org/abs/2508.07852 (Vertex Features)
- https://cg.cs.tsinghua.edu.cn/people/~kun/2022neuralGI/NeuralGlobalIllumination.pdf (Neural GI, dynamic area lights)
- https://arxiv.org/abs/2510.18189 (Generalizable LT embedding)
- https://openaccess.thecvf.com/content/ICCV2023/html/Xu_ReNeRF_Relightable_Neural_Radiance_Fields_with_Nearfield_Lighting_ICCV_2023_paper.html (ReNeRF)

**Geometry-conditioned shared learned scattering**
- https://rgl.epfl.ch/publications/Vicini2019Learned and https://github.com/rgl-epfl/learned-subsurface-scattering (Vicini 2019)
- https://arxiv.org/abs/1709.05418 (Deep Scattering)
- https://arxiv.org/abs/2011.03082 (Leonard et al.)
- https://spock-the-wizard.github.io/garment-fold-rendering (garment folds)
- https://wangningbei.github.io/2022/NLBRDF.html (Neural Layered BRDFs)

**Analytic and semi-analytic subsurface transport**
- https://graphics.stanford.edu/papers/bssrdf/ (dipole)
- https://graphics.ucsd.edu/~henrik/papers/fast_bssrdf/ (Jensen–Buhler hierarchical)
- https://naml.us/paper/subsurface/ (quantized diffusion)
- https://cs.dartmouth.edu/~wjarosz/publications/habel13pbd.html (PBD)
- https://people.compute.dtu.dk/jerf/code/dirsss/ (directional dipole)
- https://graphics.pixar.com/library/ApproxBSSRDF (Christensen–Burley)
- https://diglib.eg.org/handle/10.2312/EGWR.EGSR05.283-290 (Li–Pellacini–Torrance)
- https://www.cs.cornell.edu/projects/layered-sg14/ (layered materials)

**Precomputed transport, compression and hierarchical kernels**
- https://jankautz.com/publications/prtSIG02.pdf (PRT)
- https://history.siggraph.org/?p=115879 (CPCA)
- https://history.siggraph.org/?p=113968 (LDPRT)
- https://jankautz.com/publications/gradientSH_RS04.pdf (SH gradients)
- https://history.siggraph.org/?p=113965 (shadow fields)
- https://history.siggraph.org/?p=113966 (translucent PRT)
- https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting (Direct-to-Indirect)
- https://history.siggraph.org/?p=108787 (Kernel Nyström)
- https://cs.dartmouth.edu/wjarosz/publications/loos11modular.html (Modular Radiance Transfer)
- https://graphics.stanford.edu/papers/rad/ (hierarchical radiosity) and https://history.siggraph.org/?p=118612 (wavelet radiosity)
- https://doi.org/10.1007/s006070050015 (H-matrices)
- https://research.nvidia.com/labs/rtr/publication/xu2022lightweight (neural basis functions)
- https://arxiv.org/abs/2509.18497 (Gaussian-surfel radiosity)

**Neural operators, Green's functions and learned radiative transfer**
- https://arxiv.org/abs/2003.03485 (GKN)
- https://arxiv.org/abs/2006.09535 (MGNO)
- https://arxiv.org/abs/2010.08895 (FNO)
- https://arxiv.org/abs/1910.03193 (DeepONet)
- https://www.jmlr.org/papers/v24/21-1524.html (Neural Operator, JMLR)
- https://arxiv.org/abs/2309.00583 (GINO)
- https://arxiv.org/abs/2402.02366 (Transolver)
- https://arxiv.org/abs/2306.05697 (G-FNO)
- https://arxiv.org/abs/2102.09844 (EGNN)
- https://arxiv.org/abs/2102.00491 (Boullé–Townsend)
- https://arxiv.org/abs/2204.13247, https://arxiv.org/abs/2306.02925, https://arxiv.org/abs/2602.12349, https://arxiv.org/abs/2511.01924, https://arxiv.org/abs/2310.03642 (Green's-function learning; snippet-level)
- https://arxiv.org/abs/2009.13291 (PINN RTE)
- https://arxiv.org/abs/2505.23190 (DeepRTE)
- https://arxiv.org/abs/2308.02467 (Element Learning)
- https://arxiv.org/abs/2604.23265 (Filtered MgNet; snippet-level)
- https://diglib.eg.org/handle/10.2312/EGWR.EGSR04.355-362 (Lattice-Boltzmann Lighting)
- https://www.osti.gov/biblio/6326046 (first-collision source method)

**Equivariance**
- https://arxiv.org/abs/2206.03858 (RENI)

**Status: COMPLETED.**
