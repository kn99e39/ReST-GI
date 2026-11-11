

# ANGLE: Precomputed and learned light-transport OPERATORS (PRT and all-frequency/glossy variants, direct-to-indirect and wavelet/meshless transport matrices, modular reduced operators with interfaces, per-obj

### Sloan, Kautz, Snyder — Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments (2002, SIGGRAPH 2002) [read_abstract_or_secondary]
- mechanism: Per-vertex linear transfer (vectors for diffuse, matrices for glossy) maps distant low-frequency lighting in spherical harmonics to exit radiance, with self-shadowing and self-interreflection baked in. 'Neighborhood transfer' stores the same kind of operator on a volume around a rigid object so it can cast soft shadows, glossy reflections and caustics onto arbitrary receivers.
- precomputed / learned: Per-object SH transfer vectors/matrices (self-transfer) and a sampled transfer volume around the object (neighborhood transfer); nothing learned.
- valid under geometry edits: Self-transfer of a rigid object survives rigid motion of that object (lighting is rotated into its frame). Neighborhood transfer survives motion of the receiver relative to one object. Anything involving two or more mutually influencing objects, or deformation, is invalidated and needs a new precompute (recalled: receivers do not feed back onto the sender).
- sharp glossy interreflection: Excluded by construction: lighting and transfer are band-limited to low SH orders, so glossy response is low-frequency only.
- cost claims: Real-time rendering after an offline per-object simulation; no path-tracing comparison at matched quality in the sources I read.
- limitation: Low-frequency, distant lighting; operator tied to a fixed object; multi-object coupling not representable.
- relevance: Origin of the 'source-independent linear operator applied to an encoding of incident light' idea, and of attaching the operator to an object's own frame so rigid motion is free. Shows the price paid from the start: rigid-motion validity was obtained together with a band limit.
- url: https://jankautz.com/publications/prtSIG02.pdf

### Ng, Ramamoorthi, Hanrahan — Triple Product Wavelet Integrals for All-Frequency Relighting (2004, SIGGRAPH 2004 (ACM TOG 23(3))) [read_abstract_or_secondary]
- mechanism: Shading is the integral of a product of three functions (lighting, visibility, BRDF), each projected to Haar wavelets; the paper analyses triple-product integrals and gives linear/sublinear-time algorithms for Haar, so lighting and view can both change at all frequencies.
- precomputed / learned: Per-vertex visibility in a wavelet basis and BRDF slices in the same basis; the factors are kept separate rather than baked into one transfer matrix.
- valid under geometry edits: BRDF and lighting factors are geometry-independent; the per-vertex visibility factor is scene-configuration-specific and must be recomputed after any edit (recalled: static geometry only).
- sharp glossy interreflection: Sharp shadows and sharp BRDF lobes for DIRECT lighting only; interreflection is not part of the triple product.
- cost claims: All-frequency lighting up to 6x256x256 cubemaps, changing materials and viewpoint 'in a few seconds' per frame (as summarised from the abstract).
- limitation: Direct lighting, static geometry, per-vertex storage; sparsity is obtained by nonlinear coefficient selection that depends on the particular configuration.
- relevance: Shows that all-frequency directional relations were made tractable by sparse selection in a localized basis, not by low rank, and by factorising the operator into geometry-independent and geometry-dependent parts — the geometry-dependent part (visibility) is exactly what an edit invalidates.
- url: https://graphics.stanford.edu/papers/allfreqmat

### Liu, Sloan, Shum, Snyder — All-Frequency Precomputed Radiance Transfer for Glossy Objects (concurrent: Wang, Tran, Luebke — All-Frequency Relighting of Non-Diffuse Objects using Separable BRDF Approximation) (2004, Eurographics Symposium on Rendering 2004) [read_abstract_or_secondary]
- mechanism: Glossy BRDFs are factorised into view-dependent and light-dependent parts (about 10 terms); per-vertex transfer matrices from a 6x32x32 lighting cube map to these factors are compressed with Haar wavelets over 24 lighting segments plus clustered PCA.
- precomputed / learned: Per-vertex transfer matrices (lighting basis -> BRDF light-factor coefficients), wavelet + CPCA compressed.
- valid under geometry edits: Only a rigid transform of the whole object relative to distant lighting; any relative motion of parts invalidates the matrices.
- sharp glossy interreflection: All-frequency shadows on glossy receivers under distant light. Whether interreflection is included in the transfer is not confirmed from what I read; no sender-side sharp lobe is resolved in any case because the output side is a 10-term separable BRDF.
- cost claims: 1-5 Hz on models up to 50,000 vertices with 2-3 orders of magnitude more lighting coefficients than SH PRT.
- limitation: Static geometry, distant lighting, low-term separable BRDF limits specular sharpness on the view side.
- relevance: The glossy all-frequency transfer matrix is the classical analogue of a (patch x direction) transport table; its size forced both a separable angular factorisation and scene-specific compression.
- url: https://www.microsoft.com/en-us/research/publication/frequency-precomputed-radiance-transfer-glossy-objects/

### Zhou, Hu, Lin, Guo, Shum — Precomputed Shadow Fields for Dynamic Scenes (2005, SIGGRAPH 2005 (ACM TOG 24(3))) [read_primary_source]
- mechanism: Each rigid entity carries a field sampled on concentric spheres around it: a source radiance field (SRF) for lights and an object occlusion field (OOF) for occluders, each sample being a cube map in the entity's local frame. At run time the incident radiance at a receiver is assembled by rotating and multiplying the fields of the entities between receiver and light (SH products for low frequency, wavelet products for all-frequency).
- precomputed / learned: Per-entity SRF/OOF on shells (e.g. 6x16x16 directions for low frequency; 32 spheres of 6x32x32 for all-frequency), SH or wavelet compressed (about 268 non-zero wavelet coefficients per sample); per-vertex self-visibility. Nothing learned.
- valid under geometry edits: Everything: fields are independent of scene configuration, so arbitrary rigid motion of any object or local light needs no recomputation, and instances share one field. Deformation is not covered.
- sharp glossy interreflection: Excluded. The paper states it handles only direct lighting and lists interreflection as future work. All-frequency refers to shadows.
- cost claims: Low-frequency: 12-30 fps at 15-50 MB; all-frequency: 0.1 fps (Spaceships, 280 MB) to 7-16 fps (relighting, 500 MB); precompute under two hours for all-frequency fields.
- limitation: Direct lighting only; sampling density chosen by experiment with no theory; all-frequency fields are large and slow; fields cannot be evaluated inside another object's bounding region without care.
- relevance: The clearest demonstration that full rigid-edit validity is obtainable by making each object own a field in its own frame and composing at run time — and that doing so at all frequencies already cost two orders of magnitude in speed for visibility alone, before any interreflection.
- url: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/shadowfields.pdf

### Pan et al. — Precomputed Radiance Transfer Field for Rendering Interreflections in Dynamic Scenes (related: Iwasaki, Dobashi, Yoshimoto, Nishita — PRT for Dynamic Scenes Taking into Account Light Interreflection, EGSR 2007; Mei, Shi, Wu — Spherical Radiance Transport Maps, CGF 2004) (2007, Eurographics 2007 (CGF 26(3))) [read_abstract_or_secondary]
- mechanism: A radiance transfer field records, for an object, the radiance it sends to sample points in its surrounding space as a linear function of the radiance incident on the object. Because this is independent of the rest of the scene it plugs into the shadow-field framework and adds inter-object interreflection for rigidly moving objects.
- precomputed / learned: Per-object linear operator: incident lighting coefficients on the object -> radiance field around it, compressed with clustered PCA.
- valid under geometry edits: The per-object operator is valid under any rigid rearrangement; composition is done at run time. Deformation invalidates it.
- sharp glossy interreflection: Diffuse and glossy senders are supported, but only under low-frequency lighting (SH); sharp sender lobes are band-limited away.
- cost claims: Abstract claims negligible extra cost over shadow fields for adding interreflection; data highly compressible by CPCA. No numbers read.
- limitation: Low-frequency restriction; incident light on the sender is summarised by a few coefficients (recalled: effectively one incident-lighting estimate per object), so near-field and sharp directional relations between sender and receiver are lost.
- relevance: The nearest classical precedent to a source-independent, per-object, composable interreflection operator that survives rigid edits. Its validity under edits was bought with exactly the restriction (low angular frequency of the sender response) that the light -> sharp reflector -> diffuse family violates.
- url: https://diglib.eg.org/handle/10.2312/CGF.v26i3pp485-493

### Hašan, Pellacini, Bala — Direct-to-Indirect Transfer for Cinematic Relighting (2006, SIGGRAPH 2006 (ACM TOG 25(3))) [read_abstract_or_secondary]
- mechanism: For a fixed view, multi-bounce indirect light at view samples is a linear transform of direct light on a large cloud of gather samples. The huge matrix is precomputed hierarchically and stored sparsely in a wavelet basis imposed on the unstructured gather cloud.
- precomputed / learned: A sparse wavelet-space direct-to-indirect matrix (gather samples -> view samples), source-independent: any direct lighting, computed by any shader, is the input.
- valid under geometry edits: Nothing; fixed geometry and fixed camera. Any edit requires recomputing the matrix (precompute under three hours on one processor).
- sharp glossy interreflection: Multi-bounce diffuse indirect; recalled: glossy allowed at the final view sample but gather samples are treated as diffuse senders, so sharp sender lobes are not carried.
- cost claims: 10-20 fps on scenes with over 2 million polygons after an under-3-hour precompute.
- limitation: Fixed view and geometry; sender side is direction-collapsed.
- relevance: The canonical 'direct light encoding in, indirect out' source-independent operator. It is the explicit matrix that a geometry-conditioned generator would have to emit, and it shows that its sparsity structure (wavelet) is specific to one geometry.
- url: https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting

### Kontkanen, Turquin, Holzschuch, Sillion — Wavelet Radiance Transport for Interactive Indirect Lighting (2006, Eurographics Symposium on Rendering 2006) [read_abstract_or_secondary]
- mechanism: Radiance is represented in a 4D (space x direction) wavelet basis; a Global Transport Operator that propagates radiance through the scene (all bounces) is precomputed hierarchically; at run time direct lighting is projected onto the basis and multiplied by the operator.
- precomputed / learned: A 4D-wavelet global transport operator (direct radiance -> converged indirect radiance) for diffuse and glossy materials.
- valid under geometry edits: Nothing; the abstract states static scenes only. The operator is a property of one configuration.
- sharp glossy interreflection: Glossy materials are included in the operator through the directional dimensions of the wavelet basis; how sharp a lobe was actually demonstrated is not verified here.
- cost claims: Interactive manipulation of lighting; no numbers read.
- limitation: Static geometry; hierarchical 4D wavelets were demonstrated on simple, smooth surface parameterisations (as noted by the later meshless work).
- relevance: A direct classical counterpart of a (patch x outgoing-direction) <- (patch x incident-direction) transport table with glossy senders. It establishes that such an operator is representable in a hierarchical directional basis for one fixed scene, and says nothing about reuse across configurations.
- url: https://hal.archives-ouvertes.fr/inria-00379398

### Lehtinen, Zwicker, Turquin, Kontkanen, Durand, Sillion, Aila — A Meshless Hierarchical Representation for Light Transport (2008, SIGGRAPH 2008 (ACM TOG 27(3))) [read_abstract_or_secondary]
- mechanism: A hierarchical basis built from scattered-data approximation over points, decoupled from the surface representation, is used for a precomputed direct-to-indirect transport algorithm on complex geometry.
- precomputed / learned: Direct-to-indirect transport operator expressed in the meshless hierarchical basis.
- valid under geometry edits: The basis construction is geometry-agnostic, but the operator is precomputed per scene; an edit requires recomputation (recalled: static scenes).
- sharp glossy interreflection: Not verified; recalled as primarily diffuse indirect with smooth basis functions, which would not resolve sharp sender lobes.
- cost claims: Abstract claims a large increase in the scene complexity PRT can handle and shorter precomputation than per-vertex bases; no numbers read.
- limitation: Per-scene operator; smooth spatial basis.
- relevance: Shows the operator basis can be freed from meshing/parameterisation (relevant to patch choice), while the operator entries themselves remain tied to one configuration.
- url: https://people.csail.mit.edu/jaakko/meshless

### Loos, Antani, Mitchell, Nowrouzezahrai, Jarosz, Sloan — Modular Radiance Transfer (2011, SIGGRAPH Asia 2011 (ACM TOG 30(6))) [read_primary_source]
- mechanism: A small dictionary of simple shapes (cubes with missing faces, cylinders) each carries reduced direct-to-indirect operators. A low-rank 'light prior' (SVD of direct lighting from implicit lights, kd = 64, kself = 32) defines the input subspace; light leaving a shape through an open face is carried as a reduced interface lightfield (raw 12^2 spatial x 24^2 directional, then SVD) and propagated to neighbouring shapes by precomputed interface-to-interface operators. Shapes are instanced, warped and connected at run time.
- precomputed / learned: Per dictionary shape, once and scene-independently: light prior bases, shape self-transfer, surface-to-interface, interface-to-interface and interface-to-surface operators, plus a volumetric operator for lighting dynamic characters. All dense low-rank matrices.
- valid under geometry edits: All operators: scenes are assembled and re-assembled from the dictionary without scene-dependent precomputation. Actual scene geometry is only approximated by warped proxies; dynamic objects receive light from a volume but do not cast indirect shadows (the paper states indirect shadows are ignored).
- sharp glossy interreflection: Excluded. The paper targets diffuse, coarse-scale indirect light and states that extending the approach to high-frequency glossy transport while keeping its benefits is a challenging open problem.
- cost claims: Over 475 FPS on high-end GPUs and over 45 FPS on mobile for a 19-block scene; 2.1 ms versus 'many hours' for ground truth (not at matched quality).
- limitation: Stated in the paper: lighting conditions outside the low-rank prior cannot be captured (e.g. a light too close to a wall); results are 'plausible' rather than accurate; a single proxy degenerates to an 'intelligent ambient term'.
- relevance: The strongest precedent for reusable reduced operators with explicit interfaces that stay valid under re-arrangement. It documents the generic behaviour of reduced operators: what is outside the prior's span is replaced by a smooth, geometrically unspecific response.
- url: https://cs.dartmouth.edu/~wjarosz/publications/loos11modular.html

### Drettakis, Sillion — Interactive Update of Global Illumination Using a Line-Space Hierarchy (1997, SIGGRAPH 1997) [read_abstract_or_secondary]
- mechanism: Hierarchical-radiosity links between clusters/surfaces are augmented with shafts (the set of lines through both linked elements). When an object moves, traversing this implicit line-space hierarchy identifies exactly the links whose transport changed and re-solves only those, with the traversal depth bounded by the frame-time budget.
- precomputed / learned: The link structure (the discretised transport operator) and its shafts; nothing learned.
- valid under geometry edits: All links whose shaft is not intersected by the swept moved object stay valid untouched; intersected links are recomputed, and subdivision that is no longer needed is cleaned up.
- sharp glossy interreflection: Excluded: diffuse hierarchical radiosity with clustering.
- cost claims: Interactive updates (2-3 frames per second) after object motion for scenes of several thousand polygons on 1997 hardware.
- limitation: Diffuse only; update cost grows with the number of affected links, which grows with the directional resolution of the operator.
- relevance: Establishes that the subset of operator entries invalidated by a rigid move is geometrically identifiable without re-solving everything — the explicit-operator answer to 'what stays valid under an edit'. It was never carried to directional (glossy) links in the work I found.
- url: https://history.siggraph.org/?p=117251

### Xu, Cao, Ma, Dong, Wang, Hu — A Practical Algorithm for Rendering Interreflections with All-Frequency BRDFs (2014, ACM TOG 33(1)) [read_abstract_or_secondary]
- mechanism: With BRDFs represented as spherical Gaussians, one-bounce interreflection from a reflector triangle to a shading point is evaluated by an analytic formula plus a piecewise-linear approximation; a hierarchical integration over triangles with bounded error handles complex scenes on the GPU.
- precomputed / learned: Essentially nothing scene-specific (SG fits of the BRDFs; a triangle hierarchy); the transport is evaluated analytically per frame.
- valid under geometry edits: Everything, since no transport is cached (recalled: dynamic scenes, lights and BRDFs supported).
- sharp glossy interreflection: This is its target: the integration is sender-centred (per reflector triangle, with the sender's SG lobe in closed form) so highly glossy senders, where VPL methods fail, are handled; one bounce only.
- cost claims: Interactive GPU rates reported; no figure read.
- limitation: One-bounce interreflection; visibility for the indirect bounce is approximate (recalled); cost scales with reflector triangles.
- relevance: A non-learned, edit-proof evaluation of exactly the light -> glossy sender -> receiver family, obtained by integrating over the sender with its lobe known analytically rather than gathering from the receiver. It marks what a reusable operator has to compete against for that family.
- url: https://cg.cs.tsinghua.edu.cn/people/~kun/interreflection/index.html

### Ren, Wang, Gong, Lin, Tong, Guo — Global Illumination with Radiance Regression Functions (2013, SIGGRAPH 2013 (ACM TOG 32(4))) [read_abstract_or_secondary]
- mechanism: A small feed-forward network regresses indirect radiance at a surface point from its position, view direction and the (point) light's parameters, augmented with local attributes; the input space is partitioned and one small network is fitted per region.
- precomputed / learned: Per-scene networks trained on offline path-traced shading samples.
- valid under geometry edits: Nothing in the transport itself: geometry is baked into the weights; only light position and view are free inputs (the abstract covers dynamic local lights; moving geometry is not addressed).
- sharp glossy interreflection: Claims real-time 'multiple-bounce high-frequency glossy interreflections' and changing caustics for the trained scene; not linear in the source (one point light as input).
- cost claims: Real-time evaluation after offline precomputation; no numbers read.
- limitation: Scene-specific and source-specific (parameterised light), no geometric generalisation.
- relevance: Earliest learned transport response with sharp glossy indirect effects; shows such effects are learnable when geometry is fixed and the source is a low-dimensional parameter — the opposite corner from a geometry-conditioned, source-independent operator.
- url: https://www.microsoft.com/en-us/research/publication/global-illumination-radiance-regression-functions/

### Rainer, Bousseau, Ritschel, Drettakis — Neural Precomputed Radiance Transfer (see also Lyu et al., Neural Radiance Transfer Fields, ECCV 2022, arXiv 2207.13607) (2022, Eurographics 2022 (CGF 41(2))) [read_abstract_or_secondary]
- mechanism: Small per-scene networks take per-pixel G-buffer attributes and an encoded environment light and output radiance. The PRT-inspired variant first maps G-buffer inputs to a transport code, then combines it with the light code; further variants factor out albedo and split diffuse/specular tracks.
- precomputed / learned: Per-scene network weights (transport code predictor + light encoder), trained on rendered data.
- valid under geometry edits: Nothing; one fixed scene. Lighting (distant) and view vary.
- sharp glossy interreflection: View-dependent effects via a specular track; a later paper (Raghavan et al. 2023) characterises it as suited to largely diffuse scenes with less accurate indirect view-dependent effects.
- cost claims: Over 30 FPS neural renderers at a fixed parameter budget.
- limitation: Fixed scene, distant lighting, transport code is not constrained to be linear or non-negative.
- relevance: Shows that architecture mirroring the operator structure (transport code separate from light code) matters more than size; but the 'transport code' is still memorised per scene.
- url: https://repo-sam.inria.fr/fungraph/neural-prt/

### Raghavan, Xiao, Lin, Sun, Bi, Xu, Li, Ramamoorthi — Neural Free-Viewpoint Relighting for Glossy Indirect Illumination (2023, EGSR 2023 (CGF 42(4)), arXiv 2307.06335) [read_primary_source]
- mechanism: The indirect light-transport function T_k(x, w_o) is represented in the Haar wavelet basis over a cube map: outgoing radiance is a dot product of the lighting's largest (typically 64) wavelet coefficients with transport coefficients predicted by a small MLP from a tensor-factorised feature field indexed by position and wavelet index, plus reflected direction, normal and BRDF parameters.
- precomputed / learned: Per scene: about 4,000 path-traced training images; hash-grid spatial features, a 6x64x64 wavelet feature grid and an MLP (about 113 MB). Linear in lighting by construction.
- valid under geometry edits: Nothing; the paper restricts itself to static scenes and explicitly excludes dynamic objects. Retraining is required after any edit.
- sharp glossy interreflection: Its focus: glossy indirect reflections and view-dependent caustics with changing view, kept sharp by staying in an orthonormal localized basis. Reported failure: a mirror of roughness 0.05 (the reflected painting is not reconstructed); only 64 wavelets are used and selection changes can flicker.
- cost claims: 1-3 hours precomputation plus about 16 hours training on an RTX 3090 Ti; 24 FPS at 512x512 and 13 FPS at 800x600 on an RTX 4090; frame time independent of geometric complexity.
- limitation: Static geometry, distant lighting, many hours per scene, residual blur at near-mirror roughness.
- relevance: The current reference point for a learned, source-independent linear transport operator that keeps glossy indirect detail — and it does so only per fixed scene, with a localized (not low-rank) lighting basis, and still degrades at the near-mirror end.
- url: https://arxiv.org/abs/2307.06335

### Hadadan, Chen, Zwicker — Neural Radiosity (2021, SIGGRAPH Asia 2021 (ACM TOG 40(6)), arXiv 2105.12319) [read_primary_source]
- mechanism: A network representing the full 4D radiance field of one scene is optimised to minimise the norm of the rendering-equation residual, with the same network on both sides (self-training); views are rendered afterwards with one query per sample (left-hand side) or one extra bounce (right-hand side).
- precomputed / learned: Per-scene network weights and learnable multi-resolution features; source and geometry are both baked in.
- valid under geometry edits: Nothing formally; the paper shows fine-tuning from a related configuration (a few steps between two poses of one model) instead of retraining from scratch, and names dynamic scenes as future work.
- sharp glossy interreflection: Perfect mirrors are never stored: rays are traced through specular bounces until a non-specular surface and the network is queried there. Glossy reflections depend on capacity and on feeding local surface properties; narrower networks struggle with reflections.
- cost claims: Training 22 minutes (Cornell box) to about 7 hours (bedroom) on an RTX 2080; equal-time LHS rendering beats 16-spp path tracing in one example; the authors state training is not competitive with state-of-the-art Monte Carlo.
- limitation: A solver, not a reusable operator: the solution is for one scene and one lighting.
- relevance: Shows the standard division of labour for sharp relations in learned GI: trace specular chains explicitly and query the learned quantity only at rough vertices. It also is the nearest thing to a 'neural solver' of the rendering equation; no source-independence, no transfer.
- url: https://arxiv.org/abs/2105.12319

### Su, Dong, Ren, Jin, Chen, Wang, Li — Dynamic Neural Radiosity with Multi-grid Decomposition (2024, SIGGRAPH Asia 2024 Conference Papers) [read_abstract_or_secondary]
- mechanism: Neural radiosity extended to variable scenes: an explicit scene-parameter vector v conditions the radiance field, which is factorised into 3D grids, 2D feature planes over (space, variable) pairs and small MLPs.
- precomputed / learned: Per-scene feature grids and MLPs trained by the residual objective over the predefined ranges of the scene variables.
- valid under geometry edits: Edits that correspond to a point inside the trained variable ranges are served without retraining; any new degree of freedom or range requires retraining (inferred from the formulation; the project page does not state out-of-range behaviour).
- sharp glossy interreflection: Not stated on the page read; inherits the neural-radiosity treatment.
- cost claims: Abstract claims reduced training and inference cost versus prior state of the art at practical run-time; no numbers read.
- limitation: Per scene; variables must be enumerated in advance; memory grows with the number of variables.
- relevance: Represents the 'rigid pose as an input coordinate of a per-scene radiance solution' route to edit validity: validity is interpolation inside a trained hypercube, not transfer of an operator to an unseen configuration.
- url: https://woaixuexisr.github.io/papers/dynamic-neural-radiosity

### Ren, Jin, Pang, Chen, Wang, Li — Neural Cone Radiosity for Interactive Global Illumination with Glossy Materials (2025); Ren et al. — OctaOctree Neural Radiosity for Real-time Glossy Material Rendering (2026) (2025-2026, arXiv 2509.07522; arXiv 2606.08469) [read_abstract_or_secondary]
- mechanism: Both address the failure of position-encoded radiance caches on narrow lobes. Cone radiosity encodes a reflectance-aware ray cone with a pre-filtered multi-resolution hash grid approximating the BSDF lobe. OctaOctree stores outgoing radiance in an adaptive octree whose nodes hold octahedral directional maps, with coarser spatial levels carrying higher angular resolution.
- precomputed / learned: Per-scene learned encodings and networks in the neural-radiosity framework.
- valid under geometry edits: Not stated in either abstract; both are in the per-scene neural-radiosity line, so an edit implies re-solving.
- sharp glossy interreflection: Their stated purpose: cached radiance that remains accurate when queried through sharp lobes, by trading spatial for angular resolution (OctaOctree) or by lobe-matched pre-filtering (cone encoding); claimed range from mirror-like to matte.
- cost claims: Real-time, single network query at the primary hit (OctaOctree); no numbers in the abstracts.
- limitation: Abstract-level evidence only; per-scene; no statement on dynamic geometry.
- relevance: Direct evidence that 'a lookup at a near-mirror receiver exposes localised errors' is a recognised problem of cached outgoing-radiance tables, and that the published responses are angular-resolution-adaptive storage at fixed geometry.
- url: https://arxiv.org/abs/2606.08469

### Diolatzis, Philip, Drettakis — Active Exploration for Neural Global Illumination of Variable Scenes (2022, ACM TOG 41(5) / SIGGRAPH 2022, arXiv 2203.08272) [read_primary_source]
- mechanism: A per-scene MLP generator (8 layers x 512) maps per-pixel G-buffers plus an explicit scene-parameter vector (object translation/rotation, light position/intensity, material parameters, on/off events) to radiance; training samples are chosen by an MCMC 'active exploration' over the parameter space interleaved with on-the-fly path tracing and sample reuse.
- precomputed / learned: One network per variable scene, trained on path-traced patches over a normalised hypercube of 6-11 (up to 128) variables.
- valid under geometry edits: Rigid edits that are among the declared variables and inside the trained range; the paper reports no test outside the range. New edits require retraining.
- sharp glossy interreflection: All path types, including mirror reflections and caustics, are learned from data; the authors note high-frequency specular effects may not be reproduced exactly and suggest tracing specular bounces during buffer generation to reduce what the network must infer.
- cost claims: Training 5-18 hours on one RTX 6000; inference 4-6 fps at 900x900 (unoptimised).
- limitation: Stated: memory grows with the tiled variable vector; the 128-variable scene has missing shadows and highlights; scene-specific.
- relevance: Shows that sharp transport under rigid pose changes can be learned when the edit space is declared in advance and sampled adaptively, at a cost of hours per scene; it is conditioning on a pose vector, not on geometry, so it does not transfer.
- url: https://arxiv.org/abs/2203.08272

### Granskog, Rousselle, Papas, Novák — Compositional Neural Scene Representations for Shading Inference (2020); Granskog, Schnabel, Rousselle, Novák — Neural Scene Graph Rendering (2021) (2020-2021, SIGGRAPH 2020 (ACM TOG 39(4)); SIGGRAPH 2021 (ACM TOG 40(4))) [read_abstract_or_secondary]
- mechanism: Learned scene representations are partitioned into lighting, material and geometry components that can be recombined across scenes (2020); in the 2021 work geometry and material are learned vectors at scene-graph leaves, transformations (translation, bending, hue shift) are neural modules at inner nodes, and a streaming neural renderer draws a variable number of objects.
- precomputed / learned: Encoders/decoders and per-asset latent vectors trained across many scenes.
- valid under geometry edits: The learned modules are reused when objects are re-arranged via scene-graph transforms; fidelity is that of a learned image generator, with no transport guarantees.
- sharp glossy interreflection: Not a stated capability in the abstracts read; these are image-space generators evaluated on comparatively simple shading.
- cost claims: None read.
- limitation: No linearity in the source, no physical constraints; inter-object transport is whatever the generator infers.
- relevance: Precedent for modular learned components that survive re-arrangement without retraining, obtained by structure (partitioning / scene graph) rather than by a physical operator.
- url: https://research.nvidia.com/labs/rtr/publication/granskog2021graph

### Zheng, Huo, Mo, Zhong, Wu, Hua, Wang, Bao — NeLT: Object-Oriented Neural Light Transfer (2023, ACM TOG 42(5), article 163) [read_abstract_or_secondary]
- mechanism: Global illumination is decomposed into per-object neural light-transfer functions describing the light exchange between an object and its surroundings; the per-object functions are composed explicitly so objects can be transformed under dynamic lights.
- precomputed / learned: A neural transfer function per object (details of inputs and lighting encoding not read from the primary source).
- valid under geometry edits: According to the abstract and secondary summaries, per-object functions are reused under object transformation; deformation is named as missing. Not verified in the paper text.
- sharp glossy interreflection: Not verified. The abstract compares quality to a real-time denoiser, which suggests plausible rather than reference-level transport.
- cost claims: Interactive global illumination for dynamic scenes; no numbers read.
- limitation: Primary text not accessible to me; rigid-only; quality target is 'close to a real-time denoiser'.
- relevance: A learned counterpart of radiance-transfer fields: object-owned, composable, rigid-edit-tolerant transfer. Among the closest learned precedents in intent; its angular fidelity is the open question.
- url: https://history.siggraph.org/?p=164203

### Deng, Han, Fang — TransGI: Real-Time Dynamic Global Illumination With Object-Centric Neural Transfer Model (2025, arXiv 2506.09909 (preprint)) [read_primary_source]
- mechanism: Per-vertex latent vectors (7 values) plus an MLP decoder output SH transfer coefficients (4th-degree, 75 values) as a function of outgoing direction, normal and albedo; shading is the dot product with SH lighting from an 8x8x8 grid of local light probes that are re-generated every frame with a neighbour 'radiance-sharing' scheme.
- precomputed / learned: Per scene: vertex latents per object and a decoder shared across the scene (object-specific decoders are possible but costly), trained on about 41 M surface samples (about 30 minutes sampling, about 2 hours training). The transfer is linear in lighting.
- valid under geometry edits: The transfer features travel with the vertices, so the paper claims rigid transforms work without retraining; the evidence is one desk translation and one sofa shift, rotation is not separately evaluated. Inter-object light is not in the transfer; it is re-supplied by the probes each frame.
- sharp glossy interreflection: Excluded in effect: a fixed 25-function SH basis per channel; the paper states specular responses need an extra per-pixel traced ray, largest errors are on glossy surfaces, and sparse probes miss bright interreflection between two glossy objects.
- cost claims: 6.7 ms per frame versus 1.7 ms for DDGI on an RTX 5000; better RMSE/SSIM than DDGI; network 799 KB, vertex features up to 42 MB.
- limitation: Low angular frequency; inter-object interreflection limited by probe density; preprint with thin dynamic evaluation.
- relevance: A recent learned system that is explicit about the same split: object-owned linear transfer survives rigid motion, while everything between objects is delegated to a per-frame low-frequency lighting state — and sharp inter-object relations are what is lost.
- url: https://arxiv.org/abs/2506.09909

### Yu, Guo, Fathi, Chang, Chan, Gao, Funkhouser, Wu — Learning Object-Centric Neural Scattering Functions for Free-Viewpoint Relighting and Scene Composition (2023, TMLR 2023, arXiv 2303.06138) [read_primary_source]
- mechanism: Each object has a network mapping (position, distant light direction, outgoing direction) to a cumulative radiance-transfer value and density, i.e. the object's response to an unobstructed distant light. Scenes are composed by volume-rendering shadow rays through other objects' densities and Monte Carlo sampling one indirect bounce.
- precomputed / learned: One neural scattering function per object, learned from about 500 images per object.
- valid under geometry edits: Per-object functions are reused in new arrangements without retraining; inter-object shadows and one bounce are recomputed by ray marching at render time.
- sharp glossy interreflection: Not modelled: the paper states the unobstructed-distant-light assumption biases inter-reflection estimates especially for shiny objects, omits transmitted light and caustics, and computes one indirect bounce only.
- cost claims: 0.27 FPS for the base model, 16 FPS with a distilled KiloOSF variant on synthetic opaque objects.
- limitation: Distant-light response is not the near-field response; composition is approximate and stated as not guaranteed physically correct.
- relevance: Shows what goes wrong when a per-object source-independent response is defined for distant illumination and then reused in near-field multi-object configurations: near-field, sharp inter-object transport is precisely the part that is biased.
- url: https://arxiv.org/abs/2303.06138

### Ren, Huo, et al. — LightFormer: Light-Oriented Global Neural Rendering in Dynamic Scene (2024, SIGGRAPH 2024 (ACM TOG 43(4), article 75)) [read_abstract_or_secondary]
- mechanism: Instead of encoding the scene, each light generates virtual point lights that are encoded, with screen-space cues such as visibility, into a neural light representation; a pixel-light attention step gathers the light representations per shading point and a small decoder predicts radiance.
- precomputed / learned: Networks trained across data (light encoder, attention, decoder); VPLs and screen-space cues are regenerated per frame.
- valid under geometry edits: Learned weights are intended to be reused under dynamic lights, materials, cameras and animated objects, since geometry enters only through per-frame VPLs and G-buffers.
- sharp glossy interreflection: Not stated on the page read; VPL-based formulations classically handle diffuse to moderately glossy transport and fail on sharp glossy senders (as noted in Xu et al. 2014).
- cost claims: Real-time; no numbers read.
- limitation: Screen-space cues; 'reasonable and realistic' rather than reference-level; generalisation claim not quantified on the page.
- relevance: Example of obtaining edit tolerance by conditioning a learned gatherer on per-frame traced samples (here VPLs) rather than on a memorised scene — structurally similar to feeding stratified visibility rays to a generator, and with the same receiver-side sampling of sender detail.
- url: https://wylighting.github.io/lightformer

### Zeng, Dong, Peers, Wu, Tong — RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination (follow-up: Du et al., RenderFormer++, arXiv 2606.30380, abstract only) (2025, SIGGRAPH 2025, arXiv 2505.21925) [read_primary_source]
- mechanism: A sequence of triangle tokens (geometry + GGX reflectance + emission) is processed by a view-independent transformer stage modelling triangle-to-triangle light transport, then a view-dependent stage turns ray-bundle tokens into pixels, with no per-scene training.
- precomputed / learned: Transformer weights trained once on the order of millions of synthetic scenes (Objaverse objects in template rooms, Cycles 4,096-spp targets; about 8 days on 8 A100s — exact counts uncertain).
- valid under geometry edits: The weights: any edit is a new token sequence and a new forward pass; the view-independent stage can be reused across views of a static scene. It is not rotation-invariant (mitigated by augmentation).
- sharp glossy interreflection: Shows diffuse and specular interreflection and glossy reflections, but the paper reports that on average about three specular bounces are modelled correctly and higher orders are dropped; larger triangles degrade reflections.
- cost claims: 0.06-0.10 s per 512x512 frame on an A100 versus 2.7-4.7 s (Cycles, 4,096 adaptive spp + denoise) or 7.8-12 s (4,096 spp); not an equal-quality comparison.
- limitation: Stated: trained up to 4,096 triangles (quadratic attention), detail lost at 45k; only 1-8 white diffuse lights outside the scene (workaround: render lights separately and sum); single material model without textures; camera outside the scene. The follow-up adds rendering-equation-inspired attention biases, a transport-consistency loss and object-level tokens (abstract only).
- relevance: The most developed learned, geometry-conditioned transport model that transfers to unseen configurations without retraining. It is neither linear in the source nor source-independent (superposition is a workaround), it is not constrained to be non-negative or passive, and its documented weak points are higher-order specular chains and scale.
- url: https://arxiv.org/abs/2505.21925

### Xu, Varma T, Wang, Li, Wu, Wronski, Ramamoorthi, Salvi — A Generalizable Light Transport 3D Embedding for Global Illumination (2025, arXiv 2510.18189 (preprint)) [read_primary_source]
- mechanism: A scene is a point cloud (about 20k points with position, normal, albedo, emissivity; lights are points too). A PointTransformerV3 'light transport encoder' produces latent codes; a query decoder cross-attends to about 32 nearest codes to predict irradiance (or, for glossy, an incident radiance field over 1,024 directions). No rasterised or path-traced cues are input.
- precomputed / learned: Encoder/decoder weights trained across a large indoor dataset (about 5 days on four A10s; dataset about 250 CPU/GPU-days). The paper motivates the design by the transport-matrix view and an attention/matrix-vector analogy but states this is conceptual, and does not claim linearity in lighting.
- valid under geometry edits: The weights: moving a sofa, moving an area light or changing floor albedo are shown qualitatively without retraining; the encoder is re-run.
- sharp glossy interreflection: Diffuse irradiance is the main result. Glossy is described as preliminary (conductors at GGX roughness 0.1, fine-tuned encoder). Listed failures include light leaks and missed intricate indirect highlights.
- cost claims: Encoder 208 ms (amortised, view-independent) plus 368 ms decoding for 512x512; compared with 64-spp path tracing; the authors state it is not yet real-time and not a drop-in replacement for path tracing.
- limitation: Stated: distribution dependence (indoor), each training scene had a fixed light configuration, colour shifts, light leaks, no quantitative edit evaluation.
- relevance: The closest published statement of the goal 'learn a transport embedding conditioned on geometry that is reused across scenes instead of precomputing per scene', by authors from the PRT lineage. It uses no visibility rays, is diffuse-first, and reports sharp indirect highlights as a failure mode.
- url: https://arxiv.org/abs/2510.18189

### Hermosilla, Maisch, Ritschel, Ropinski — Deep-learning the Latent Space of Light Transport (2019, EGSR 2019 (CGF 38(4)), arXiv 1811.04756) [read_abstract_or_secondary]
- mechanism: A 3D point-cloud network encodes geometry, material and illumination into a per-point latent representation, which a second network projects to the 2D image; presented as a learned extension of point-based global illumination and deferred shading.
- precomputed / learned: Network weights trained across scenes for ambient occlusion, global illumination and subsurface scattering against path-traced references.
- valid under geometry edits: The weights; the network is re-evaluated on the edited point cloud.
- sharp glossy interreflection: Not a target; effects shown are smooth (AO, diffuse GI, SSS).
- cost claims: Claims most of the efficiency of 2D screen-space methods with better temporal coherence; no numbers read.
- limitation: Smooth transport only; no linearity or physical constraints.
- relevance: Early proof that a transport response conditioned on 3D geometry transfers to unseen configurations without retraining — for low-frequency effects.
- url: https://arxiv.org/abs/1811.04756

### Kallweit, Müller, McWilliams, Gross, Novák — Deep Scattering: Rendering Atmospheric Clouds with Radiance-Predicting Neural Networks (2017, SIGGRAPH Asia 2017 (ACM TOG 36(6)), arXiv 1709.05418) [read_abstract_or_secondary]
- mechanism: Low-order scattering is integrated by Monte Carlo; multiple scattering is predicted by a network fed a hierarchical 3D descriptor of the cloud density around the shading point, oriented with respect to the light, fed progressively into the network.
- precomputed / learned: A network pre-trained on tens of cloud exemplars; no per-cloud precomputation.
- valid under geometry edits: The weights: the descriptor is re-sampled from the current density, so the prediction applies to unseen and animated clouds.
- sharp glossy interreflection: Not applicable: high-order volumetric scattering is smooth; the sharp, low-order part is kept in Monte Carlo.
- cost claims: Images nearly indistinguishable from the reference within seconds, interactively, versus very long path-traced convergence for high-albedo clouds.
- limitation: One material class (clouds), one distant light direction as conditioning; smooth transport.
- relevance: The cleanest precedent for the hybrid 'path-trace the first events exactly, predict the higher-order remainder from a geometry descriptor sampled around the query, no retraining for new geometry'. It works where the remainder is smooth and the descriptor is centred on the query point.
- url: https://arxiv.org/abs/1709.05418

### Vicini, Koltun, Jakob — A Learned Shape-Adaptive Subsurface Scattering Model (2019, SIGGRAPH 2019 (ACM TOG 38(4))) [read_abstract_or_secondary]
- mechanism: A conditional autoencoder samples the exit location of subsurface light directly on the surface, conditioned on a low-order polynomial fit of the local geometry (capturing curvature, thickness, corners) and the medium parameters, replacing internal path tracing.
- precomputed / learned: A network trained once across shapes and media; at render time only a local polynomial fit of the current geometry is computed.
- valid under geometry edits: The weights; the geometric conditioning is recomputed locally, so it applies to unseen shapes.
- sharp glossy interreflection: Not applicable (diffusive subsurface transport); it is a sampling model with a pdf, used inside a path tracer.
- cost claims: Retains the speed of analytic BSSRDF models with accuracy closer to volumetric path tracing.
- limitation: Local geometric descriptor; transport is smooth and short-range.
- relevance: A learned transport response conditioned on geometry that transfers without retraining AND remains an unbiased-style sampler inside path tracing, because it is formulated as a sampling distribution rather than a table of radiance.
- url: https://rgl.epfl.ch/publications/Vicini2019Learned

### Zhu, Tang, Ma — DeepRTE: Pre-trained Attention-based Neural Network for Radiative Transfer (2025, arXiv 2505.23190 (cs.LG)) [read_abstract_or_secondary]
- mechanism: A mesh-free, attention-based neural operator for the steady-state radiative transfer equation whose architecture and training data follow Green's-function theory: it is pre-trained on delta-function inflow boundary conditions and is then applied 'zero-shot' to other inputs.
- precomputed / learned: Operator network weights (boundary/source data -> solution), few parameters by the authors' account.
- valid under geometry edits: The abstract claims zero-shot capability but does not say whether it covers unseen geometries or coefficient fields; not established.
- sharp glossy interreflection: Not applicable: participating-media RTE in scientific-computing settings, not surface rendering with specular kernels.
- cost claims: Claims higher efficiency than traditional solvers and other neural approaches; no numbers read.
- limitation: Outside graphics; small canonical domains; generalisation scope unclear. A separate neutron-transport operator-learning preprint found in the same search (arXiv 2602.15890) reports limited generalisation when the physical configuration changes.
- relevance: The nearest verified instance of explicit Green's-function operator learning for a transport equation: training on delta sources is the operator-learning analogue of source-independent supervision. I found no counterpart for the surface rendering equation.
- url: https://arxiv.org/abs/2505.23190

### Li, Wiedemann, Mitchell — Deep Precomputed Radiance Transfer for Deformable Objects (2019, I3D 2019 (Proc. ACM CGIT 2(1))) [read_abstract_or_secondary]
- mechanism: A U-Net operating on geometry images (harmonic-map parameterisation) of the current shape predicts per-frame spherical-harmonic transfer coefficients, replacing Monte Carlo PRT precomputation for each deformation state.
- precomputed / learned: A CNN trained on PRT data of one deformable object across its deformations.
- valid under geometry edits: The network is re-evaluated on the deformed shape and emits a new linear transfer operator; no retraining within the object's deformation family.
- sharp glossy interreflection: Excluded: SH transfer under low-frequency distant lighting for a single object.
- cost claims: Real-time transfer prediction versus per-frame Monte Carlo projection; compared with a linear morphable-model baseline.
- limitation: One object family, low frequency, distant light, self-transfer only.
- relevance: A verified case of exactly the pattern 'a network sees current geometry and emits the coefficients of a linear, source-independent transfer operator', demonstrated only for band-limited self-transfer.
- url: https://www.napier.ac.uk/research-and-innovation/research-search/outputs/deep-precomputed-radiance-transfer-for-deformable-objects

## Implications
- In the surveyed literature, validity under rigid edits and sharp glossy interreflection do not occur together in any precomputed or learned operator. Operators that carry glossy indirect detail (Kontkanen 2006, RRF 2013, Raghavan 2023, neural-radiosity variants) are fixed-geometry and must be recomputed or retrained after an edit; operators that survive rigid edits (neighborhood transfer, shadow fields, radiance transfer fields, MRT, TransGI, NeLT as described) are low-frequency or diffuse between objects, or drop inter-object interreflection altogether (shadow fields: direct lighting only, by the authors' statement).
- Rigid-edit validity in the classical line comes from one structural move: the operator is owned by an entity and stored in that entity's frame, and inter-entity transport is composed at run time. The composition step is where angular frequency is lost: fields are sampled on sparse shells, and a sender's response is indexed by a coarse summary of what reaches it. Shadow fields at all-frequency resolution already cost 0.1 fps and hundreds of MB for visibility alone.
- Reduced (low-rank) operators reproduce only what their input prior spans. MRT states this as a limitation (lighting outside the prior cannot be captured; one mode acts as an 'intelligent ambient term') and names high-frequency glossy transport as an open problem. A low-rank global term that compensates without geometric specificity is the documented behaviour of this class.
- All-frequency PRT obtained sharpness through sparse, nonlinear selection in a localized basis (Haar wavelets in Ng 2004, Liu 2004, Raghavan 2023), not through low rank. The sparsity pattern is a property of one configuration, which is why these operators are per-scene, and the best learned version still fails at mirror roughness 0.05 with 64 wavelets.
- Learned GI systems that show sharp effects keep the sharp relation in explicit ray tracing and query the learned quantity only at rough vertices (Neural Radiosity traces specular chains to the first non-specular hit; Diolatzis et al. suggest tracing specular bounces in buffer generation; TransGI needs an extra per-pixel ray for specular). Querying a cached outgoing-radiance table through a narrow lobe is recognised as a failure mode, and the published responses (cone encoding, OctaOctree) are angular-resolution-adaptive storage at fixed geometry.
- The one surveyed method that evaluates light -> glossy sender -> receiver transport robustly in dynamic scenes (Xu et al. 2014) does it by integrating over the sender with the sender lobe in closed form, i.e. sender-centred rather than receiver-gathered, and caches nothing; it is limited to one bounce.
- Hierarchical-radiosity work (Drettakis and Sillion 1997) shows that the operator entries invalidated by a rigid move are geometrically identifiable (links whose shaft is cut) and can be updated selectively; this was shown for diffuse links only, and the number of affected entries grows with the operator's directional resolution.
- Learned methods that tolerate edits do so in one of two ways: (a) the pose is an input coordinate of a per-scene model trained over a declared range (Diolatzis 2022: 5-18 h, quality drops as variables grow; Dynamic Neural Radiosity), which is interpolation, not transfer; or (b) a cross-scene network is re-evaluated on the new geometry (RenderFormer, Xu et al. 2025, Hermosilla 2019, LightFormer), which gives up linearity in the source and any passivity guarantee and is reported to be weak on higher-order specular chains, sharp indirect highlights and scale (RenderFormer: about three specular bounces, 4,096 training triangles, diffuse lights only; Xu et al.: glossy 'preliminary', missed intricate indirect highlights).
- Hard structural constraints of the tested mechanism (source independence, linearity, non-negativity) exist in classical transfer matrices and partly in per-scene neural PRT (dot product with lighting coefficients in Raghavan 2023 and TransGI). Among cross-scene learned models I found none that enforces them; RenderFormer handles multiple or coloured lights by rendering separately and summing.
- Geometry-descriptor-conditioned predictors that transfer without retraining and coexist with exact low-order path tracing exist for smooth transport (Deep Scattering for clouds, the shape-adaptive SSS model). In both, the predicted remainder is smooth and the descriptor is centred on the query; neither involves a narrow sender lobe far from the query.
- Economic evidence is thin for the stated target. Classical PRT speed-ups are bought with hours of per-scene or per-object precompute and tens to hundreds of MB. Cross-scene networks report 28-183x versus Cycles at 4,096 spp (RenderFormer, small scenes, not equal quality) or sub-second frames versus 64 spp while stating they are not a drop-in replacement (Xu et al. 2025). No surveyed work reports a matched-quality cost advantage over path tracing for sharp glossy interreflection after a geometry edit.
- I found no verified work applying neural-operator / Green's-function learning (DeepONet, FNO or similar) or learned preconditioners to the surface rendering equation. The nearest verified items are DeepRTE (Green's-function-motivated operator for the RTE in media), neural radiosity (residual minimisation with self-training, per scene) and generic learned preconditioners for sparse PDE systems; the operator-learning literature found alongside reports weak extrapolation when the physical configuration changes.
## Closest precedents
- Li, Wiedemann, Mitchell 2019 (DeepPRT): a network reads the current geometry and emits coefficients of a linear, source-independent transfer operator without retraining — demonstrated only for low-frequency SH self-transfer of one deformable object under distant light.
- Loos et al. 2011 (Modular Radiance Transfer): source-independent reduced direct-to-indirect operators per shape, coupled through reduced interface lightfields and re-assembled under re-arrangement — diffuse, low-rank prior, glossy explicitly left open.
- Pan et al. 2007 (radiance transfer fields) on top of Zhou et al. 2005 (shadow fields): per-object linear operator from incident light on the object to the radiance field around it, valid under arbitrary rigid motion and composed at run time — low-frequency lighting only; shadow fields themselves are direct-only.
- Xu et al. 2025 (Generalizable Light Transport 3D Embedding): cross-scene transformer that encodes geometry/material/emitter point clouds into a transport embedding reused on unseen and edited scenes — not linear in the source, no visibility rays, diffuse-first with preliminary glossy.
- Zeng et al. 2025 (RenderFormer, and RenderFormer++ 2026 by abstract): view-independent triangle-to-triangle transport stage that transfers to unseen configurations without retraining — source enters as emissive tokens (not source-independent), about three specular bounces, small scenes.
- Deng et al. 2025 (TransGI) and Zheng et al. 2023 (NeLT): object-owned learned transfer that travels with rigid objects, inter-object light supplied by a per-frame lighting state — SH-band-limited (TransGI, read) / fidelity unverified (NeLT, abstract only).
- Kallweit et al. 2017 (Deep Scattering): path-trace low orders exactly and predict the higher-order remainder from a geometry descriptor sampled at render time, no retraining for new geometry — smooth volumetric transport, descriptor centred on the query point.
- Hašan et al. 2006 / Kontkanen et al. 2006 / Lehtinen et al. 2008: the explicit direct-to-indirect and global transport operators (wavelet, 4D wavelet with glossy, meshless hierarchical) that a geometry-conditioned generator would have to reproduce — all precomputed for one fixed scene.
- Raghavan et al. 2023: learned, linear-in-lighting, wavelet-basis indirect transport with glossy reflections and caustics — per scene, static geometry, about 16 h training, degrades at near-mirror roughness.
- Zhu, Tang, Ma 2025 (DeepRTE): operator network trained on delta-function inflow (Green's-function supervision) for the radiative transfer equation — the nearest verified operator-learning analogue of source-independent supervision, outside surface rendering.
## Unverified / uncertain
- NeLT (Zheng et al., ACM TOG 2023, DOI 10.1145/3596491): existence, authors and abstract confirmed; the mechanism details (lighting encoding, how per-object functions are composed, whether inter-object glossy transport is represented, performance) come from secondary summaries only — the paper text was not accessible.
- Pan et al. 2007 (radiance transfer fields): the Eurographics digital library was unreachable for direct fetch; abstract content is from a search-result summary. Author list is reported inconsistently across records (Pan, Wang, Liu, Peng, Bao per the EG metadata). The statement that incident light on the sender is summarised coarsely is my recollection.
- Iwasaki, Dobashi, Yoshimoto, Nishita 2007 (EGSR, PRT for dynamic scenes with interreflection) and Mei, Shi, Wu 2004 (Spherical Radiance Transport Maps, CGF 23(3)): existence confirmed by library records; mechanisms known to me only at abstract/snippet level and folded into the Pan et al. entry.
- Works confirmed to exist in search results but not read, and therefore not given entries: Wang, Tran, Luebke 2004 (separable BRDF all-frequency relighting); Green, Kautz, Matusik, Durand 2006 (non-linear Gaussian view-dependent transport, rigid objects, distant light); Sloan, Luna, Snyder 2005 (local deformable PRT with zonal harmonics); Ren et al. 2006 (SH exponentiation soft shadows); Dachsbacher et al. 2007 (implicit visibility and antiradiance — I could not confirm from the abstract whether glossy transport is handled; a related paper notes dynamic objects are restricted by tessellation needs); Müller et al. 2021 (real-time neural radiance caching, online training with no pretraining 'since generalizing a pretrained network to new dynamic scenes is hard'); Xu, Zeng, Wu, Wang, Yan 2022 (lightweight neural basis functions with learned product/rotation operators — the 10-40x figure came from a truncated snippet); Coomans et al. 2024 (real-time neural rendering of dynamic light fields — rigid-object handling not confirmed); Gao et al. 2022 (Neural Global Illumination under dynamic area lights, TVCG — PDF too large to fetch); Gai, Wang, Li 2025 (arXiv 2508.08826, screen-space diffuse indirect, claims unseen-scene generalisation).
- Zheng et al. 2024, 'Neural Global Illumination via Superposed Deformable Feature Fields' (SIGGRAPH Asia 2024): seen only as a citation inside another paper's reference list; not independently verified or read.
- Details stated from memory rather than from a source read in this session: that PRT neighborhood transfer has no feedback from receiver to sender; that Ng et al. 2004 is restricted to static geometry; that gather samples in Hašan et al. 2006 are diffuse senders; that Lehtinen et al. 2008 is primarily diffuse and static; that Xu et al. 2014 needs no scene precomputation and approximates indirect visibility; whether Liu et al. 2004 includes interreflection in its transfer.
- Dynamic Neural Radiosity (Su et al. 2024): only the project-page abstract was read (the PDF exceeded the fetch limit); the supported variable types, out-of-range behaviour, glossy handling and cost numbers are not verified.
- Neural Cone Radiosity (arXiv 2509.07522) and OctaOctree Neural Radiosity (arXiv 2606.08469), RenderFormer++ (arXiv 2606.30380), DeepRTE (arXiv 2505.23190): arXiv abstracts only; all are preprints as far as I could see. A search snippet attributed the phrase 'pre-filtered neural operators' to the cone-radiosity paper; the abstract itself speaks of a pre-filtered multi-resolution hash grid.
- RenderFormer training-set size and schedule: the extracted HTML appeared to duplicate digits; the figures given (order of millions of scenes, about 8 days on 8 A100s) are approximate.
- TransGI is an arXiv preprint with no peer-reviewed venue listed; the '75 values = 25 SH coefficients x 3 channels' reading is an interpretation, and rotation under rigid motion was not separately evaluated in the paper.
- RRF (Ren et al. 2013): I read only the abstract page; that geometry is fixed and only the local light and view vary is consistent with the abstract but the treatment of moving objects was not checked in the paper.
- Negative search results are not proof of absence: I found no paper applying DeepONet/FNO-style neural operators, learned Green's functions or learned preconditioners specifically to the surface rendering equation, and no 'neural shadow fields' style work, in a limited number of queries. Foundational operator formalisms (Arvo 1995; Veach 1997) and static-geometry probe methods (e.g. Kristensen et al. 2005; Silvennoinen and Lehtinen 2017) were not searched or verified in this session.


# ANGLE: Theory and practice of sharp / specular light transport and of principled partitions of transport: Heckbert path classes and multi-pass diffuse/specular splits, why gathering fails and shooting / bidi

### Heckbert, 'Adaptive Radiosity Textures for Bidirectional Ray Tracing' (1990, SIGGRAPH 1990 (Computer Graphics 24(4))) [read_abstract_or_secondary]
- mechanism: Splits each surface interaction into diffuse plus specular; the specular part is computed on the fly as in ray tracing, the diffuse part (radiosity) is stored in adaptive per-surface textures ('rexes') filled by a light pass that shoots photons from lights and bright surfaces, then read by an eye pass. The regular-expression path notation L(S|D)*E used to classify which paths each pass owns is attributed to Heckbert (Veach's thesis section 8.3.1 reviews it as 'Heckbert's regular expression notation').
- precomputed / learned: View-independent radiosity textures on diffuse surfaces (the L(S|D)*D prefix), adaptively subdivided to the resolution the image needs; a 'size pass' decides that resolution.
- valid under geometry edits: Nothing is designed for edits: the textures hold source- and geometry-specific radiosity, so any edit that changes visibility or the light paths requires re-shooting the affected light paths.
- sharp glossy interreflection: Caustics (LS+D) are deposited by shooting from the light, never gathered; specular chains toward the eye (DS*E) are traced from the eye. Glossy (neither D nor S) has no natural owner in the notation.
- cost claims: None quantitative in the abstract.
- limitation: The exclusive split relies on a binary D/S material model; the stored quantity is radiosity, not a transport operator, so it is source-dependent.
- relevance: Origin of the path-class bookkeeping that makes a partition provably exclusive: a path belongs to exactly one class by the D/S types of its vertices, and the light -> sharp reflector -> diffuse family is assigned to a light-driven (shooting) pass by construction.
- url: https://publications.ri.cmu.edu/adaptive-radiosity-textures-for-bidirectional-ray-tracing

### Wallace, Cohen, Greenberg, 'A Two-Pass Solution to the Rendering Equation' and Sillion & Puech, 'A General Two-Pass Method Integrating Specular and Diffuse Reflection' (1987 / 1989, SIGGRAPH 1987 / SIGGRAPH 1989) [read_abstract_or_secondary]
- mechanism: Pass 1 is a view-independent radiosity solve extended so that diffuse-to-diffuse transfer includes paths that pass via specular reflection/transmission (Sillion & Puech: 'extended form factors', computed by ray-traced adaptive sampling for arbitrary geometry and refraction); pass 2 is view-dependent and adds the specular chain toward the eye. Sillion & Puech analyse the approaches by 'introducing the corresponding reflection operators'.
- precomputed / learned: A geometry-dependent, source-independent matrix of (extended) form factors between diffuse patches; then a source-dependent radiosity solution.
- valid under geometry edits: The form-factor matrix is purely geometric, so entries whose patch pair and intervening specular chain are untouched by an edit remain valid; entries whose visibility or specular chain changed must be recomputed (the papers do not address edits).
- sharp glossy interreflection: Ideal specular only. The D S* D transfer is folded into the operator by tracing through the mirrors from the diffuse sender/receiver, i.e. the specular chain is resolved geometrically per patch pair rather than by angular binning. No glossy lobes.
- cost claims: None quantitative in the abstracts; extended form factors cost a ray tree per form-factor sample.
- limitation: Spatial resolution of the caustic is the patch mesh; restricted to diffuse + ideal specular; limited specular depth in the 1987 version (planar mirrors, recalled from memory).
- relevance: Earliest precedent for a geometry-only linear operator that already contains sharp specular interreflection: the split 'D-to-D including S* (stored) / S*E (view pass)' is physically exclusive, and the sharp relation is carried by exact ray geometry inside the operator entry, not by a directional basis.
- url: https://history.siggraph.org/?p=116088 ; https://history.siggraph.org/?p=119385

### Immel, Cohen, Greenberg, 'A Radiosity Method for Non-Diffuse Environments' and Sillion, Arvo, Westin, Greenberg, 'A Global Illumination Solution for General Reflectance Distributions' (1986 / 1991, SIGGRAPH 1986 / SIGGRAPH 1991) [read_abstract_or_secondary]
- mechanism: Finite-element radiance: Immel et al. discretise surfaces and the hemisphere of directions and solve for 'the light leaving each surface in each direction' (viewer-independent); Sillion et al. replace direction bins by spherical-harmonic coefficients for both BRDFs and per-patch outgoing intensity distributions, inside a two-pass ray-casting radiosity framework.
- precomputed / learned: A patch x direction (bins or SH) representation of outgoing radiance and the corresponding patch-direction to patch-direction transfer coefficients.
- valid under geometry edits: Not addressed. Transfer coefficients are geometric and in principle reusable where unaffected; the solved directional radiance is source- and geometry-specific.
- sharp glossy interreflection: Immel et al. demonstrate 'secondary light sources' such as a spotlight reflected by a mirror onto a wall, but only at the resolution of the direction discretisation; Sillion et al. explicitly keep ideal specular out of the SH representation and handle it by a separate ray-traced pass (recalled from memory beyond the abstract).
- cost claims: None in the abstracts; the method of Immel et al. is widely described (e.g. by Christensen et al. 1997) as tractable only for very small environments.
- limitation: Storage and solve cost scale with patches x directions; a fixed angular basis blurs or aliases narrow lobes; SH ringing for sharp lobes.
- relevance: This is the same data structure as a (surface patch x direction bin) table driven by a linear operator. The historical outcome was that fixed directional discretisations were kept for broad lobes and sharp lobes were moved to a ray-traced pass.
- url: https://history.siggraph.org/?p=116032 ; https://history.siggraph.org/?p=120586

### Aupperle & Hanrahan, 'A Hierarchical Illumination Algorithm for Surfaces with Glossy Reflection' (1993, SIGGRAPH 1993) [read_abstract_or_secondary]
- mechanism: Reformulates glossy transport as discrete three-point transport (source element -> reflector element -> receiver element) with a new reflectance measure, 'area reflectance', which yields an error estimate over triples of elements and hence hierarchical refinement.
- precomputed / learned: A hierarchy of element-triple interactions refined to an error bound; radiance is stored per element pair (i.e. direction is represented by 'toward which other element').
- valid under geometry edits: Not addressed. Triple links are geometric, so they are reusable only if none of the three elements nor their mutual visibility changed.
- sharp glossy interreflection: Glossy lobes are resolved by refining the triples until the area reflectance is well approximated; a sharper lobe forces deeper refinement of all three elements.
- cost claims: O(n log n) interactions for sufficient refinement (nearly linear in trials) versus O(n^3) element triples for a naive implementation.
- limitation: Refinement depth, and so the constant, grows as lobes narrow; ideal mirrors are outside the finite-element formulation.
- relevance: Shows an alternative directional parameterisation: direction encoded by sender-receiver element pairs (two-point radiance) instead of fixed angular bins, so angular resolution adapts to geometry and BRDF automatically - at the price of the link structure depending on scene geometry.
- url: https://graphics.stanford.edu/papers/glossy

### Christensen, Stollnitz, Salesin, DeRose, 'Global Illumination of Glossy Environments Using Wavelets and Importance' and Christensen, Lischinski, Stollnitz, Salesin, 'Clustering for Glossy Global Illumination' (1996 / 1997, ACM TOG 15(1) / ACM TOG 16(1)) [read_primary_source]
- mechanism: Radiance and importance are represented in a four-dimensional (2 spatial + 2 angular) Haar wavelet basis per patch, the transport operator is wavelet-decomposed (transport coefficients are six-dimensional integrals), refinement is importance-driven, and a final gather renders the image. The 1997 paper treats clusters as points carrying directional distributions of radiance and importance, with error bounds for cluster-to-cluster transfers.
- precomputed / learned: Sparse adaptive 4D wavelet coefficients of radiance / importance and the sparse set of transport links actually needed for the view.
- valid under geometry edits: Not addressed; link structure and refinement are specific to the geometry, lighting and (through importance) the view.
- sharp glossy interreflection: Glossy reflection is handled by adaptive refinement of angular wavelets where radiance x importance is large. The 1997 paper calls the 1996 method 'the wavelet version of Immel's algorithm' and notes these methods are 'the most efficient to date ... in small glossy environments'.
- cost claims: 1996 method needs O(p^2) initial interactions for p surfaces; clustering gives O(p) links / O(p) solve for fixed directional resolution, with O(p log p) hierarchy construction.
- limitation: Per-cluster storage is proportional to the number of directional basis functions; complexity claims hold 'for a fixed directional resolution', i.e. they do not cover the growth needed for sharp lobes. A final gather is required to hide finite-element artefacts.
- relevance: The most developed attempt at a hierarchical spatial-angular basis for glossy interreflection. Its affordability came from adaptivity driven by the specific light and view (importance), which a source-independent and view-independent operator does not have.
- url: https://www.seanet.com/~myandper/tog96.pdf ; https://www.seanet.com/~myandper/tog97.pdf

### Chen, Rushmeier, Miller, Turner, 'A Progressive Multi-Pass Method for Global Illumination' (1991, SIGGRAPH 1991) [read_abstract_or_secondary]
- mechanism: Combines progressive-refinement radiosity, Monte Carlo path tracing and light ray tracing, organised as a radiosity pass, a high-frequency refinement pass and a low-frequency refinement pass, with 'light source reclassification' and 'caustics reconstruction'.
- precomputed / learned: A coarse radiosity solution used as a preprocess for path tracing, plus caustic maps reconstructed from light rays.
- valid under geometry edits: Not addressed.
- sharp glossy interreflection: High-frequency transport (caustics) is taken out of the radiosity solution and computed by light ray tracing; care is needed so each path class is counted once (source reclassification).
- cost claims: None quantitative in the abstract; progressive display of 'most useful information ... as early as possible'.
- limitation: Partition is by frequency class chosen per source and material type; glossy middle ground is handled by path tracing.
- relevance: A worked example of a partition by frequency content (low-frequency stored / high-frequency traced from the light / remainder path traced) with explicit bookkeeping to keep the classes exclusive.
- url: https://history.siggraph.org/?p=120584

### Jensen, 'Global Illumination using Photon Maps' (1996, Eurographics Rendering Workshop 1996 (Rendering Techniques '96)) [read_abstract_or_secondary]
- mechanism: Two-pass: photons are emitted from lights, traced, and stored at every non-specular hit in a geometry-independent kd-tree; a high-resolution caustics photon map is visualised directly, a low-resolution global map is read after a final-gather bounce and also guides sampling.
- precomputed / learned: Source-dependent flux samples (photon maps), decoupled from surface tessellation.
- valid under geometry edits: Nothing by design; photons whose paths intersect the changed geometry are invalid (later selective photon tracing work re-shoots only those; see unverified list).
- sharp glossy interreflection: Caustic paths are found by shooting from the light, which samples the reflector's lobe exactly; density estimation then trades variance for spatial blur. The abstract motivates the method by radiosity becoming 'prohibitive' with glossy reflections.
- cost claims: None quantitative in the abstract; compared against the Radiance program.
- limitation: Stores radiance for one lighting state; density-estimation bias (blur) on the caustic; on glossy receivers needs many photons or path continuation.
- relevance: The standard demonstration that the light -> specular -> diffuse family is cheap from the light side and expensive from the receiver side, and that the practical partition is 'caustic part read directly, smooth part read through a diffuse gather'.
- url: https://cseweb.ucsd.edu/~henrik/papers/dag96.html

### Veach, 'Robust Monte Carlo Methods for Light Transport Simulation' (PhD thesis), esp. chapters 4, 7, 8.3 (1997, Stanford University) [read_primary_source]
- mechanism: Linear-operator formulation of transport (scattering and propagation operators, measurement, importance transport by adjoint operators), a self-adjoint formulation for physically valid scenes, and the path-integral formulation. Section 8.3 extends Heckbert's notation to full paths and proves Theorem 8.3: a path can be generated by local sampling iff it contains the substring DD; otherwise 'there are not enough D vertices'. Section 8.3.4 proposes 'connectors' that generate specular chains between two D vertices.
- precomputed / learned: Nothing; this is theory plus unbiased estimators (BDPT with multiple importance sampling, Metropolis).
- valid under geometry edits: Not applicable (no stored state).
- sharp glossy interreflection: A specular vertex forces one neighbour to be generated by sampling its BSDF (Lemma 8.1). For near-specular configurations the paths are not missing but 'will be sampled with high variance'. Bidirectional sampling succeeds for LS+DE because the specular vertex is sampled from the light side and the DD edge is the connection.
- cost claims: None for this section.
- limitation: The theorem is for ideal S; for narrow glossy lobes it becomes a variance statement, not an impossibility.
- relevance: Gives the exact criterion for which path families a receiver-side gather can and cannot resolve, and the vocabulary (chain separators, connectors) for partitions that are exclusive. A fixed set of receiver rays that must land inside a sender's narrow lobe is the high-variance limit of the missing-DD case.
- url: https://graphics.stanford.edu/papers/veach_thesis/

### Kaplanyan & Dachsbacher, 'Path Space Regularization for Holistic and Robust Light Transport' (2013, Computer Graphics Forum 32(2) (Eurographics 2013)) [read_primary_source]
- mechanism: Identifies paths that cannot be sampled without bias (Veach's criterion) and selectively mollifies individual S interactions into D (angular delta replaced by a finite-support kernel of angle alpha = arctan(r/l)), with a bandwidth that shrinks to zero under derived consistency conditions.
- precomputed / learned: Nothing; an estimator modification.
- valid under geometry edits: Not applicable.
- sharp glossy interreflection: Sharp lobes are widened just enough to be connectable, only on paths that need it; bias vanishes only as the mollification bandwidth goes to zero. Notes photon mapping relaxes the DD condition by merging light and camera sub-paths.
- cost claims: Equal-time comparisons (e.g. 2 h) showing effects missing from PT / MLT / PPM; no asymptotic cost claim beyond convergence-rate conditions.
- limitation: Biased at any finite bandwidth; convergence slower than unbiased sampling where the latter exists.
- relevance: Formalises what a finite angular bin does to a sharp lobe: it is a mollifier of fixed, non-vanishing bandwidth, so its error is a blur of that width and does not decrease with more samples.
- url: https://cg.ivd.kit.edu/publications/p2013/PSR_Kaplanyan_2013/PSR_Kaplanyan_2013.pdf

### Mitchell & Hanrahan, 'Illumination from Curved Reflectors' (1992, SIGGRAPH 1992) [read_abstract_or_secondary]
- mechanism: Computes illumination reflected from curved mirrors onto other surfaces by finding extremal (Fermat) paths from the light to the receiver via the mirrors with interval analysis, then computing irradiance from the Gaussian curvature of the geometric wavefront.
- precomputed / learned: Nothing stored; per-receiver-point root finding.
- valid under geometry edits: Nothing stored; but the formulation is purely geometric, so after a rigid edit the same solver applies unchanged. Chen & Arvo (TOG 2000, 'Theory and Application of Specular Path Perturbation', known here only from abstract-level descriptions) expand a known specular path in a Taylor series to update it under small changes.
- sharp glossy interreflection: Exact for ideal mirrors: the light -> mirror -> receiver relation is a discrete set of deterministic paths with a closed-form intensity (wavefront curvature), not an integral that needs angular sampling.
- cost claims: None in the abstract.
- limitation: Ideal specular, implicit/analytic surfaces; interval Newton cost; multiple bounces compound.
- relevance: Establishes that the sharp sender -> receiver relation is determined by geometry alone (source position, reflector shape, receiver point) and is finite-dimensional per endpoint pair - the information a binned linear operator has to approximate by resolution.
- url: https://graphics.stanford.edu/papers/reflectors

### Jakob & Marschner, 'Manifold Exploration: A Markov Chain Monte Carlo Technique for Rendering Scenes with Difficult Specular Transport' (2012, ACM TOG 31(4) (SIGGRAPH 2012)) [read_primary_source]
- mechanism: Contributing specular paths form a manifold in path space defined by half-vector constraints (half-vector = normal at each specular vertex); a Newton-style iteration using only local differential geometry walks along it. Glossy chains are handled by 'offset manifolds' parallel to the specular manifold, since their integrand 'has its energy concentrated in a thin band near the specular manifold'.
- precomputed / learned: Nothing stored.
- valid under geometry edits: Not applicable; the constraint is local geometry so it holds after any edit.
- sharp glossy interreflection: Directly targeted: specular, near-specular and glossy chains by the same algorithm. The paper stresses that 'nearly specular paths involving glossy materials ... can be more problematic' because they elude special mechanisms for ideal specular.
- cost claims: Evaluated by equal-time comparison against prior MCMC methods; no closed-form cost.
- limitation: MCMC (correlated samples, needs seeding); requires differentiable geometry.
- relevance: Quantifies the dimensionality loss: each specular vertex removes two dimensions from path space, so the sharp family lives on a lower-dimensional set that uniform angular sampling or binning hits with vanishing measure.
- url: https://rgl.epfl.ch/publications/Jakob2012Manifold

### Zeltner, Georgiev, Jakob, 'Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints' (2020, ACM TOG 39(4) (SIGGRAPH 2020)) [read_primary_source]
- mechanism: Given two shading points, stochastically initialises a Newton manifold walk to find a specular sub-path connecting them; the set of solutions is 'discrete and finite', and an unbiased estimator of the reciprocal probability of finding each solution (basin-of-convergence area) is built by repeated trials. Generalises manifold next-event estimation (Hanika et al. 2015).
- precomputed / learned: Nothing stored.
- valid under geometry edits: Not applicable; purely geometric, usable inside unidirectional path tracing after any edit.
- sharp glossy interreflection: Handles reflective and refractive caustics, glints and SDS paths; a biased variant lowers variance. States the root problem: at specular interfaces the law of reflection 'drastically lowers the probability of sampling a valid configuration'.
- cost claims: Glint variant 'reduces memory usage hundred-fold compared to prior work' (per preprint summary); cost per connection is several Newton walks.
- limitation: Cost grows with the number of solutions and specular chain length; unbiased weight estimation adds variance.
- relevance: The current reference for computing the light -> sharp reflector -> diffuse connection per query from geometry alone, with no stored transport; it defines the 'full transport' cost that any stored representation of that family competes with.
- url: https://rgl.epfl.ch/publications/Zeltner2020Specular

### Smits, Arvo, Salesin, 'An Importance-Driven Radiosity Algorithm' (1992, SIGGRAPH 1992) [read_abstract_or_secondary]
- mechanism: Hierarchical radiosity refined by the product of two transported quantities: power from the lights and importance (the adjoint solution) from the visible surfaces, so that only interactions that matter to the image are refined.
- precomputed / learned: A view- and light-specific hierarchy of links plus radiosity and importance solutions.
- valid under geometry edits: Not addressed; the refinement is tied to the view, lights and geometry.
- sharp glossy interreflection: Diffuse only (the glossy extension is Christensen et al. 1996 and Aupperle & Hanrahan's importance paper).
- cost claims: 'Dramatic speedups' over previous radiosity algorithms for environments with detail at a wide range of scales (from a search-result summary of the abstract).
- limitation: View-dependent solution; diffuse.
- relevance: States the adjoint principle a designer must face: the resolution needed in a stored transport representation is set by what reads it. A diffuse reader tolerates coarse error; a near-mirror reader assigns high, localised importance to individual table entries.
- url: https://history.siggraph.org/?p=120440

### Hanrahan, Salzman, Aupperle, 'A Rapid Hierarchical Radiosity Algorithm'; Gortler, Schroeder, Cohen, Hanrahan, 'Wavelet Radiosity'; Smits, Arvo, Greenberg, 'A Clustering Algorithm for Radiosity in Complex Environments' (1991 / 1993 / 1994, SIGGRAPH 1991 / 1993 / 1994) [read_abstract_or_secondary]
- mechanism: The form-factor (transport) matrix is represented hierarchically by links between patches at the coarsest level that meets an error bound (HR); wavelet radiosity shows HR is the Haar instance of projecting the kernel into a wavelet basis; clustering groups whole objects so links can exist above the input-surface level with bounded error.
- precomputed / learned: A block-sparse, multiresolution representation of the diffuse transport operator (links) plus the radiosity solution.
- valid under geometry edits: Links whose shaft (set of lines between the two elements) is untouched remain valid; this is what the later line-space hierarchy exploits.
- sharp glossy interreflection: Excluded: Lambertian environments only.
- cost claims: HR: at most O(n) blocks vs n^2 form factors. Wavelet radiosity: linear-time algorithms. Clustering: initial linking O(s log s) time and O(s) space instead of O(s^2), about two orders of magnitude speed-up on moderate scenes.
- limitation: The linear-complexity argument rests on kernel smoothness between well-separated elements; visibility discontinuities and near-contact pairs force refinement.
- relevance: The classical proof that a source-independent transport operator compresses well hierarchically when the kernel is smooth - and the baseline against which glossy and near-contact transport break the compression.
- url: https://graphics.stanford.edu/papers/rad/ ; https://graphics.stanford.edu/papers/wavrad ; https://history.siggraph.org/?p=119727

### Lischinski, Tampieri, Greenberg, 'Discontinuity Meshing for Accurate Radiosity' (1992, IEEE Computer Graphics and Applications 12(6)) [read_abstract_or_secondary]
- mechanism: Object-space algorithm that computes the loci of radiance discontinuities caused by an area source and occluders in a diffuse polyhedral scene and places mesh element boundaries exactly on them.
- precomputed / learned: A geometry- and source-dependent mesh aligned with visibility discontinuities.
- valid under geometry edits: The discontinuity loci move with any source or occluder motion, so the mesh must be recomputed for the affected source/occluder pairs.
- sharp glossy interreflection: Not handled (diffuse); addresses the other source of sharp spatial structure, visibility.
- cost claims: Not retrieved.
- limitation: Combinatorial growth of discontinuity surfaces with scene complexity; primary sources only in practice.
- relevance: Shows that sharp spatial features of transported light are located by geometry (source x occluder edges), not by a fixed basis; a fixed patch grid cannot represent them without error at arbitrary positions.
- url: https://ieeexplore.ieee.org/document/163622

### Chen, 'Incremental Radiosity: An Extension of Progressive Radiosity to an Interactive Image Synthesis System' (1990, SIGGRAPH 1990) [read_abstract_or_secondary]
- mechanism: After a change (light switched, surface attribute changed, object added or moved), computes the minimum re-propagation by shooting corrective (including negative) energy for the affected interactions instead of restarting progressive radiosity.
- precomputed / learned: A converged or partially converged view-independent radiosity solution that is corrected incrementally.
- valid under geometry edits: Radiosity contributions between pairs whose mutual visibility is unchanged; the method must find which shooter-receiver interactions the moved object newly blocks or unblocks.
- sharp glossy interreflection: Excluded (diffuse radiosity).
- cost claims: Rendering time 'may drastically' drop because modelling changes are coherent and affect only parts of an image (abstract paraphrase).
- limitation: Source-dependent state; correction cost grows with the number of shooters whose light reached the changed region.
- relevance: First statement of the dynamic-update question as 'what is the minimum recomputation after an edit', and of the answer that the dirty set is defined by changed visibility between pairs, not by the moved object alone.
- url: https://history.siggraph.org/?p=122056

### Drettakis & Sillion, 'Interactive Update of Global Illumination Using a Line-Space Hierarchy' (1997, SIGGRAPH 1997) [read_abstract_or_secondary]
- mechanism: In hierarchical radiosity with clustering, each link is augmented with a shaft representing the set of lines through both linked elements; the links thus form an implicit hierarchy over line space, which is traversed to find links intersected by a moving object, remove subdivision no longer needed, and re-solve only the modified paths in the hierarchy.
- precomputed / learned: The link hierarchy (a hierarchical, geometry-only representation of the diffuse transport operator) with shafts, plus the current solution.
- valid under geometry edits: All links whose shafts the moving object does not enter or leave; only modified links are re-evaluated and the push-pull is redone along affected branches.
- sharp glossy interreflection: Not handled; the abstract frames the work within hierarchical radiosity (diffuse).
- cost claims: 'Interactive update rates' for scenes of several thousand polygons, with a quality / frame-rate trade-off by limiting traversal depth (a search summary quotes about 2-3 frames per second; that figure was not seen in the abstract).
- limitation: Diffuse only; number of links to test grows with scene complexity; refinement that depends on current lighting must be redone.
- relevance: The classical answer to 'which part of a stored transport operator does a rigid edit invalidate': index the operator by the lines it integrates over and intersect the moving object with that index.
- url: https://history.siggraph.org/?p=117251

### Granier & Drettakis, 'Incremental Updates for Rapid Glossy Global Illumination' (2001, Computer Graphics Forum 20(3) (Eurographics 2001)) [read_abstract_or_secondary]
- mechanism: Unified algorithm: hierarchical radiosity with clustering for diffuse transport and particle tracing for non-diffuse transport; on an edit, a dynamic spatial structure limits the particles that must be re-emitted, and the line-space hierarchy updates the diffuse part. Caustics are reconstructed either on the radiosity mesh or in dedicated caustic textures when high frequencies appear.
- precomputed / learned: Diffuse link hierarchy and solution; stored particle paths / caustic textures for the non-diffuse part.
- valid under geometry edits: Diffuse links outside the moved object's shafts; particles whose paths do not cross the changed region. Affected particles are re-shot.
- sharp glossy interreflection: Non-diffuse transport is taken out of the finite-element operator altogether and carried by light-side particle tracing with its own incremental mechanism.
- cost claims: Rapid local modification; 'quasi-interactive' updates with slight quality loss during motion (from search-result abstract text).
- limitation: Two separate mechanisms with separate invalidation logic; the non-diffuse part is stored per light state, not as an operator.
- relevance: The closest classical precedent to the exact problem (glossy GI kept valid under local geometry edits): its answer was a split in which only the diffuse part is a reusable geometric operator and the sharp part is recomputed by selective shooting.
- url: https://diglib.eg.org/handle/10.2312/8873

### Dachsbacher, Stamminger, Drettakis, Durand, 'Implicit Visibility and Antiradiance for Interactive Global Illumination' (2007, ACM TOG 26(3) (SIGGRAPH 2007)) [read_primary_source]
- mechanism: Reformulates the rendering equation with an unoccluded transport operator and propagates an extra directional quantity, antiradiance, emitted backwards from every surface to cancel light that wrongly passed through it; solved with spatial x directional finite elements (e.g. 128 direction bins) and hierarchical links, on the GPU.
- precomputed / learned: A hierarchy of unoccluded links (pure pairwise geometry, no visibility) with filtered form factors.
- valid under geometry edits: Pairwise propagation 'does not depend on occlusion due to other objects', so after a move 'we only have to update links to and from moving objects'; with bounded motion the link hierarchy is kept and only unoccluded form factors are recomputed.
- sharp glossy interreflection: 'For low-frequency glossy lighting, our approach can achieve a good approximation'; directional resolution limits sharpness (128 vs 512 bins visibly changes shadow blur). The symmetric iteration is not provably convergent: 'in simplified 2D examples with multiple occlusion and perfect mirror-BRDFs, eigenvalues larger than one occur ... Highly-specular BRDFs might create more problems'.
- cost claims: Office scene 18,253 elements / 632,098 links at 7-10 fps for 4 iterations; glossy scenes 6-15 fps; under 140 MB GPU memory.
- limitation: Two coupled quantities; directional discretisation of antiradiance sets both shadow and gloss resolution; the unoccluded operator 'can create energy'.
- relevance: A physically exact re-partition that makes the stored operator local under rigid edits by moving visibility out of it. It shows the cost of that move: occlusion becomes a directional cancellation problem whose accuracy is again bounded by the angular basis, and stability is no longer guaranteed for sharp BRDFs.
- url: http://www-sop.inria.fr/reves/Basilic/2007/DSDD07/ImplicitVisibilityAndAntiradiance.pdf

### Arvo, 'The Role of Functional Analysis in Global Illumination' (with Arvo, Torrance, Smits, 'A Framework for the Analysis of Error in Global Illumination Algorithms', SIGGRAPH 1994) (1995, Eurographics Rendering Workshop 1995 (Rendering Techniques '95)) [read_abstract_or_secondary]
- mechanism: Poses the rendering equation in terms of linear operators on infinite-dimensional function spaces; obtains operator norms from thermodynamic principles and shows common function spaces are closed under global illumination. The 1994 paper classifies error into perturbed boundary data, discretisation (projection onto a finite basis) and computation (inexact form factors / visibility, truncated iteration), with bounds for each.
- precomputed / learned: Nothing; analysis.
- valid under geometry edits: Not applicable. (In the operator factorisation into a local reflection operator and a geometric field/propagation operator - notation recalled from memory - only the geometric factor depends on scene layout.)
- sharp glossy interreflection: Per Soler et al. 2022, 'Arvo analyses the specific case of specular transport and explains that specular reflections result in a non-compact operator'.
- cost claims: None.
- limitation: Bounds are worst-case norms; they do not localise error.
- relevance: Source of the operator vocabulary (norms below one from energy conservation, Neumann series convergence, error from projecting an operator onto a finite basis). Passivity of a learned operator is the discrete counterpart of the thermodynamic norm bound.
- url: https://www.sciweavers.org/node/195712 ; https://history.siggraph.org/?p=119681

### Soler, Molazem, Subr, 'A Theoretical Analysis of Compactness of the Light Transport Operator' (2022, SIGGRAPH 2022 (ACM TOG / conference proceedings)) [read_primary_source]
- mechanism: Proves that the light transport operator T on radiance is not compact (it is a partial integral operator), even with all-Lambertian materials; the lower-dimensional radiant-exitance operator is compact only when its kernel is bounded (purely curved scenes), and its kernel is unbounded 'next to concave abutting edges, corners, and contact points between surfaces'. The global reflectance operator K is non-compact because it is the identity in space tensor a local angular operator.
- precomputed / learned: Nothing; theory.
- valid under geometry edits: Not applicable.
- sharp glossy interreflection: Explains 'the difficulties in extending algorithms that perform well on scenes with diffuse materials to scenes with glossy materials'; says T 'does not lend itself to low-dimensional approximations except in specific cases - away from edges and diffuse surfaces'.
- cost claims: None.
- limitation: Worst-case statement over all light distributions; says pointwise convergence still holds, so 'a well-crafted approximation ... for a specific lighting distribution' is possible.
- relevance: The strongest theoretical statement on the question: 'any discretization that relies on a finite-rank or nonadaptive finite-bases is susceptible to unbounded error over arbitrary light distributions'. A source-independent operator on a fixed patch x direction basis is exactly a light-agnostic finite-rank approximation; the paper also names probes and low-rank real-time GI as implicit instances. It adds that visibility discontinuities 'are not related to boundedness or compactness'.
- url: https://homepages.inf.ed.ac.uk/ksubr/Files/Papers/SIG22_Compactness.pdf

### Durand, Holzschuch, Soler, Chan, Sillion, 'A Frequency Analysis of Light Transport' (2005, ACM TOG 24(3) (SIGGRAPH 2005)) [read_primary_source]
- mechanism: Local spatio-angular Fourier analysis of radiance: free-space travel is a shear in the space-angle frequency domain, occlusion is a convolution by the blocker spectrum, reflection is a multiplication by the BRDF spectrum (a glossy BRDF 'band-limits' the angular signal), curvature is a shear that 'results in a transfer from the directional frequencies to the spatial frequencies'.
- precomputed / learned: Nothing; bandwidth predictions used to drive adaptive sampling (demonstrated at 0.05 shading samples per pixel on average).
- valid under geometry edits: Not applicable; the predictions are local functions of BRDF bandwidth, curvature, incidence angle and blocker distance, all recomputable after an edit.
- sharp glossy interreflection: Gives an approximate bound on required sampling rate that grows with BRDF angular cutoff, curvature, grazing angle and inverse distance to blockers (harmonic mean of blocker distance); 'curvature has no effect on the directional bandwidth of the outgoing light field'. The authors' FAQ notes infinite predicted bandwidth at contact points.
- cost claims: Proof-of-concept only (20,000 shading samples for an 800 x 500 image).
- limitation: First-order, local, single-bounce derivations; worst-case blocker spectrum; not an algorithm for multi-bounce transport.
- relevance: Provides the quantitative rule for what a finite spatial-angular basis can carry: the angular cutoff of the sender's lobe sets a minimum angular sample density, and after travel over distance d and reflection off curved geometry that angular bandwidth reappears as spatial bandwidth on the receiver. Sampling the sender's lobe with a fixed receiver-side ray set below this rate aliases.
- url: https://people.csail.mit.edu/fredo/PUBLI/Fourier/Fourier.pdf

### Ramamoorthi & Hanrahan, 'A Signal-Processing Framework for Inverse Rendering' (2001, SIGGRAPH 2001) [read_abstract_or_secondary]
- mechanism: Treats the reflected light field (distant lighting, convex object) as a convolution of lighting and BRDF, i.e. a product of spherical-harmonic coefficients; inverse rendering is deconvolution.
- precomputed / learned: Nothing; theory (used later to justify low-order SH for diffuse transfer).
- valid under geometry edits: Not applicable (no interreflection or occlusion in the model).
- sharp glossy interreflection: The BRDF is the filter: Lambertian keeps only the lowest orders, a mirror has a flat spectrum (no band limit). Excludes interreflection.
- cost claims: None.
- limitation: Distant lighting, no shadows or interreflection.
- relevance: Gives the per-material angular bandwidth: the number of angular basis functions needed at a reflector scales with the inverse lobe width, and a mirror needs an unbounded number - so 'broad transport' and 'sharp transport' are separated by BRDF spectrum, a property known per material before any geometry is seen.
- url: https://cseweb.ucsd.edu/~ravir/papers/invrend/

### Mahajan, Kemelmacher-Shlizerman, Ramamoorthi, Belhumeur, 'A Theory of Locally Low Dimensional Light Transport' (2007, ACM TOG 26(3) (SIGGRAPH 2007)) [read_primary_source]
- mechanism: Uses Szego's eigenvalue theorem on canonical configurations to derive the eigenvalue spectrum of the local transport covariance: local dimensionality (number of PCA basis functions for a given accuracy) grows linearly with patch area for glossy reflection, with slope 'of the order of the bandwidth of the BRDF or shadowing transfer function', and often as sqrt(area) for simple cast shadows.
- precomputed / learned: Analysis of clustered-PCA precomputed transfer; an adaptive subdivision rule and optimal block size.
- valid under geometry edits: Not addressed (static-scene PRT).
- sharp glossy interreflection: 'Dimensionality increases more rapidly ... for highly specular materials or complex cast shadows, and more slowly for diffuse surfaces.' Direct transport only in the canonical analysis.
- cost claims: Rendering time sub-linear in image resolution; interactive all-frequency relighting at 1024 x 1024.
- limitation: Local, canonical 2D/3D cases under a convolution assumption; no interreflection in the theory.
- relevance: Direct statement of what limits rank: rank needed over a region is about (region size) x (BRDF or visibility bandwidth). A global low-rank term therefore cannot hold a sharp sender-receiver relation over more than a small region; the rank must scale with area times lobe sharpness.
- url: https://cs.columbia.edu/cg/pdfs/1178550610-transportdim.pdf

### Lessig & Fiume, 'On the Effective Dimension of Light Transport' (2010, Computer Graphics Forum 29(4) (EGSR 2010)) [read_primary_source]
- mechanism: Characterises the effective dimension of light transport over a neighbourhood by the spectrum of the spatio-spectral concentration problem (Slepian functions): the count of significant eigenvalues is about the product of the region size and the bandlimit (a Shannon number), with a logarithmic transition band; Slepian functions give optimal local representations.
- precomputed / learned: Nothing; theory with optimal bases for band-limited transport localised to a region.
- valid under geometry edits: Not applicable.
- sharp glossy interreflection: Assumes transport band-limited to SH order L; the dimension is of order (L+1)^2 times the fractional region size, so it grows quadratically with angular sharpness.
- cost claims: None.
- limitation: Band-limit assumption; radially symmetric lighting in the occlusion analysis.
- relevance: Supplies the counting rule (space-bandwidth product) for how many coefficients a region of a spatial-angular table needs; narrowing a lobe by a factor k multiplies the count by about k^2 on the sphere.
- url: https://www.dgp.toronto.edu/public_user/lessig/effective-dimension/

### Many-light matrix methods: Walter et al., 'Lightcuts'; Hasan, Pellacini, Bala, 'Matrix Row-Column Sampling for the Many-Light Problem'; Hasan, Krivanek, Walter, Bala, 'Virtual Spherical Lights for Many-Light Rendering of Glossy Scenes' (2005 / 2007 / 2009, SIGGRAPH 2005 / SIGGRAPH 2007 / SIGGRAPH Asia 2009) [read_primary_source]
- mechanism: Indirect light is a sum over virtual lights, i.e. the column sum of a (receiver sample x light) transfer matrix. Lightcuts clusters lights in a tree with error bounds; row-column sampling reveals the matrix structure from a few rows and columns; virtual spherical lights replace point evaluation by integration over a non-zero solid angle to remove spikes from narrow BRDF lobes and the geometry term.
- precomputed / learned: Nothing persistent across frames; a per-frame low-rank / clustered approximation of the transfer matrix.
- valid under geometry edits: Nothing stored, so nothing is invalidated; cost is paid per frame.
- sharp glossy interreflection: Lightcuts: instant radiosity 'cannot reproduce some types of indirect illumination (e.g., caustics)' and 'has difficulties with short range and/or glossy indirect effects'; clamping is used. VSL: 'Including glossy transport in the matrix increases its numerical rank, so we used more rows (900-1,024) and columns (5,000-15,000)'; the method is consistent but biased and 'blurs some lighting features'.
- cost claims: Lightcuts: strongly sublinear in number of lights. MRCS: seconds per frame on GPU. VSL: 1-4 minutes where path tracing or photon mapping take hours.
- limitation: Clamping removes exactly the sharp, short-range energy; un-clamped point evaluation spikes; the VSL fix is a spatial mollifier whose radius must shrink with light count (memory-limited near 250,000 lights).
- relevance: Empirical confirmation of the rank limit on a transfer matrix: diffuse transport is recoverable from a few rows and columns, glossy transport raises the numerical rank and forces either many more samples or a controlled blur.
- url: https://www.graphics.cornell.edu/~bjw/lightcuts.pdf ; https://www.cs.cornell.edu/~kb/projects/rowcolumnSampling ; https://cgg.mff.cuni.cz/~jaroslav/papers/2009-sigasia-vsl/2009-vsl-paper.pdf

### Hasan, Pellacini, Bala, 'Direct-to-Indirect Transfer for Cinematic Relighting' (2006, ACM TOG 25(3) (SIGGRAPH 2006)) [read_abstract_or_secondary]
- mechanism: Precomputes a large linear operator that maps direct illumination on a cloud of gather samples to multi-bounce indirect illumination at view samples of a fixed viewpoint; the operator is stored as sparse matrices in a wavelet basis built over the unstructured gather cloud and applied on the GPU.
- precomputed / learned: A source-independent direct-to-indirect transfer matrix (hierarchical algorithm plus a photon-mapping variant, under three hours on one processor).
- valid under geometry edits: Nothing: geometry, materials and viewpoint are fixed; only the direct lighting (arbitrary shaders) may change.
- sharp glossy interreflection: Supports 'diffuse and glossy materials' at the final (view-sample) bounce; the gather samples carry diffuse direct illumination (detail beyond the abstract recalled from memory).
- cost claims: 10-20 fps relighting for scenes over 2 million polygons.
- limitation: Fixed view and static scene; sender side is not directional.
- relevance: The canonical example of the exact operator shape in question - linear, source-independent, applied to an encoding of direct light to yield higher-order light - and of its classical scope: static geometry, broad sender radiance.
- url: https://www.cs.cornell.edu/~kb/projects/directToIndirectRelighting

### Potter, Bertone, Schorghofer, Mazarico, 'Fast hierarchical low-rank view factor matrices for thermal irradiance on planetary surfaces' (2022, arXiv:2209.07632) [read_abstract_or_secondary]
- mechanism: Compresses the radiosity view-factor matrix in a hierarchical off-diagonal low-rank-like format: elements are partitioned by quadtree/octree, each block is compressed by a sparse SVD, and the hierarchical matrix is assembled by dynamic programming.
- precomputed / learned: A compressed, source-independent diffuse transport (view factor) matrix reused for many time-varying right-hand sides (moving sun).
- valid under geometry edits: Static geometry assumed; the matrix is reused only across changes of illumination.
- sharp glossy interreflection: Excluded (thermal / diffuse radiosity).
- cost claims: Assembly in quadratic time (comparable to the dense matrix) with much less memory; compressed size and matrix-vector product scale linearly across tested tolerances.
- limitation: Diffuse only; assembly remains quadratic; a search-result summary of the paper notes off-diagonal blocks are not necessarily numerically low rank because of occlusion and directionality.
- relevance: A recent, independent confirmation outside graphics that hierarchical-matrix compression of a source-independent transport operator pays off for diffuse kernels when the operator is reused over many sources - the economic argument for an operator - with nothing said for glossy kernels.
- url: https://arxiv.org/abs/2209.07632

### Loos, Antani, Mitchell, Nowrouzezahrai, Jarosz, Sloan, 'Modular Radiance Transfer' (2011, ACM TOG 30(6) (SIGGRAPH Asia 2011)) [read_abstract_or_secondary]
- mechanism: Stores compact light-transport operators on simple proxy shapes, in low-dimensional subspaces obtained with a prior on incident lighting; the shapes' transport is 'instanced, warped and connected on-the-fly' through interfaces to approximate indirect light in large scenes.
- precomputed / learned: Scene-independent, shape-dependent low-rank transport operators and interface operators.
- valid under geometry edits: The per-shape operators are reused under re-arrangement of the modules; coupling is recomputed at run time from how modules are connected.
- sharp glossy interreflection: Not targeted: 'coarse-scale, distant indirect lighting effects'; low-rank subspaces.
- cost claims: More than 475 fps on high-end GPUs and more than 45 fps on mobile.
- limitation: Approximate, low-frequency, proxy geometry; no sharp directional relations.
- relevance: Precedent for a reusable transport operator attached to a piece of geometry and composed with others after rearrangement, achieved by restricting the transport to low dimension.
- url: https://cs.dartmouth.edu/~wjarosz/publications/loos11modular.html

### Hadadan, Chen, Zwicker, 'Neural Radiosity' and Zheng et al., 'NeLT: Object-oriented Neural Light Transfer' (2021 / 2023, ACM TOG 40(6) (SIGGRAPH Asia 2021) / ACM TOG 42(5)) [read_abstract_or_secondary]
- mechanism: Neural Radiosity represents the full four-dimensional radiance distribution by a network trained to minimise the norm of the rendering-equation residual, noting that classical radiosity bases 'are typically limited to representing isotropic scattering from diffuse surfaces'. NeLT is 'a novel modular neural representation of the dynamic light transportation between an object and the environment', composed explicitly across objects.
- precomputed / learned: Neural Radiosity: a scene- and lighting-specific radiance field. NeLT: per-object neural light-transfer functions.
- valid under geometry edits: Neural Radiosity: nothing (the solution is for one scene state). NeLT: per-object transfer is reused as objects move, with inter-object effects composed at run time (from the abstract and a search-result summary).
- sharp glossy interreflection: Neural Radiosity demonstrates 'a variety of scenes with non-diffuse surfaces'; NeLT's summary lists diffuse and glossy illumination, dynamic shadowing and indirect illumination, with quality 'comparable to the recent advanced real-time denoiser'. Neither abstract makes a claim about sharp sender lobes onto diffuse receivers.
- cost claims: NeLT: interactive rendering for dynamic scenes. Neural Radiosity: efficient synthesis of arbitrary views after the solve.
- limitation: Only abstracts were read; no information here on failure cases, rank or angular resolution.
- relevance: The nearest neural counterparts found from this angle: one replaces the fixed finite-element basis with an adaptive learned one but stores a solution, not an operator; the other stores per-object operators that survive rigid rearrangement.
- url: https://arxiv.org/abs/2105.12319 ; https://history.siggraph.org/?p=164203

## Implications
- Exclusivity of a transport split is a property of path classes, not of estimators: a partition is exact when every path is assigned to exactly one class by the types of its vertices (Heckbert / Veach full-path expressions). The classical exclusive splits are (a) by vertex type - D-to-D transfer including intervening S* chains is stored, the trailing S*E chain is traced from the eye; (b) by the first non-specular vertex from the light - LS+D (caustic) is carried by a light-side pass, the rest by a gather; (c) by path length - k scattering events traced, the remainder stored. Each is exclusive only for a binary D/S material model.
- Glossy lobes are neither D nor S. For them the split is a convention (a roughness threshold or a lobe-wise weight) that must be applied identically by both sides of the partition; otherwise energy is double counted or lost. The literature treats near-specular as 'more problematic' than ideal specular because it escapes the mechanisms built for the latter (Jakob & Marschner).
- Veach's Theorem 8.3: local sampling can generate a path iff it contains a DD edge. A specular vertex forces one neighbour to be produced by sampling its BSDF. A receiver-side gather of a sharp sender violates this in the limit: the receiver's rays are chosen independently of the sender's lobe, so the hit probability scales with the lobe's solid angle. Every classical multi-pass system assigned light -> specular -> diffuse to a pass that samples the reflector's lobe from the light side (extended form factors, light ray tracing, caustic photon maps, particle tracing).
- A stored table read at a path vertex makes that vertex behave as a D-type connection at the table's resolution. In the terms of path-space regularisation, a direction bin is a mollifier of fixed bandwidth: its bias is a blur of the bin width and does not vanish with more samples or more training.
- The light transport operator on radiance is not compact, even for Lambertian scenes (Soler, Molazem, Subr 2022). Finite-rank or non-adaptive finite-basis approximations cannot bound error uniformly over light distributions; only pointwise (per-lighting) convergence is assured. A source-independent operator on a fixed patch x direction basis is in the class the theorem covers. The exitance operator fails to be compact where its kernel is unbounded: concave abutting edges, corners, contact points.
- Rank is bounded below by a space-bandwidth product. Local dimensionality grows linearly with patch area with slope of the order of the BRDF (or shadow) bandwidth (Mahajan et al.), and as region size times bandlimit in the spatio-spectral concentration analysis (Lessig & Fiume). A global low-rank factor can hold broad transport over large regions or sharp transport over small ones, not sharp transport over large ones.
- Required sampling density is predictable from local quantities: the sender BRDF's angular cutoff, surface curvature (which shears angular into spatial frequency and creates caustics), incidence angle, and inverse distance to blockers (Durand et al.). Bandwidth is unbounded at contact. Halving a lobe's width roughly quadruples the angular samples needed on the sphere.
- Importance (the adjoint) determines how much stored error is visible. A diffuse reader integrates the table over a hemisphere and tolerates local error; a near-mirror reader is a point evaluation that assigns all its importance to a few entries. Classical glossy finite-element methods were affordable only because refinement was driven by a specific light and a specific view; an operator that is both source-independent and reader-independent forgoes that adaptivity.
- In the operator factorisation, local scattering is spatially local and independent of scene layout; only the geometric propagation / visibility factor changes under a rigid edit. Classical dynamic methods exploit this by indexing the stored operator by the lines it integrates over (shafts, line-space hierarchy) and invalidating only entries whose line sets the moving object enters or leaves. That machinery was demonstrated for diffuse links; for non-diffuse transport the precedent (Granier & Drettakis) re-shoots particles instead of updating an operator.
- Moving visibility out of the operator (antiradiance) makes pairwise links depend only on the two endpoints and therefore local under rigid edits, but occlusion then has to be reproduced by cancelling directional quantities at the resolution of the angular basis, the unoccluded operator can create energy, and convergence is not guaranteed for mirror-like BRDFs (eigenvalues above one were observed in 2D examples).
- The sharp sender -> receiver relation through an ideal reflector is a finite, discrete set of geometric solutions per endpoint pair (Fermat paths; specular manifolds), with intensity given by wavefront curvature. It is determined by geometry alone and remains computable after any rigid edit without stored state; its cost is root finding per connection and grows with chain length and number of solutions. This is the 'full transport' price for that family against which any stored representation is measured.
- Complexity claims for hierarchical operators (O(n) links, linear-time wavelet methods, O(p) glossy clustering) are stated for smooth kernels or for a fixed directional resolution. They do not include the growth of angular resolution with lobe sharpness; the naive glossy count is O(n^3) element triples, and many-light methods report needing several times more rows and columns once glossy transport enters the matrix.
- Visibility discontinuities and glossy lobes are two distinct sources of high rank. Discontinuity meshing shows shadow edges are located by source x occluder geometry; the compactness analysis states that visibility discontinuities are not what breaks compactness (unbounded kernels and the partial-integral structure are); Mahajan et al. find cast shadows raise dimensionality roughly as the square root of area and glossy lobes linearly. A representation can therefore fail on one and not the other.
- An operator pays off economically when it is reused across many right-hand sides (many lights, many frames) and the kernel compresses; the diffuse precedents (hierarchical radiosity, H-matrix view factors, direct-to-indirect transfer) meet both conditions. No source found here reports a compressed, source-independent operator that carries sharp glossy interreflection and is also reused across geometry edits.
## Closest precedents
- Extended form factors (Wallace et al. 1987; Sillion & Puech 1989): a geometry-only, source-independent linear operator between diffuse patches whose entries already include transfer via ideal specular chains, obtained by tracing through the mirrors per patch pair. Static scenes; ideal specular only.
- Directional finite-element radiance (Immel et al. 1986; Sillion et al. 1991; Christensen et al. 1996/1997): the patch x direction (bins, spherical harmonics or 4D wavelets) representation with a linear transfer operator between such elements - structurally the same object as a patch x direction-bin table driven by a linear operator. Affordable only with light- and view-driven adaptive refinement; sharp lobes moved to ray-traced passes.
- Three-point transport (Aupperle & Hanrahan 1993): radiance indexed by sender-receiver element pairs with hierarchical refinement of element triples, so that angular resolution follows geometry and BRDF instead of a fixed bin set.
- Direct-to-indirect transfer (Hasan, Pellacini, Bala 2006; meshless hierarchical variant by Lehtinen et al. 2008): a precomputed, source-independent linear operator applied to an encoding of direct light to give multi-bounce indirect light, wavelet-compressed. Static geometry; broad sender radiance.
- Line-space hierarchy for moving objects (Drettakis & Sillion 1997) and its glossy successor (Granier & Drettakis 2001): a stored hierarchical diffuse operator kept valid under object motion by invalidating only links whose shafts are crossed; non-diffuse transport handled by a separate incremental particle trace rather than by the operator.
- Implicit visibility / antiradiance (Dachsbacher et al. 2007): a spatial x directional link operator made independent of third-party occluders, so only links touching the moved object change; low-frequency glossy only, stability not guaranteed for sharp BRDFs.
- Modular Radiance Transfer (Loos et al. 2011): low-rank transport operators attached to shapes, reused and connected through interfaces after rearrangement; low-frequency diffuse indirect light only.
- Many-light transfer matrix (Hasan et al. 2007; virtual spherical lights 2009): the receiver x virtual-light matrix as a per-frame, low-rank-sampled operator; documents the rank increase caused by glossy transport and the blur needed to control it.
- Hierarchical low-rank view-factor matrices (Potter et al. 2022): a compressed source-independent diffuse operator reused over many illumination states with linear-scaling application cost; static geometry.
- Neural counterparts seen from this angle (abstract level only): Neural Radiosity (Hadadan et al. 2021) replaces the fixed basis by a network but stores a per-scene solution, not an operator; NeLT (Zheng et al. 2023) stores per-object neural light transfer that is composed across objects in dynamic scenes. Other surveys in this series presumably cover neural precedents in depth; none was examined in full here.
## Unverified / uncertain
- Confidence labels: 'read_primary_source' means the paper text was extracted and the quoted statements were located in it (Veach thesis section 8.3; Soler et al. 2022; Durand et al. 2005; Mahajan et al. 2007; Lessig & Fiume 2010; Dachsbacher et al. 2007; Kaplanyan & Dachsbacher 2013; Jakob & Marschner 2012; Zeltner et al. 2020; Christensen et al. 1996 and 1997; Lightcuts 2005; virtual spherical lights 2009). For those, only the passages relevant to this angle were read, not every section. All other entries rest on an abstract (official or archive page) or on a search-result summary of the abstract; mechanism details beyond the abstract in those entries are recalled from memory and are marked as such where they matter.
- Heckbert 1990: the abstract page does not show the L(S|D)*E notation; the attribution rests on Veach's thesis ('Heckbert's regular expression notation') and course material, not on reading the 1990 paper's body.
- Smits, Arvo, Salesin 1992 (importance-driven radiosity): the archive page had no abstract; the description comes from a search-result summary of the Cornell copy. The Cornell graphics server refused direct connections during this survey, so the Cornell-hosted PDFs for this paper, Arvo-Torrance-Smits 1994 and Sillion et al. 1991 were not read.
- Granier & Drettakis 2001: the Eurographics digital library returned HTTP 403 and HAL returned a bot-protection page; the description is from search-result abstract text only. The same applies to the body of Drettakis & Sillion 1997 (abstract read on the SIGGRAPH history archive; the '2-3 frames per second' figure appears only in a search summary; the diffuse-only scope is inferred from the hierarchical-radiosity framing).
- Lischinski, Tampieri, Greenberg 1992 (discontinuity meshing): bibliographic details and a one-paragraph description from search results; the IEEE page was not fetched.
- Durand et al. 2005: the bandwidth bound (equation 20) was garbled by text extraction; only its stated dependencies (BRDF cutoff, curvature, normal-view angle, distance, harmonic mean of blocker distance) are reported, not the formula. The 'infinite bandwidth at contact points' remark is from the authors' project-page FAQ.
- Aupperle & Hanrahan 1993: the O(n log n) claim is from the Stanford abstract; a second archive copy of the abstract is garbled and a Princeton tech-report copy is a draft.
- Dachsbacher et al. 2007: two different DOIs appear in indexes (10.1145/1276377.1276453 and 10.1145/1239451.1239512); not resolved. The PDF URL given is the INRIA-hosted copy that was downloaded and read.
- NeLT: indexes list ACM TOG 42(5), 2023; the SIGGRAPH history archive lists presentation at SIGGRAPH 2024. Only a four-sentence abstract was read; statements about glossy handling come from a search-result summary.
- Arvo 1995: the abstract page was truncated; the factorisation of the transport operator into a local reflection operator and a geometric field operator (K and G) and the Neumann-series treatment are recalled from memory, not verified here. That Arvo showed specular reflection makes the operator non-compact is taken from Soler et al. 2022's description.
- Potter et al. 2022: the abstract does not discuss occlusion or block rank; the remark that off-diagonal blocks are not necessarily numerically low rank comes from a search-result summary of the paper body.
- Works known or cited but not verified in this survey, and therefore not given entries: Arvo, 'Backward Ray Tracing' (SIGGRAPH 1986 course notes; only secondary descriptions found); George, Sillion, Greenberg, 'Radiosity Redistribution for Dynamic Environments' (1990; search did not return it); Forsyth, Yang, Teo 1994 and Shaw 1997 on hierarchical radiosity for dynamic environments (from memory); Pellegrini 1999 (cited by the antiradiance paper as the earlier visibility-free rendering equation; not read); Dmitriev, Brabec, Myszkowski, Seidel, 'Interactive Global Illumination Using Selective Photon Tracing' (EGWR 2002; search summary only, project page returned 403); Soler & Sillion, 'Hierarchical Instantiation for Radiosity' (2000; search summary only, venue unconfirmed); Chen & Arvo, 'Theory and Application of Specular Path Perturbation' (TOG 19(4), 2000; search summary only); Schroeder & Hanrahan, 'Wavelet Methods for Radiance Computations' (EGWR 1994; search summary only); Stamminger et al., three-point clustering for radiance computations (from memory); Kontkanen et al., 'Wavelet Radiance Transport for Interactive Indirect Lighting' (EGSR 2006; bibliographic entry only); Lehtinen et al. 2008 meshless hierarchical representation (abstract read, no entry); Bagher et al. 2012 bandwidth prediction and Belcour et al. 2013 5D covariance tracing (abstract or search summary only); Georgiev et al. / Hachisuka et al. 2012 vertex connection and merging (not searched successfully); Shirley 1990 'A Ray Tracing Method for Illumination Calculation in Diffuse-Specular Scenes' (citation only); Hanika et al. 2015 manifold next-event estimation (known only through Zeltner et al.'s description).
- No source located in this survey states a quantitative bound of the form 'N receiver-side rays resolve a sender lobe of width w with error e' for interreflection; the nearest results are the Durand et al. sampling criterion (single bounce, image space) and the dimensionality results of Mahajan et al. and Lessig & Fiume (direct transport). Extending them to the light -> sharp reflector -> diffuse family is an inference, not a cited result.
- The statement in the implications that no source reports a compressed source-independent operator carrying sharp glossy interreflection across geometry edits is a statement about what this survey found for this angle, not a proof of absence; precomputed radiance transfer for dynamic scenes and recent neural methods were outside this angle and were not searched systematically.


# ANGLE: Runtime caches and real-time global-illumination competitors (irradiance/radiance caching, photon maps and final gathering, probes, voxel cone tracing, online neural caches, ReSTIR GI/PT, path guiding

### Ward, Rubinstein, Clear - A Ray Tracing Solution for Diffuse Interreflection (irradiance caching) (1988, SIGGRAPH 1988) [read_abstract_or_secondary]
- mechanism: Indirect illuminance is computed by hemisphere Monte Carlo sampling only at sparse surface points chosen during rendering, stored, and interpolated over surfaces in place of an ambient term; deeper bounces use proportionally fewer samples.
- precomputed / learned: Nothing learned. A lazily built, view-driven set of irradiance records (scalar/RGB irradiance at a point), reused across views of a static scene.
- valid under geometry edits: Formally nothing: every record whose hemisphere saw the moved object or its shadow/bounce is wrong, and the classic algorithm has no invalidation test. The common practice reported by Gautron (course notes) is full recomputation per frame, which wastes work and flickers because record positions change.
- sharp glossy interreflection: Excluded from the cache. Only the diffuse indirect term is cached; specular components are traced separately (Jensen 1996 notes that Radiance path-traces glossy surfaces).
- cost claims: Abstract claims only qualitative savings (computation restricted to visible areas, reuse across views, fewer samples at deeper bounces).
- limitation: Irradiance has no directional content, so it cannot feed a glossy receiver; static-scene assumption.
- relevance: Origin of the 'sparse gather + interpolate' cache. Establishes that the classic economical cache works precisely because it discards the directional relation the project needs to keep.
- url: https://radsite.lbl.gov/radiance/papers/sg88/paper.html

### Krivanek, Gautron, Pattanaik, Bouatouch - Radiance Caching for Efficient Global Illumination Computation (2005, IEEE TVCG 11(5)) [read_primary_source]
- mechanism: Extends irradiance caching to cache incoming radiance as (hemi)spherical-harmonic coefficients, with translational gradients and rotation of coefficients; the glossy integral becomes a dot product with BRDF coefficients.
- precomputed / learned: Per-BRDF hemispherical-harmonic tables (order adapts per outgoing direction, max order 10 in the examples) computed before rendering; radiance records (about 3.6 kB each at order 10) created lazily.
- valid under geometry edits: Records are reused across camera motion in a static scene (164 extra records for a second view). No geometry-edit handling in this paper; a moved object invalidates the records that see it.
- sharp glossy interreflection: Explicitly excluded. 'Radiance caching is not used for high-frequency BRDFs since many coefficients would be needed'; those are importance-sampled, and perfect specular uses one deterministic ray. The paper also states: 'radiance caching, as a gathering technique, cannot solve certain types of light transfer phenomena, such as caustics.'
- cost claims: 2.2 GHz Pentium 4: indirect glossy term 33.3 s with 600 records versus 35 s for 12-ray importance sampling at much higher noise; animation 4.9 s/frame versus 35 s/frame (about 7x) in the Cornell box; 15 s versus 27 s/frame on the curved-surface scene.
- limitation: Restricted to low-frequency BRDFs (tested lobes were 'near the limit'); curved glossy receivers reduce reuse and need costly alignment; gathering cannot do caustics.
- relevance: Directly names the failure the project observed: a receiver-side gathering cache with a fixed angular basis cannot represent light -> sharp reflector -> receiver transport, and the directional budget grows with lobe sharpness.
- url: https://cgg.mff.cuni.cz/~jaroslav/papers/tvcg2005/krivanek05radiance_caching.pdf

### Gautron, Bouatouch, Pattanaik - Temporal Radiance Caching (read via Gautron's 2008 irradiance-caching course chapter) (2007, IEEE TVCG 13(5); SIGGRAPH 2008 course notes) [read_abstract_or_secondary]
- mechanism: Each cache record gets a temporal weighting function and temporal gradients; a record is reused over a validity interval estimated from the predicted change of incoming lighting (by reprojection), instead of recomputing the cache per frame.
- precomputed / learned: Nothing learned. Records plus an estimate of next-frame incoming radiance and temporal gradients.
- valid under geometry edits: Records far from the change keep an unbounded lifespan if their environment is static; records affected by motion get a short lifespan (user cap 'such as 5 to 10 frames'). Record positions are kept fixed over time to avoid flicker. Requires predefined animation (future frame known).
- sharp glossy interreflection: Inherits radiance caching's low-frequency restriction; not addressed further in the text read.
- cost claims: Qualitative 'significant speedup' and reduced flicker in the chapter read; no numbers extracted.
- limitation: Thresholds set by trial and error; temporal gradients are first order and smooth fast changes; 'ghost' artifacts from obsolete records; needs knowledge of the next frame.
- relevance: Shows the classical answer to 'what does a rigid move invalidate': a per-record validity time derived from measured lighting change, i.e. a transport-dirty set estimated physically rather than a retrained model.
- url: https://cgg.mff.cuni.cz/~jaroslav/papers/2008-irradiance_caching_class/09-pascal-temporal-notes.pdf

### Jensen - Global Illumination using Photon Maps (caustic map + global map + final gathering) (1996, Eurographics Workshop on Rendering 1996) [read_primary_source]
- mechanism: Two light-traced photon maps: a high-resolution caustics map (photons emitted toward specular objects, stored on diffuse hits) visualised directly, and a low-resolution global map used only as an approximate radiance estimate after a diffuse bounce and to guide sampling in a distribution ray tracer.
- precomputed / learned: Nothing learned. 200,000-500,000 photons per test scene, in a kd-tree; source dependent.
- valid under geometry edits: Not addressed; both maps are scene- and light-specific and are rebuilt for a changed scene (see Dmitriev 2002 for selective update).
- sharp glossy interreflection: Split by path family. Specular and highly glossy reflection seen by the eye: Monte Carlo ray tracing. Caustics (light -> specular -> diffuse): 'We never compute caustics via Monte Carlo sampling since this is almost impossible in most situations' - they come only from the light-traced caustics map. The approximate global-map lookup is used only after a diffuse reflection or beyond a distance threshold; radiance estimates from the map are 'limited to surfaces ranging from Lambertian to slightly glossy'.
- cost claims: Glossy Cornell box about 7x faster than Radiance; photon count affects render time only slightly.
- limitation: Density-estimation blur and bias; memory in photons; no dynamic-scene mechanism.
- relevance: The tested mechanism (exact 1-2 events, table lookup at the second camera vertex) has the same structure as final gathering over a global map, and this paper already states the two rules the failure violated: the approximate store may only be read through a diffuse/wide footprint, and the light -> specular -> diffuse family must be carried by a separate light-side estimator.
- url: https://www.cs.princeton.edu/courses/archive/fall14/cos526/papers/jensen96.pdf

### Dmitriev, Brabec, Myszkowski, Seidel - Interactive Global Illumination Using Selective Photon Tracing (2002, Eurographics Workshop on Rendering 2002) [read_primary_source]
- mechanism: Quasi-Monte Carlo photon tracing with density estimation on a mesh; photons are grouped using the periodicity of the Halton sequence so that a few 'pilot' photons that hit a moving object identify whole groups of coherent 'corrective' photons to re-shoot (negative energy for the old state, positive for the new).
- precomputed / learned: Nothing learned. A photon pool partitioned into groups; illumination bucketed on mesh elements.
- valid under geometry edits: All photon groups whose pilots did not intersect the moved object (or its previous position, the 'scene phantom') are kept; only invalid groups are re-traced, prioritised by pilot hits and view visibility. Detection is explicitly not conservative.
- sharp glossy interreflection: Out of scope: 'In this paper we focus on global illumination for diffuse environments'; the authors note hierarchical radiosity poorly supports glossy/specular transfer.
- cost claims: 8 fps (ROOM) and 1.1 fps (HOUSE) on a dual 1.7 GHz P4 with a GeForce3; update converges progressively over iterations.
- limitation: Mesh-based reconstruction of indirect light (coarse); probabilistic invalidation can miss affected photons; diffuse only.
- relevance: A purely physical precedent for selective recomputation after a rigid move: a geometry-indexed partition of light paths, a cheap probe that identifies the transport-dirty subset, and exact reuse of the rest - without any learned state.
- url: https://www.graphicon.ru/oldgr/en/publications/text/spt.pdf

### Yang, Ouyang - Real-Time Ray Traced Caustics (adaptive anisotropic photon scattering) (2021, Ray Tracing Gems II, chapter 30) [read_abstract_or_secondary]
- mechanism: Photons are traced from the light through metallic / transparent surfaces each frame, splatted with anisotropic footprints from photon differentials, and a negative-feedback loop redistributes photon density toward regions near the camera or temporally unstable in screen space.
- precomputed / learned: Nothing precomputed or learned; per-frame photon buffer plus light-space feedback buffers.
- valid under geometry edits: Stated as fully dynamic with no preprocessing: the estimator is recomputed each frame; only the feedback (density) state carries over and re-adapts.
- sharp glossy interreflection: This is its sole purpose: caustics from highly polished metallic and transparent objects after multiple bounces are produced light-side, then shaded on the receiver (GGX shading pass using the receiver G-buffer).
- cost claims: Described as low cost; a per-GPU timing table exists (RTX 2060 to 3090, 1080p/1440p) but I could not extract it reliably.
- limitation: Screen-space driven (targets visible receivers); denoiser dependent; I read only the introduction and parts of the method.
- relevance: Shows what real-time practice does for the exact family that broke the learned operator: emit from the light side with adaptive density, rather than gather from the receiver side.
- url: https://developer.download.nvidia.com/ray-tracing-gems/rtg2-chapter30-preprint.pdf

### McGuire, Mara, Nowrouzezahrai, Luebke - Real-Time Global Illumination using Precomputed Light Field Probes (2017, I3D 2017) [read_primary_source]
- mechanism: A grid of octahedral probes stores radiance, normals and radial distance (plus filtered irradiance and distance moments); world-space rays are marched through the probes' depth maps in a shader to fetch radiance with correct visibility, for diffuse and glossy gathers.
- precomputed / learned: Precomputed probes (e.g. a 4^3 grid of 1024x1024-texel probes), 8 bytes/texel uncompressed; nothing learned.
- valid under geometry edits: Static environments only; dynamic objects can receive but the probes are not updated. A geometry edit requires re-rendering the probes.
- sharp glossy interreflection: Handled by ray marching into radiance probes, with a stated resolution rule: to avoid aliasing on perfect mirrors, probe texel solid angle must not exceed the pixel's solid angle at the probe centre. Second- and higher-order mirror reflections are 'viewpoint biased according to probe locations'.
- cost claims: Living-room global illumination in 4.7 ms.
- limitation: Precomputed and static; higher-order specular parallax error; probe resolution scales with the lowest roughness in the scene.
- relevance: A physically computed cache that does keep sharp directional relations - at the price of probe resolution tied to mirror sharpness and no validity under geometry edits.
- url: https://research.nvidia.com/publication/2017-02_real-time-global-illumination-using-precomputed-light-field-probes

### Majercik, Guertin, Nowrouzezahrai, McGuire - Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields (DDGI); and Majercik et al. - Scaling Probe-Based Real-Time Dynamic Global Illumination for Production (2021, arXiv 2009.10796) (2019 / 2021, JCGT 8(2); JCGT 10(2)) [read_primary_source]
- mechanism: A probe grid stores low-resolution octahedral irradiance and depth moments; every frame each probe traces a few rays against the current scene and blends results with hysteresis; multi-bounce comes from shading probe rays with the previous frame's probes. The 2021 paper adds probe states (sleeping/awake), adaptive hysteresis and cascades.
- precomputed / learned: Nothing precomputed or learned; 8x8-texel irradiance and depth per probe, updated online.
- valid under geometry edits: Nothing is declared valid; all values relax toward the new solution by exponential blending (hysteresis 0.85-0.98, i.e. a time constant of roughly 7-50 frames by my arithmetic), with a further one-frame-per-bounce lag. 2021: expanded bounding boxes of dynamic objects wake sleeping probes; probes are not repositioned around dynamic objects; scene-change events temporarily lower hysteresis for all probes.
- sharp glossy interreflection: Not cached. First-order glossy/specular reflection is ray traced per pixel; at second-hit surfaces the glossy energy is folded into the matte term (2019), and second-order reflections are forced to maximum roughness (2021).
- cost claims: 6 ms/frame versus 1 min/frame path traced (RTX 2080 Ti, 1080p). Table 2: indirect diffuse total 2.6 ms for 32x8x32 probes x 64 rays, plus 2.4 ms for brute-force ray-traced glossy. 2021: probe sleeping saves 30-50 % on average.
- limitation: Diffuse-only cache; temporal lag and ghosting with small bright lights; leak handling by heuristics; hysteresis heuristics act globally.
- relevance: The canonical physically computed cache that survives rigid edits without retraining. It delivers the broad-transport saving at 2-3 ms with tens of frames of lag, and its authors chose to exclude sharp glossy transport from the cache altogether.
- url: https://jcgt.org/published/0008/02/01/paper-lowres.pdf

### Rodriguez, Leimkuhler, Prakash, Wyman, Shirley, Drettakis - Glossy Probe Reprojection for Interactive Global Illumination (2020, ACM TOG 39(6) (SIGGRAPH Asia 2020)) [read_abstract_or_secondary]
- mechanism: Paths with multiple glossy interactions are stored in precomputed light probes with an adaptive parameterisation (more resolution for shinier surfaces and complex geometry) and reprojected per view using reflection estimation and a rasterisation-based search; probes are precomputed at lower roughness and an adaptive bilateral filter restores the true roughness at runtime.
- precomputed / learned: Precomputed glossy probes plus light maps for diffuse; nothing learned.
- valid under geometry edits: Static scenes with opaque objects only; a geometry edit requires recomputing the probes.
- sharp glossy interreflection: Central topic: multi-glossy-bounce paths are baked into probes; parallax at occlusion boundaries would sharpen reflections, addressed by the two-step roughness split.
- cost claims: Interactive rates 'fitting in the memory, bandwidth, and computation constraints of current hardware' (abstract only).
- limitation: Static geometry and lighting; memory grows with shininess.
- relevance: Evidence that a cache holding sharp multi-glossy interreflection has been built only as a static precomputation with resolution that adapts to shininess - there is no dynamic-geometry counterpart in this survey.
- url: https://research.nvidia.com/publication/2020-11_glossy-probe-reprojection-interactive-global-illumination

### Crassin, Neyret, Sainz, Green, Eisemann - Interactive Indirect Illumination Using Voxel Cone Tracing (2011, Computer Graphics Forum 30(7) (Pacific Graphics)) [read_primary_source]
- mechanism: Direct light is injected into the leaves of a sparse voxel octree storing pre-filtered geometry, incoming radiance and Gaussian-lobe normal distributions; final gathering is approximated by tracing a few cones through the mip hierarchy (several diffuse cones plus one specular cone whose aperture follows the specular exponent).
- precomputed / learned: Static octree built once (about 280 ms); nothing learned.
- valid under geometry edits: The static part of the octree persists (time-stamped); animated objects are re-voxelised every frame (about 5.5 ms/frame for a 16K-triangle mesh) and radiance is re-injected and re-filtered each frame.
- sharp glossy interreflection: Glossy handled as one pre-filtered cone (example 10 degree aperture); storing incoming rather than outgoing radiance permits glossy senders, but voxel averaging limits sharpness and causes leaking (the 'two red-green wall' problem). Mirror-sharp interreflection is not claimed.
- cost claims: 25-70 fps on a GTX 480; 32 ms versus 37 ms for light propagation volumes at 512^2; roughly 1024 MB video memory, 'especially due to the support for indirect specularity'.
- limitation: Two bounces; visibility is approximate (opacity averaging, leaks); memory.
- relevance: A geometry-indexed, pre-filtered cache where the angular footprint is explicit (cone aperture). It shows that narrowing the cone raises both cost (16.6 ms at 10 degrees versus 3.9 ms at 60 degrees for 3 AO cones) and memory.
- url: https://research.nvidia.com/labs/rtr/publication/crassin2011givoxels/

### Wright, Narkowicz, Kelly - Lumen: Real-time Global Illumination in Unreal Engine 5 (2022, SIGGRAPH 2022 course, Advances in Real-Time Rendering in Games) [read_primary_source]
- mechanism: Per-mesh 'cards' form a surface cache (material + direct + indirect lighting in an atlas) that secondary rays read; the first bounce is solved by a screen-space probe final gather (8x8 directional resolution) backed by a world-space radiance cache (32x32); multi-bounce comes from surface-cache feedback across frames; reflections have their own pipeline.
- precomputed / learned: Cards and mesh distance fields are generated at mesh import (about 0.2 ms and 0.6 ms for a 1.5M-triangle mesh); lighting is entirely runtime; nothing learned.
- valid under geometry edits: Object-local data (cards, mesh distance fields) move with the object. Global distance field bricks are updated only where modified; surface-cache lighting is refreshed under a fixed texel budget per frame (1024^2 direct, 512^2 indirect) - 'fixed update cost, variable lighting latency'; history rejection gives 'slow reaction to lighting changes, which shows up as ghosting behind moving objects'. Skeletal meshes have no surface-cache parameterisation.
- sharp glossy interreflection: Banded by roughness: 0.4-1.0 resample the screen radiance cache (no reflection rays, 50-70 % cost reduction); 0.3-0.4 short ray then world radiance cache (32x32); below 0.3 dedicated reflection rays. Mirror reflections need hardware ray tracing and hit lighting; for mirror, cards would need to match screen pixel density; distance fields 'aren't perfect for mirror reflections'.
- cost claims: 1080p internal: about 6.4 ms (High, 60 fps target) and 11.3 ms (Epic, 30 fps target) total on console for The Matrix Awakens, reflections 2.3 / 4.6 ms of that; surface-cache multi-bounce below 1/16 of Lumen's budget. The offline lighting build 'has a hundred thousand times more processing time'.
- limitation: Coarse voxel lighting in the surface cache; latency; skinned meshes; material response in the surface cache is approximated ('approximate specular and subsurface' by modifying albedo).
- relevance: The most complete shipped answer to 'cache what is broad, trace what is sharp'. It quantifies where a cache with fixed angular resolution stops being acceptable (roughness about 0.3-0.4) and shows object-local state surviving rigid motion while radiance values are amortised.
- url: https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf

### Halen, Brinck, Hayward, Bei - Global Illumination Based on Surfels (GIBS) (2021, SIGGRAPH 2021 course, Advances in Real-Time Rendering in Games) [read_primary_source]
- mechanism: Surfels are spawned from the G-buffer onto visible geometry, persist, and accumulate ray-traced diffuse irradiance with a variance-tracking moving average; each surfel also stores depth moments for a Chebyshev visibility test.
- precomputed / learned: Nothing precomputed or learned ('no need for time consuming bakes').
- valid under geometry edits: Surfel placement survives motion: each surfel stores a local position relative to a transform ID (rigid object or dominant skinning bone) and follows it every frame. Irradiance values are not declared valid; when a surfel sees its mean change it requests more rays and re-converges, then returns to an 'almost dormant' ray count.
- sharp glossy interreflection: Out of scope: the method computes indirect diffuse only ('the BRDF we are dealing with is essentially the cosine lobe').
- cost claims: About 7 ms total in a large-draw-distance scene; about 11 ms while converging in a 1,400-light stress case with a 1M ray budget; near-converged after 15 frames with 4-sample lightcut sampling; one example 'converged to acceptable quality in about five seconds'. Global ray budget is artist-set.
- limitation: Diffuse only; convergence time in many-light scenes; one-bone skinning approximation.
- relevance: Shows that 'state attached to the moving object stays valid under rigid edits' is cheap for the parameterisation, and that the radiance stored on it is re-estimated physically with a variance-driven budget - a non-learned selective recompute.
- url: https://www.advances.realtimerendering.com/s2021/SIGGRAPH%20Advances%202021%20-%20Surfel%20GI.pdf

### Boisse, Meunier, de Dinechin, Bartels, Veselov, Eto, Harada - GI-1.0: A Fast Scalable Two-Level Radiance Caching Scheme for Real-Time Global Illumination (2022 (arXiv 2023), AMD technical report 22-10-9831; arXiv 2310.19855) [read_primary_source]
- mechanism: Level 1: screen probes (8x8 octahedral, one per 8x8 tile) store incoming radiance at primary vertices; level 2: a world-space spatial-hash cache stores outgoing radiance at secondary vertices; probe rays terminate in the hash cache; reservoir light sampling feeds it.
- precomputed / learned: No preprocessing, nothing learned.
- valid under geometry edits: Probes are reprojected with motion vectors and reused only if plane distance and normal agree; holes get reallocated rays; a shadow-preserving hysteresis adapts blend strength to the difference between new and reprojected radiance; hash cells decay when unaccessed. No latency figure is given.
- sharp glossy interreflection: Not delivered: glossy reflections are 'still in progress'; the plan is to sample GGX directions into the caches and to cast a ray for low roughness.
- cost claims: 1.9-3.4 ms total for direct + indirect at 1080p with 1/4 sample per pixel (RX 6900 XT / RTX 3080); 2.4 ms versus 2.6 ms for Lumen in the authors' internal test.
- limitation: Infinite-bounce feedback only where the contributing surface was visible in the previous frame; light-grid clamping bias; no glossy.
- relevance: A recent, fully physical two-level cache with exactly the 'few exact events, then table lookup' structure of the tested mechanism. It reaches about 2-3 ms without learning, and still has no answer for glossy.
- url: https://ar5iv.labs.arxiv.org/html/2310.19855

### Binder, Fricke, Keller - Massively Parallel Path Space Filtering (2019 (rev. 2021), arXiv 1902.05942; MCQMC 2020) [read_primary_source]
- mechanism: Contributions of path vertices that fall in the same jittered, quantised hash cell (position, normal, optionally more) are averaged, replacing neighbourhood searches by one hash lookup; works at non-primary vertices.
- precomputed / learned: Nothing; a hash table of per-cell averages maintained across frames.
- valid under geometry edits: For static scenes averages converge; for dynamic environments two sets of averages are combined with an exponential moving average, and the paper shows that this blurs moving shadow boundaries while full temporal integration averages them out.
- sharp glossy interreflection: 'On highly reflective surfaces ... filtering is pointless. Therefore, vertices on such surfaces should not be selected in the first place. On glossy surfaces, however, filtering may reduce variance efficiently, again at the cost of a certain bias.' Layered materials are filtered per layer with different keys.
- cost claims: About 1.5 ms overhead at HD; 1 path per pixel plus filtering 'very closely approximates' a 1024-path reference in the shown scene.
- limitation: Biased; quantisation and light leaking; temporal adaptation versus integration trade-off.
- relevance: The physical analogue of a patch-indexed radiance table. Its authors state the selection rule for which vertices may read the table (not specular ones), and it costs about 1.5 ms.
- url: https://arxiv.org/abs/1902.05942

### Pantaleoni - Online Path Sampling Control with Progressive Spatio-Temporal Filtering (2020, arXiv 2005.07547) [read_primary_source]
- mechanism: A spatio-directional hash (position x outgoing direction for outgoing radiance; position x incoming direction for incoming radiance) is filled by filtering path samples with temporal feedback, described as a hybrid of progressive radiosity and path tracing; the approximation is used for importance sampling, control variates or a biased estimator.
- precomputed / learned: Nothing offline; online-estimated hash cells (48 bytes; 4M-entry table), optionally k-d tree or Gaussian-mixture guiding records.
- valid under geometry edits: Per-cell counters blend linear and exponential averaging with a maximum temporal window; counters 'can be reset locally, on demand, or globally after large structural changes'; the solver lags by about one frame per feedback step.
- sharp glossy interreflection: The directional hash key exists specifically to encode 'directional variations like those due to glossy reflections', and the author reports reproducing glossy interreflections; caustics are stated as not handled by next-event estimation and needing other techniques.
- cost claims: RTX 2080 Ti, 1080p, 1 spp: +2-4 ms on 16-20 ms of path tracing for the control-variate / biased estimators; guiding adds about 30-70 ms; k-d tree control variate about 13x faster than path tracing with NEE at equal variance.
- limitation: Bias constant chosen by hand; guiding overhead; not compared with offline guiding.
- relevance: The closest non-learned precedent to the tested table: a (patch x outgoing-direction) radiance store built progressively by physical transport with feedback. It attains glossy directionality by hashing direction at fixed resolution and updates under edits by counter resets.
- url: https://ar5iv.labs.arxiv.org/html/2005.07547

### NVIDIA SHaRC - Spatially Hashed Radiance Cache (integration guide) (2023-2025, NVIDIA RTXGI SDK documentation) [read_primary_source]
- mechanism: A world-space hash grid of voxels stores accumulated radiance; an update pass path-traces from a sparse subset of pixels (e.g. one per 5x5 block, about 4 % of paths) and back-propagates; rendering paths query the cache at every non-primary hit and terminate on success.
- precomputed / learned: Nothing; hash entries (8 B) plus accumulation and resolved buffers (16 B each); about 160 MiB for 2^22 entries, 256 MiB with SH directional encoding.
- valid under geometry edits: Values are blended over a maximum accumulated-frame count (higher = better quality, slower response); entries expire after a stale-frame limit; a 'responsive lighting' mode trades long-term accumulation for reaction speed. The guide does not discuss dynamic geometry specifically.
- sharp glossy interreflection: Guarded lookup: 'unlike diffuse lobes, specular ones should be treated with care' - the cache may be used only when the lobe's estimated cone spread, 2 x ray length x sqrt(0.5 a^2 / (1 - a^2)), exceeds the voxel size; otherwise the path continues. Optional SH encoding prevents highlights leaking to other view directions.
- cost claims: No frame-time figures in the guide; occupancy 10-20 % for a static camera.
- limitation: Hash collisions; artifacts on short segments and glossy surfaces unless the spread checks are applied.
- relevance: States, as an engineering rule, the footprint condition under which a cached value may substitute for transport. A lookup at a near-mirror vertex fails this test by construction.
- url: https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md

### Muller, Rousselle, Novak, Keller - Real-time Neural Radiance Caching for Path Tracing (NRC) (2021, ACM TOG 40(4) (SIGGRAPH 2021)) [read_primary_source]
- mechanism: A small MLP (5 hidden layers x 64) maps position, direction, normal, roughness and reflectances to scattered radiance; paths are terminated into the cache once their area spread exceeds a threshold relative to the primary footprint (c = 0.01); the network is trained online each frame on self-bootstrapped targets.
- precomputed / learned: No pretraining ('pretrained networks would struggle to generalize to new dynamic scenes'); weights learned online from 65,536 records per frame, 4 gradient steps, EMA of weights 0.99.
- valid under geometry edits: Nothing is retrained offline, but the weights are continuously re-fitted; summarised from the paper: usable after about 8 frames and overall colours correct after about 64 frames from scratch. Old content is forgotten at the learning rate.
- sharp glossy interreflection: Roughness and direction are inputs, and the spread heuristic lengthens paths after sharp bounces so that 'cache artifacts are observed through rough reflections and therefore averaged out'. Sharp detail not carried by the inputs, 'such as a sharp contact shadow or caustic', is poorly captured.
- cost claims: About 2.6 ms at 1080p (query 1.59 ms + training 1.11 ms on average); lower bias but higher variance and cost than DDGI.
- limitation: Low-frequency scintillation; self-training bias; slow or failed learning of shadows and caustics with small networks or noisy targets.
- relevance: The main learned competitor that needs no retraining step under rigid edits. It is source-dependent and scene-specific (radiance, not an operator), and it too only reads the cache through a wide footprint.
- url: https://arxiv.org/abs/2106.12372

### Dereviannykh, Klepikov, Hanika, Dachsbacher - Neural Two-Level Monte Carlo Real-Time Rendering (Neural Incident Radiance Cache, NIRC) (2025, Computer Graphics Forum (Eurographics 2025); arXiv 2412.04634) [read_primary_source]
- mechanism: An online-trained network predicts incident radiance at a vertex (queried many times, e.g. 25 neural samples); a two-level Monte Carlo estimator adds an unbiased path-traced estimate of the residual between cache and truth; a balanced termination heuristic decides where the cache may end a path.
- precomputed / learned: Online-trained tiny MLP with multi-resolution hash features; no offline training.
- valid under geometry edits: Continuously re-fitted; in the dynamic Bistro test the method 'starts to outperform the standard Monte Carlo approach with the same compute budget after approximately 20 frames'.
- sharp glossy interreflection: Operates on up to 3 non-specular vertices; the termination heuristic is introduced 'acknowledging its limitations, particularly with glossy details'; the residual estimator keeps the result unbiased where the cache is wrong.
- cost claims: One cache evaluation takes 2-25x less compute than a path-tracing sample.
- limitation: Residual variance where the cache is poor; warm-up frames after changes.
- relevance: Shows a way existing work bounds the damage of a wrong cached value at a sharp receiver: the cache is a control variate with a path-traced residual, so localised cache error appears as noise rather than as a biased reflection.
- url: https://arxiv.org/abs/2412.04634

### Ouyang, Liu, Kettunen, Pharr, Pantaleoni - ReSTIR GI: Path Resampling for Real-Time Path Tracing (2021, Computer Graphics Forum 40(8) (HPG 2021)) [read_primary_source]
- mechanism: Each pixel keeps a reservoir holding one indirect sample (visible point, sample point, outgoing radiance at the sample point, random seed); samples are resampled across pixels and frames with a reconnection Jacobian.
- precomputed / learned: Nothing; screen-space reservoir buffers of fixed size.
- valid under geometry edits: Stored outgoing radiance becomes stale: 'temporal reuse may cause bias if the outgoing radiance values stored at the sample points become inaccurate ... the result may be a noticeable lag'. Mitigations: reservoir count clamps (30 temporal, 500 spatial), re-tracing stored samples for validation every 6 frames, and a shadow ray to the sample point.
- sharp glossy interreflection: Stated limitation: 'not effective with multi-bounce global illumination with highly glossy surfaces. Not only is our assumption of Lambertian scattering at sampled points inaccurate in that case, but spatial sample reuse will be much less effective since a sample point that makes a large contribution at one pixel may be outside the peak specular lobe of another.' Glossy receivers also worsen bias.
- cost claims: RTX 3090, 1080p: 1-spp two-bounce path tracing 8.0 ms versus ReSTIR GI 8.9 ms (biased) / 9.6 ms (unbiased); MSE improvement 9.3x-166x over path tracing at one sample per pixel. The authors note games typically allow less than 2 ms for GI.
- limitation: Screen-space storage (disocclusion noise, perfectly specular objects); Lambertian assumption at the sample point; temporal lag.
- relevance: Its reconnection step is the same geometric operation as gathering a sender through a receiver-side ray; the authors give the same diagnosis as the project's failure - a sample valid for one receiver falls outside the narrow lobe for another.
- url: https://d1qx31qr3h6wln.cloudfront.net/publications/ReSTIR%20GI.pdf

### Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman - Generalized Resampled Importance Sampling: Foundations of ReSTIR (ReSTIR PT); with Lin, Kettunen, Wyman - ReSTIR PT Enhanced (2026) (2022 / 2026, ACM TOG 41(4) (SIGGRAPH 2022); Proc. ACM CGIT 9(1)) [read_primary_source]
- mechanism: Full paths are reused between pixels and frames through shift mappings: reconnection where the vertices involved are rough enough, random-number replay through glossy/specular vertices (hybrid shift), with lobe-specific connectability. The 2026 paper replaces fixed roughness/distance thresholds by ray-footprint criteria and halves spatial-reuse cost.
- precomputed / learned: Nothing; 88-byte reservoirs per pixel (reconnection vertex + replay seed).
- valid under geometry edits: Nothing is assumed valid: reused paths are re-evaluated (shifted) in the current scene, so the estimator stays unbiased/consistent; the temporal confidence cap 'avoids slow sample turnover in dynamic scenes'.
- sharp glossy interreflection: Explicitly addressed. Reconnection is allowed only when the lobes at the three vertices exceed a GGX roughness of about 0.2 and the segment is long enough; otherwise the path is re-traced by random replay. 'Reconnection shifts ... often fail to reconnect on near-delta BRDFs.' Contact caustics reuse well with the hybrid shift; caustics from distant small highlights perform poorly with it; sharp caustics can still be undersampled ('noise, streaks, or splotches'). A manifold-exploration shift is named as future work.
- cost claims: RTX 3090, 1080p: reconnection 5 ms versus hybrid 12 ms in the glossy Cornell box; full prototype 'originally 40-80 ms per frame'; the enhanced version is 2-3x faster (37.1 -> 12.6 ms in its Figure 1, from a search snippet).
- limitation: Screen-space undersampling of tiny geometry / very high-frequency light; correlation artifacts in glossy scenes with hard lighting; cost.
- relevance: The strongest current evidence on how sharp directional relations are reused without bias: not by storing radiance, but by storing the path and re-deriving the constrained vertex (replay), with an explicit roughness/footprint gate deciding where a stored connection is admissible.
- url: https://graphics.cs.utah.edu/research/projects/gris/sig22_GRIS.pdf

### Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman - Conditional Resampled Importance Sampling and ReSTIR (2023, SIGGRAPH Asia 2023) [read_abstract_or_secondary]
- mechanism: Extends GRIS to conditional path spaces so that path suffixes can be reused conditioned on non-reused prefixes; the prototype is a ReSTIR PT with a final-gather pass that postpones reuse by at least one bounce.
- precomputed / learned: Nothing; suffix reservoirs.
- valid under geometry edits: As ReSTIR PT: reused suffixes are re-evaluated; no persistent validity claim.
- sharp glossy interreflection: Not stated in the abstract; the motivation is that, 'as in photon mapping, such a final gather reduces blotchy artifacts from sample correlation'.
- cost claims: None in the abstract; the code repository describes the prototype as very costly (recommends an RTX 4090).
- limitation: Cost; abstract-level reading only.
- relevance: Confirms that 'exact prefix + reused suffix behind a final gather' is the current frontier form of the tested mechanism's structure, done with resampled paths instead of a table.
- url: https://research.nvidia.com/labs/rtr/publication/kettunen2023conditional/

### Muller, Gross, Novak - Practical Path Guiding for Efficient Light-Transport Simulation (SD-tree); with Vorba et al. 2014 - On-line Learning of Parametric Mixture Models for Light Transport Simulation (2017 / 2014, Computer Graphics Forum 36(4) (EGSR 2017); ACM TOG 33(4)) [read_abstract_or_secondary]
- mechanism: An approximation of the incident radiance field is learned online (spatial binary tree x directional quadtree; or Gaussian mixtures trained from a stream of particles) and used only as a sampling density for scattering directions, with automatic budgeting between training and rendering.
- precomputed / learned: Per-scene, per-lighting incident-radiance distributions learned during rendering (Vorba: e.g. 30 training passes within a one-hour render).
- valid under geometry edits: Not addressed (offline, static scenes); the learned distributions are scene- and light-specific and would have to be re-learned.
- sharp glossy interreflection: Demonstrated on 'complex specular-glossy light transport'; because the learned field only steers sampling, errors in it raise variance but do not bias the image.
- cost claims: No fixed numbers in the abstracts; training cost is said to be amortised by better sampling.
- limitation: Directional resolution of the guide limits how well narrow features are found; memory; learning time.
- relevance: Establishes the alternative role for a learned or cached transport representation - as a proposal distribution rather than as the answer - under which a localised error at a mirror cannot appear as a wrong reflection.
- url: https://www.jannovak.info/publications/PathGuide/index.html

### Dittebrandt, Hanika, Dachsbacher - Temporal Sample Reuse for Next Event Estimation and Path Guiding for Real-Time Path Tracing (2020, Eurographics Symposium on Rendering 2020 (DL track)) [read_primary_source]
- mechanism: A world-space octree whose leaves hold a visibility light cache (lights with unoccluded samples) and a compressed directional quadtree (32-128 bits) for guiding; both, and the octree itself, are adapted every frame from the previous frame's path samples.
- precomputed / learned: No precomputation ('must not require precomputation exceeding the budget within one frame'); per-leaf sampling structures adapted online.
- valid under geometry edits: Structures are incrementally adapted from last frame's samples; for dynamic surface points whose current leaf is empty a fallback is used; an exploration technique keeps the estimator unbiased while the guide is stale.
- sharp glossy interreflection: Not specifically; the directional quadtree is most useful with strong contribution from a specific direction (e.g. the sun) and gives no benefit under uniform illumination.
- cost claims: Update and rebuild of all structures about 1 ms; about 6 ms added to path tracing at 1080p; MSE reduced by at least an order of magnitude in many situations.
- limitation: Very low directional resolution; scene-dependent benefit.
- relevance: A real-time, dynamic-scene instance of guiding: a stale directional cache costs variance only, and re-adaptation takes one frame of samples and about 1 ms.
- url: https://cg.ivd.kit.edu/publications/2020/tsr_nee_pg_rtpt/paper.pdf

### Keller - Instant Radiosity; Kollig, Keller - Illumination in the Presence of Weak Singularities (bias compensation) (1997 / 2004 (publ. 2006), SIGGRAPH 1997; MCQMC 2004) [read_abstract_or_secondary]
- mechanism: Light subpaths are traced from the sources and their vertices become virtual point lights (VPLs); indirect light is the sum of direct lighting from VPLs with shadow maps. Contributions are clamped to remove 1/r^2 and narrow-lobe spikes; Kollig and Keller restore the clamped energy with an unbiased path-traced term.
- precomputed / learned: Nothing learned; a set of VPLs plus one shadow map each, specific to the light and scene.
- valid under geometry edits: Not addressed in the originals; any VPL whose light path or shadow map sees the mover is invalid (see Laine 2007).
- sharp glossy interreflection: Classical VPLs are diffuse-only and clamped; as characterised by Hasan et al. 2009, 'light bouncing off glossy surfaces is ignored' and clamping loses glossy energy; the compensation term in glossy scenes can cost nearly as much as path tracing the whole image (3.5 hours in the VSL paper's example versus 34 s for the clamped render).
- cost claims: See above (secondary).
- limitation: Clamping bias; compensation degenerates to path tracing when the missing energy is significant.
- relevance: The many-light formulation is the sender-side dual of receiver-side gathering and fails at the same place: a narrow lobe makes single connections spiky, and removing the spikes removes real energy that is expensive to restore.
- url: https://cgg.mff.cuni.cz/~jaroslav/papers/2009-sigasia-vsl/2009-vsl-paper.pdf

### Laine, Saransaari, Kontkanen, Lehtinen, Aila - Incremental Instant Radiosity for Real-Time Indirect Illumination (2007, Eurographics Symposium on Rendering 2007) [read_primary_source]
- mechanism: Single-bounce indirect light from a few hundred VPLs whose paraboloid shadow maps are reused across frames; only a few VPLs (4-8 of 256) are replaced per frame to keep a good distribution as the primary light moves.
- precomputed / learned: No precomputation; persistent VPLs and shadow maps.
- valid under geometry edits: Validity relies on static bouncing geometry: 'We require that the geometry used for bouncing off indirect illumination remains static. Dynamic objects cannot [bounce light]' but can receive it; VPL validity is checked by a shadow ray against the static scene only.
- sharp glossy interreflection: Diffuse only; the authors note that with highly specular BRDFs the contributions of individual VPLs would become more [uneven], i.e. the approach degrades for sharp lobes.
- cost claims: 49 fps at 1024x768 with 256 VPLs on an 80K-triangle scene; 31 fps with 1024 VPLs versus 2 fps for the comparison method; up to 7x faster.
- limitation: 'Perhaps the biggest limitation ... indirect illumination can bounce from static geometry only'; single bounce; not view-driven.
- relevance: A clean example of cache reuse whose validity condition is exactly 'the sender geometry did not move' - the asymmetry (movers receive but do not send) is the cheap half of dynamic GI.
- url: https://research.nvidia.com/publication/2007-06_incremental-instant-radiosity-real-time-indirect-illumination

### Walter, Fernandez, Arbree, Bala, Donikian, Greenberg - Lightcuts: A Scalable Approach to Illumination (2005, ACM TOG 24(3) (SIGGRAPH 2005)) [read_abstract_or_secondary]
- mechanism: Point lights (including VPLs) are clustered in a binary light tree; for each shading point a cut through the tree is chosen so that the bounded approximation error stays below a perceptual threshold.
- precomputed / learned: The light tree, built per frame/scene; nothing learned.
- valid under geometry edits: Not addressed (offline); the tree depends on light and VPL positions and would be rebuilt.
- sharp glossy interreflection: The abstract claims 'non-diffuse materials' at the receiver; cluster error bounds depend on the receiver BRDF, so sharper lobes force deeper cuts. Senders remain VPLs.
- cost claims: Strongly sublinear: hundreds of thousands of point lights approximated with a few hundred shadow rays, tens with reconstruction cuts.
- limitation: Inherits VPL clamping and diffuse-sender limits (per Hasan 2009); abstract-level reading only.
- relevance: A physically computed, error-bounded hierarchical approximation of the receiver <- many senders sum; it shows the cost of the gather can be made sublinear in senders without learning, but not that narrow sender lobes are resolved.
- url: https://www.cs.cornell.edu/~kb/projects/lightcuts

### Hasan, Pellacini, Bala - Matrix Row-Column Sampling for the Many-Light Problem (2007, ACM TOG 26(3) (SIGGRAPH 2007)) [read_abstract_or_secondary]
- mechanism: Transport is viewed as a large matrix of sample-light interactions whose column sum is the image; a small number of full rows and columns, rendered on the GPU with shadow maps, reveals the matrix structure and yields a reduced set of representative lights.
- precomputed / learned: Nothing learned; a reduced (low-rank-like) approximation of the transfer matrix computed per frame from sampled rows and columns.
- valid under geometry edits: Not addressed; the sampled rows/columns contain visibility and would be recomputed.
- sharp glossy interreflection: Abstract claims 'arbitrary shaders'; glossy sender transport still depends on the VPL formulation (addressed later by VSLs, which reuse this algorithm).
- cost claims: Light transfer computed 'within a few seconds' for scenes with indirect and environment illumination.
- limitation: Abstract-level reading; quality depends on the matrix being close to low rank.
- relevance: A direct non-learned precedent for obtaining a compact approximation of the transport matrix by physically sampling a few rows and columns (compare: a generator fed stratified visibility rays emitting low-rank factors).
- url: https://www.cs.cornell.edu/~kb/projects/rowcolumnSampling

### Hasan, Krivanek, Walter, Bala - Virtual Spherical Lights for Many-Light Rendering of Glossy Scenes (2009, ACM TOG 28(5) (SIGGRAPH Asia 2009)) [read_primary_source]
- mechanism: Each virtual light's contribution is integrated over a non-zero solid angle (a sphere around the light-path vertex) using both BRDFs, removing the spikes from narrow glossy lobes and the 1/r^2 term, so no clamping is needed; rendered with matrix row-column sampling.
- precomputed / learned: Nothing learned; up to about 250,000 VSLs per image (limited by reduced-matrix memory).
- valid under geometry edits: Not addressed (offline, per image).
- sharp glossy interreflection: Central topic. The paper states prior many-light methods 'are incorrect for two reasons': diffuse-only VPLs ignore light bouncing off glossy surfaces, and clamping removes energy. Using the full glossy BRDF at a point VPL 'does not help either, since even more clamping has to be applied'. VSLs preserve most glossy effects but blur some lighting features.
- cost claims: High-quality images in 1-4 minutes (GPU) where path tracing references took 2.2-316 hours on 8 cores; Kollig-Keller compensation 3.5 hours and photon mapping 1 hour on the motivating example.
- limitation: Biased but consistent (converges as VSL count grows and radius shrinks); blur; memory cap on VSL count.
- relevance: Shows the known price of making a finite sender set carry glossy-to-glossy transport: angular integration at both ends, with blur proportional to the support - narrow lobes are regularised, not resolved.
- url: https://cgg.mff.cuni.cz/~jaroslav/papers/2009-sigasia-vsl/2009-vsl-paper.pdf

### Diolatzis, Philip, Drettakis - Active Exploration for Neural Global Illumination of Variable Scenes (2022, ACM TOG 41(5); arXiv 2203.08272) [read_primary_source]
- mechanism: A neural generator renders a variable scene from first-hit G-buffers plus an explicit vector of scene parameters (object positions/rotations, materials, lights); training data are path-traced renderings chosen by a Markov-chain exploration of the parameter space.
- precomputed / learned: One network per variable scene, trained 5-18 hours on an RTX 6000 (under an hour when specular bounces are ray traced at runtime).
- valid under geometry edits: Valid without retraining only for edits inside the trained parameterisation (the declared variables and their ranges, including moving objects); anything else requires retraining.
- sharp glossy interreflection: Learns all paths including glossy reflection and caustics, but 'high-frequency effects, such as sharp mirror reflections, may not be reproduced exactly'; the authors suggest a hybrid with real-time ray tracing for specular interactions.
- cost claims: 4-6 fps at 900x900 on an RTX 3090 in an unoptimised prototype.
- limitation: Per-scene; scaling in the number of variables (128 variables already loses some shadows and highlights); source-dependent output.
- relevance: A learned competitor that does survive rigid edits without retraining - by enumerating them at training time - and that also hands sharp specular transport back to ray tracing.
- url: https://ar5iv.labs.arxiv.org/html/2203.08272

### Zeng, Dong, Peers, Wu, Tong - RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination (2025, SIGGRAPH 2025 Conference Papers; arXiv 2505.21925) [read_abstract_or_secondary]
- mechanism: Rendering as sequence-to-sequence: triangle tokens with reflectance go through a view-independent transformer stage modelling triangle-to-triangle light transport, then a view-dependent stage turns ray-bundle tokens into pixel patches.
- precomputed / learned: Transformer weights learned once across scenes; 'does not require per-scene training or fine-tuning'.
- valid under geometry edits: By construction any edit is a new input sequence, so no retraining; the whole forward pass is re-run. Light sources are part of the token set (not a source-independent operator applied afterwards).
- sharp glossy interreflection: Abstract claims evaluation on 'scenes with varying complexity in shape and light transport'; no specific claim on sharp glossy interreflection in what I read.
- cost claims: None in the abstract; a secondary summary reports a limit of 4,096 triangles.
- limitation: Scale (token count); no physical guarantees stated in the abstract; abstract-level reading only.
- relevance: The closest published instance of a learned, geometry-conditioned, cross-scene transport model with an explicit view-independent transport stage; it bounds what has been shown (small scenes, no operator-level linearity/passivity contract).
- url: https://arxiv.org/abs/2505.21925

## Implications
- The path family that failed (light -> sharp reflector -> diffuse patch) is the LS+D caustic family, and its failure under receiver-side gathering is documented prior art rather than a property of the learned operator: Krivanek 2005 states that radiance caching 'as a gathering technique, cannot solve ... caustics'; Jensen 1996 never computes caustics by Monte Carlo gathering and carries them in a separate light-traced map; GRIS 2022 reports sharp caustics as an undersampling failure of reuse; ReSTIR GI states that a sample contributing strongly at one receiver lies outside the peak lobe of another.
- Every cache in this survey that is read by a path guards the read with a footprint or roughness test, and a lookup at a near-mirror second vertex violates all of them: Jensen (approximate estimate only after a diffuse bounce or beyond a distance threshold), NRC (area-spread termination), SHaRC (cone spread must exceed the voxel size), path space filtering (specular vertices 'should not be selected'), Lumen (screen cache only for roughness 0.4-1.0, world cache for 0.3-0.4, rays below 0.3), DDGI (second-order glossy forced to maximum roughness), ReSTIR PT (reconnection only above GGX roughness of about 0.2).
- No surveyed runtime cache stores sharp glossy interreflection in a form that survives a geometry edit. The two systems that do cache sharp multi-glossy transport (light field probes 2017, glossy probe reprojection 2020) are precomputed and static, and both tie storage resolution to the sharpest lobe (probe texel solid angle <= pixel solid angle for mirrors; more resolution for shinier surfaces). Lumen makes the same statement for its surface cache (mirror reflections need cards at screen pixel density).
- For broad (diffuse and rough-glossy) transport, physically computed caches with no learned parameters already deliver the saving at published costs of roughly 1.5-3.5 ms per 1080p frame on 2019-2022 GPUs (path space filtering 1.5 ms; DDGI 2.6 ms diffuse + 2.4 ms ray-traced glossy; NRC 2.6 ms; GI-1.0 1.9-3.4 ms) and 6-11 ms for a complete production system including reflections (Lumen). A learned operator restricted to broad transport is competing against those numbers.
- What those physical caches do not provide is instant validity: after a rigid move their stored radiance is wrong and relaxes over frames (DDGI hysteresis 0.85-0.98; ReSTIR GI validation every 6 frames and reservoir clamp 30; NRC about 8 frames usable / 64 frames for correct colours; NIRC about 20 frames; GIBS about 15 frames near-converged; Gautron record lifespans 5-10 frames; Lumen 'fixed update cost, variable lighting latency' with ghosting behind movers). Any economic comparison therefore has to state whether the target is steady-state cost per frame, lag after an edit, or error during the transient.
- All runtime caches surveyed store source-dependent radiance or irradiance, not a source-independent operator; a light change invalidates them just as a geometry change does. Source independence plus zero-lag validity after a pose change are the properties a geometry-conditioned operator offers that these caches do not, and the operator's cost of regenerating factors per pose (visibility rays plus network inference) is the quantity to set against the caches' per-frame ray budgets.
- Rigid motion preserves object-attached state at near-zero cost in shipped systems (GIBS surfels follow a transform ID; Lumen cards and mesh distance fields are per mesh; acceleration structures refit), while everything that encodes inter-object visibility or received radiance is recomputed. Laine 2007 shows the cheap asymmetric case: moving objects may receive cached indirect light but may not send it.
- Physical precedents for selective recomputation exist without learning: Dmitriev 2002 identifies invalid photon groups with pilot photons and re-shoots only those (non-conservatively); Gautron 2007 assigns each record a validity interval from measured lighting change; Lumen updates only modified distance-field bricks and a budgeted subset of cache pages; Pantaleoni 2020 resets cell counters locally.
- For sharp transport the prior art's economic lever is sampling, not storage: ReSTIR PT re-derives the constrained vertex by random replay and remains unbiased; path guiding uses a learned or cached field only as a proposal density; NIRC adds a path-traced residual to a cached value. In all three a wrong cached value at a sharp receiver becomes variance. A table lookup, as in ReSTIR GI's Lambertian-sample assumption or a radiance cache, turns the same error into a biased, visibly localised reflection.
- Removing narrow-lobe energy from a finite-sample formulation and compensating elsewhere has a documented price: VPL clamping loses glossy energy; unbiased compensation approaches the cost of path tracing in glossy scenes (3.5 hours versus 34 s in the VSL paper's example); the VSL remedy integrates over a solid angle and blurs. A low-rank global term that compensates for a near-zero single-flight term is in the same position as clamping plus compensation.
- Angular resolution is the cost driver wherever directional radiance is cached: Krivanek's records are about 3.6 kB at harmonic order 10 and still cannot represent Lafortune exponents of 88-186 accurately; voxel cone tracing costs 16.6 ms at a 10 degree cone versus 3.9 ms at 60 degrees and spends about 1 GB largely on indirect specular; Lumen needs 8x8 directions for rough and 32x32 for mid-glossy lobes. A fixed patch x direction-bin table inherits this scaling.
- Even ReSTIR PT, the method that most directly targets glossy and specular reuse, reports that reconnection fails on near-delta BRDFs, that caustics from small distant highlights are poorly handled by its hybrid shift, and that its prototype cost 40-80 ms per frame on an RTX 3090 before the 2026 optimisations (2-3x faster). Sharp glossy interreflection under real-time budgets is not a solved problem that a new method merely has to match.
- Published 'cheaper than path tracing at matched quality' claims in this area are stated as equal-time error ratios (ReSTIR GI: 9.3x-166x MSE at about +1 ms over an 8 ms 1-spp frame; Pantaleoni: about 13x at equal variance; Dittebrandt: >= 10x MSE) or as time-to-quality with a denoiser. The reference cost of 1-spp two-bounce path tracing at 1080p on an RTX 3090 is about 8-19 ms in those papers.
## Closest precedents
- Hasan, Pellacini, Bala 2007 (matrix row-column sampling): a physically computed low-rank-style approximation of the sample x light transport matrix from a few sampled rows and columns - the non-learned counterpart of 'visibility rays in, low-rank factors out'. Per frame, source-dependent, not geometry-generalising. https://www.cs.cornell.edu/~kb/projects/rowcolumnSampling
- Pantaleoni 2020 (progressive spatio-temporal filtering): a (position x outgoing-direction) hashed radiance table built by progressive-radiosity-like feedback of path samples - the closest physically computed analogue of the tested patch x direction-bin table, including glossy directionality. Source-dependent; updated by counter resets. https://arxiv.org/abs/2005.07547
- Jensen 1996 (global photon map + final gathering, separate caustic map): same two-tier structure as the tested mechanism (exact first events, approximate store read after a wide bounce), with the LS+D family routed to a separate light-side estimator. https://www.cs.princeton.edu/courses/archive/fall14/cos526/papers/jensen96.pdf
- Zeng et al. 2025 (RenderFormer): learned, geometry-conditioned, cross-scene, with an explicit view-independent triangle-to-triangle transport stage; lights are tokens rather than an input to a separate linear operator; small scenes. https://arxiv.org/abs/2505.21925
- Diolatzis, Philip, Drettakis 2022 (Active Exploration): learned generator conditioned on explicit rigid-pose parameters, valid without retraining inside the trained range; per-scene, source-dependent, sharp mirror reflections not exact. https://arxiv.org/abs/2203.08272
- Ren et al. 2024 (LightFormer, SIGGRAPH 2024): neural many-light formulation - each light generates VPLs encoded into a neural light representation, composited per pixel by attention; claims fully dynamic scenes in real time. Read at project-page level only; glossy handling and generalisation to unseen scenes not confirmed. https://wylighting.github.io/lightformer
- Muller et al. 2021 (NRC) and Dereviannykh et al. 2025 (NIRC): learned radiance (not operator), trained online, so valid under any edit after a warm-up of roughly 8-64 frames; NIRC adds an unbiased residual. https://arxiv.org/abs/2106.12372 ; https://arxiv.org/abs/2412.04634
- Hadadan, Chen, Zwicker 2021 (Neural Radiosity, arXiv 2105.12319): a network represents the full radiance field by minimising the rendering-equation residual; per-scene optimisation, handles non-diffuse surfaces per the abstract; Su et al. 2024 (Dynamic Neural Radiosity, SIGGRAPH Asia 2024) extends it to variable scenes with an explicit scene vector (search-snippet level only). https://arxiv.org/abs/2105.12319
- Dmitriev et al. 2002 (selective photon tracing): not learned, but the closest precedent for a geometry-indexed partition of transport with a cheap probe that identifies the dirty subset after a rigid move. https://www.graphicon.ru/oldgr/en/publications/text/spt.pdf
- Precomputed transport operators applied to the current direct illumination in a static scene are cited by Laine et al. 2007 as a separate category ('precompute transport operators that are used at runtime ... in a static scene'); that family (PRT, direct-to-indirect transfer) was outside this angle and is not surveyed here.
## Unverified / uncertain
- I found no published runtime cache that retains sharp glossy interreflection under geometry edits; this is absence of evidence within one search session, not a proof that none exists.
- ReSTIR PT Enhanced: the 37.1 ms -> 12.6 ms figure comes from a search snippet of the paper's Figure 1; from the PDF text I confirmed only '2-3x faster' and 'originally 40-80 ms per frame on an RTX 3090'. Year is 2026 (PACMCGIT 9(1)), although the task brief's field may know it as 2025.
- Lumen numbers were extracted from slide-note text with imperfect table alignment; the association of 6.38 / 11.31 ms totals and 2.30 / 4.62 ms reflections with The Matrix Awakens on console rests on the accompanying sentence ('On High settings the total Lumen cost is 6.4 milliseconds, while Epic ... 11.3'). The hardware for that table was not unambiguous in the extracted text.
- GIBS timings (about 7 ms, about 11 ms in a stress case, 15 frames, 'about five seconds') are from slide notes without a consolidated table; hardware not extracted. I found no statement on glossy in the slides beyond the diffuse-only scope.
- DDGI convergence in frames (about 7-50) is my conversion of the published hysteresis range 0.85-0.98 to an exponential time constant, not a number stated in the paper.
- NRC adaptation figures ('usable after about 8 frames (~70 ms)', 'overall colours correct after the first 64 frames') come from an automated summary of the ar5iv full text; I did not locate the sentences myself.
- NIRC: per-frame timings in its teaser (e.g. path tracing 41.2 ms versus biased variant 37.9 ms) were visible in extracted text but I could not map them to scenes/hardware with confidence, so they are omitted from the entry.
- RenderFormer's 4,096-triangle limit is from a search summary, not from the abstract I read.
- Keller 1997 (Instant Radiosity) and Kollig & Keller 2004 were not read directly; their content is taken from descriptions in Hasan et al. 2009 and search summaries.
- Ward 1988: only the abstract page was read; the statement that specular components are traced separately is supported indirectly (abstract mentions 'diffuse and specular components'; Jensen 1996 says Radiance path-traces glossy surfaces).
- Gautron et al. 2007 was read through the author's 2008 course chapter, not the TVCG paper; no speedup numbers extracted.
- Real-Time Ray Traced Caustics (RTG2 ch. 30): a timing table exists but did not extract cleanly; no numbers reported.
- Ritschel et al. 2008, Imperfect Shadow Maps (SIGGRAPH Asia 2008): verified to exist; reported as fully dynamic VPL-based indirect illumination at 19 fps on a GeForce 8800 GTX (project-page snippet). Glossy support not confirmed; the MPI page returned 403. Not included as an entry.
- Boisse 2021, World-Space Spatiotemporal Reservoir Reuse (SIGGRAPH Asia 2021 Technical Communications): PDF read; implemented for single-bounce diffuse GI only; a performance table exists but values did not extract. Folded into the GI-1.0 lineage rather than given its own entry.
- Dachsbacher et al. 2014, Scalable Realistic Rendering with Many-Light Methods (CGF 33(1)): existence and scope verified; its discussion of glossy limitations was not read.
- Walter, Khungurn, Bala 2012, Bidirectional Lightcuts: seen only in a search snippet (claims gloss handling); not read.
- Zeltner, Georgiev, Jakob 2020, Specular Manifold Sampling (SIGGRAPH 2020): abstract fragments only; relevant as the deterministic-connection alternative for reflective caustics, not surveyed in depth.
- Recalled from memory and not verified in this session: Tawara, Myszkowski, Seidel 2004 (temporal coherence in final gathering); Smyk et al. 2005 (temporally coherent irradiance caching); Walter et al. 1999 Render Cache and Tole et al. 2002 Shading Cache; Tabellion & Lamorlette 2004 (cited inside Laine 2007 and the Lumen notes, not read); Enlighten runtime radiosity; Embark's Kajiya renderer; AMD Brixelizer GI; Sannikov's Radiance Cascades; Kaplanyan & Dachsbacher 2010 light propagation volumes (cited in the VCT paper as the comparison baseline).
- The LuxCoreRender forum summary claiming ReSTIR GI 'cannot handle caustics' is a secondary source; the paper itself states the glossy multi-bounce and Lambertian-sample limitations quoted in the entry.
- Cost numbers span hardware from a 2002 dual Pentium 4 to an RTX 3090 and are not normalised; cross-entry comparisons are indicative only.