# SEQ_A gap 50 mm — ReST-GI SSS scene as a Blender file (WL115, user request)

- `SEQ_A_gap50mm_ReSTGI_SSS_scene.blend` (Blender 5.1, Cycles; not rendered)
- Built by `experiments/ecro_sr115/blend115_build.py` from `SEQ_A_gap50mm_spec.json`, which `blendspec115.py SEQ_A_lift_3` exports from the exact ReST-GI state.
- Scene: SEQ_A state 3, sphere centre (0.90, 0.40, 0.45), radius 0.4, gap 50 mm.

**Cameras.** Two cameras match ReST-GI views V0 and H2 exactly: same origin, target, up = world z, square 512 x 512, FOV 32°. Camera `camera_V0` is active. The sphere centre projects to (0.500, 0.440) in both renderers.

**Sphere.** The ReST-GI SSS semantics (WL100 / WL53-validated Cycles equivalent):
- a white **Transparent BSDF** surface: index-matched, no refraction, no Fresnel;
- a homogeneous isotropic volume inside, with σ_t = 12 m⁻¹ and albedo (0.92, 0.80, 0.65);
- Volume Scatter and Volume Absorption both use colour = albedo and density = 12.

**Light.** One point light at (1.004, 1.361, 1.294).
- ReST-GI radiant intensity is I = 27.63 W/sr. Blender power is set to P = 4πI = 347 W, with radius 0 and soft falloff off.
- If Blender's point-light convention differs, only the absolute exposure changes.

**Materials.**
- Walls and ceiling: Lambert 0.7. Box A: Lambert 0.5. Box B: Lambert 0.6.
- Floor (material B) and panel (S1) are approximated. ReST-GI uses a learned closure of a layered coated conductor (WL81). Here each is a Principled BSDF: metallic, base colour = conductor reflectance, roughness = √α_b, coat weight 1, coat roughness = √α_i, coat IOR = η. The thin scattering slab is omitted.

**Render settings.**
- Cycles, 1,024 spp, 64 bounces of every type, no clamping, no denoising.
- Box pixel filter 1 px, Standard view transform, black world.
- The device is left at the default. Set GPU in Preferences if wanted.

**ReST-GI reference images.** `ReSTGI_reference_fulldepth_SEQ_A_gap50mm_{V0,H2}.exr` are the WL114 full-depth physical references: linear HDR, mean of 2 × 4,096 spp. Open them in the Image Editor for a side-by-side comparison.

**What to look at.** The thin bright line on the lower silhouette of the sphere, strongest at the lower left and lower right.
- In ReST-GI it comes almost entirely from camera events 2 and higher: the camera ray grazes the sphere, crosses only a 5–6 cm chord (ballistic transmittance about exp(−12 × 0.056) ≈ 0.5), and lands on the lit floor behind the sphere.
- With the index-matched boundary above, Cycles should show the same rim.
- Changing the surface to a refracting / Fresnel boundary (e.g. Principled with transmission, IOR 1.4) is the comparison that would remove it.
