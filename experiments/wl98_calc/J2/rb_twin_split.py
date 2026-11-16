"""WL98 J2 scratch (design input only, not evidence).

Question: for RB-DTO (D3), how much of the claimed edit-time lever is specific to
LEARNING, versus available to a learning-free per-asset PRECOMPUTED rigid-body
transfer twin (TwinP: path-traced exits per incident cell, lobe-fitted and
tabulated per body, applied like RB-DTO's sharp channel)?

Both RB-DTO and TwinP are rigid-invariant, reused unchanged across rigid edits,
and applied per incident sample. So at edit time they differ only in apply cost
(network query vs table lookup + lobe eval) and memory; they differ in one-off
cost by per-asset path tracing (TwinP) vs one forward pass of a shared compiler.

All constants are placeholders labelled as such; ray / gather unit costs are the
WL97 E5 recorded kernel-basis rates (0.27-1.06 ns per ray, 0.39-0.81 ns per entry).
"""
import json, itertools, numpy as np

def inc_bins(alpha):
    # same rule as D3 uniform_binned_scaling: four bins across a lobe of half-width 2*alpha,
    # over the hemisphere (2*pi sr): bins ~ 2*pi / (alpha_cell^2), alpha_cell = alpha
    return int(round(2*np.pi/((alpha)**2)/ (np.pi)))  # ~2/alpha^2 ; 0.1 ->200, 0.03->2222

out = {"notes": __doc__, "rows": []}
for Nb, alpha, Ks, spc, Jint, tray in itertools.product(
        [200, 400, 1000], [0.1, 0.03], [4, 8], [256, 4096], [4, 16], [0.27, 1.06]):
    nb = inc_bins(alpha)
    cells = Nb * nb
    # TwinP memory: per incident cell Ks lobes x 10 floats (pos2, dir2, aniso/width3, weight3) float16
    memP_MB = cells * Ks * 10 * 2 / 1e6
    # TwinP one-off precompute: spc exit samples per incident cell, Jint events each, 1 ray per event
    preP_s = cells * spc * Jint * tray * 1e-9
    # RB-DTO memory (D3 example): weights ~ 2 MB shared + Nb*128 floats latents + output cache
    memL_MB = 2.0 + Nb * 128 * 4 / 1e6
    out["rows"].append(dict(Nb=Nb, alpha=alpha, inc_bins_per_patch=nb, Ks=Ks,
        exit_samples_per_cell=spc, Jint=Jint, t_ray_ns=tray,
        twinP_memory_MB=round(memP_MB, 2), twinP_precompute_s=round(preP_s, 3),
        rbdto_memory_MB=round(memL_MB, 2)))
# edit-time apply per incident sample: TwinP = locate cell + Ks lobe copies (~ Ks*10 loads), RB-DTO = network query
apply = {"twinP_ns_per_sample_placeholder": [1.0, 5.0],
         "rbdto_ns_per_sample_D3_placeholder": [5.0, 20.0],
         "reading": "edit-time apply of TwinP <= RB-DTO under these placeholders; the per-edit lever over per-update explicit sweeps is shared by both"}
out["apply"] = apply
json.dump(out, open("results/wl98_calc/J2/rb_twin_split.json", "w"), indent=1)
rows = out["rows"]
for a in [0.1, 0.03]:
    sel = [r for r in rows if r["alpha"] == a and r["Ks"] == 4 and r["Jint"] == 16 and r["t_ray_ns"] == 0.27]
    for r in sel:
        print(a, r["Nb"], r["inc_bins_per_patch"], r["exit_samples_per_cell"], "memP", r["twinP_memory_MB"], "MB pre", r["twinP_precompute_s"], "s  memL", r["rbdto_memory_MB"])
