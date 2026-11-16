# Track A: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List key experiment directories and file sizes] ls experiments/ecro86 experiments/ecro86a experiments/gm1_explicit87 experiments/ecro_mat94 experiments/ecro_attr95a experiments/ecro_util96a && wc -l experiments/ecro86/*.py experiments/ecro86a/*.py experiments/gm1_explicit87/*.py experiments/ecro_mat94/*.py experiments/ecro_attr95a/*.py experiments/ecro_util96a/*.py && ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
  -> Exit code 2
experiments/ecro86:
__pycache__
analyze86.py
data
diag_v3.py
econ86.py
ecro86.py
gen86.py
masks86.py
mats86.py
model86.py
protocol
pt86.py
run86.sh
scene86.py
tests86.py
train86.py
viz86.py
watch86.sh

experiments/ecro86a:
analyze86a.py
attrib86a.py
d2_86a.py
d3_86a.py
diag86a.py
protoco ...[cut]... y
   147 experiments/gm1_explicit87/pt87.py
   338 experiments/gm1_explicit87/quad87.py
   274 experiments/gm1_explicit87/run87.py
   342 experiments/gm1_explicit87/tests87.py
   164 experiments/gm1_explicit87/viz87.py
   207 experiments/gm1_explicit87/xport87.py
   379 experiments/ecro_mat94/an94.py
   310 experiments/ecro_mat94/c94.py
   578 experiments/ecro_mat94/mat94.py
    36 experiments/ecro_mat94/posthoc94.py
   168 experiments/ecro_mat94/viz94.py
   628 experiments/ecro_attr95a/a95.py
    29 experiments/ecro_util96a/figw3_posthoc.py
   819 experiments/ecro_util96a/w96a.py
  8991 total
RUN [List key experiment directories and file sizes] cd /c/Projects/ReST-GI && ls experiments/ecro86 experiments/ecro86a experiments/gm1_explicit87 experiments/ecro_mat94 experiments/ecro_attr95a experiments/ecro_util96a && wc -l experiments/ecro86/*.py experiments/ecro86a/*.py experiments/gm1_explicit87/*.py experiments/ecro_mat94/*.py experiments/ecro_attr95a/*.py experiments/ecro_util96a/*.py && ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Read scene86.py] cat experiments/ecro86/scene86.py
RUN [Read model86.py and ecro86.py] cat experiments/ecro86/model86.py experiments/ecro86/ecro86.py
  -> """WL86 scene family, patch / direction-bin discretisation and sampling rules (numpy only; no renderer import).

Room x,y in [-2,2], z in [0,3], open towards -y. Stationary: floor (ER B), ceiling + 3 walls (Lambert), panel (ER S1), sphere (ER S2).
Movable: box A and box B (Lambert), each given by ce ...[cut]... ntinue
        return [{'pos': [float(v) for v in pos], 'rgb': [FX['light_intensity']] * 3}]


def light_arrays(lights):
    """-> pos [2,3], rgb [2,3], axis [2,3], cos_cut [2] (point light: cos_cut = -2; unused slot: rgb = 0)."""
    pos = np.zeros((2, 3)); rgb = np.zeros((2, 3)); axis = np.tile(Z, (2, 1)); cc = np.full(2, -2.0)
    for i, l in enumerate(lights):
        pos[i] = l['pos']; rgb[i] = l['rgb']
        if 'aim' in l:
            a = np.array(l['aim'], np.float64) - pos[i]; axis[i] = a / np.linalg.norm(a); cc[i] = math.cos(math.radians(l['cone_deg']))
    return pos, rgb, axis, cc
  -> """WL86 ECRO v0 operator (torch): geometry-conditioned, source-independent, linear and non-negative.

Geometry state (no radiance): patch tables, frozen ER states of the patch materials, stratified hemisphere state rays
(hit patch, outgoing bin at the hit patch) -> generator network -> response fact ...[cut]... 'camera2'] if opt.get('--cam') == '2' else None)
        out = os.path.join(root, 'results', 'renders'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), B=B.astype(np.float32), **o)
        json.dump({'cfg': cfg, 'model': mtag, 'variant': variant, 'seed': seed, 'spp': spp, 'timing': T_, 'gpu_mem_mib': PT.gpu_mem()},
                  open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
        print(tag, json.dumps(T_), flush=True)
    elif cmd == 'contracts':
        contracts(root, args[0])


if __name__ == '__main__':
    main()
    os._exit(0)
RUN [Read scene86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/scene86.py
RUN [Read model86.py and ecro86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/model86.py experiments/ecro86/ecro86.py
RUN [Read protocol representation and pt86.py] python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(P['representation'],indent=1));print(json.dumps(P['architecture'],indent=1));print(list(P.keys()))" && cat experiments/ecro86/pt86.py
  -> {
 "family": "patch x outgoing-direction-bin table of the higher-order outgoing radiance (2..5 scattering events), produced by a non-negative linear operator from the patch x incident-direction-bin direct irradiance",
 "patches": "1270: 25 cm grid on floor / ceiling / walls, 20 cm on the panel, 3  ...[cut]... geom, ls = config(cfg)
        ls = opt.get('--ls', ls)
        ctx = Ctx(geom)
        lt = Lights(FX['light_sets'][ls])
        t0 = time.perf_counter()
        fa, ca = ctx.field(lt, spp // 2, seed)
        fb, cb = ctx.field(lt, spp // 2, seed + 500)
        out = os.path.join(root, 'results', 'fields'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), fa=fa, fb=fb, ca=ca, cb=cb)
        print(tag, 'field', round(time.perf_counter() - t0, 2), 's', 'valid', float((ca > 0).mean()), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
RUN [Read protocol representation and pt86.py] cd /c/Projects/ReST-GI && python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(P['representation'],indent=1));print(json.dumps(P['architecture'],indent=1));print(list(P.keys()))" && cat experiments/ecro86/pt86.py
RUN [Read pt86a.py] cat experiments/ecro86a/pt86a.py
  -> """WL86A diagnostic sources for the frozen WL86 transport module (pt86 is imported unchanged; nothing here is a runtime input of ECRO).

Field probes (radiance leaving a surface point made of 2..5 scattering events, exactly pt86.Ctx.field) under controlled incident fields:
  'true'            the li ...[cut]... e irradiance in the bin of its direction); `visible` = fraction of 256 patch points it reaches."""
    G = ctx.G
    j = int(bm['pid'])
    w = np.array(bm['w'], np.float64)
    loc = np.array([w @ G.PT[j], w @ G.PB[j], w @ G.PN[j]])
    iz = int(np.clip(math.floor(loc[2] * S.NZ), 0, S.NZ - 1)); ip = int(np.clip(math.floor((math.atan2(loc[1], loc[0]) + math.pi) * S.NPHI / (2 * math.pi)), 0, S.NPHI - 1))
    frac = 1.0 if bm.get('half') is None else 0.5
    s = np.zeros((ctx.N, NB, 3), np.float32)
    s[j, iz * S.NPHI + ip] = bm['E'] * loc[2] * frac
    return s, iz * S.NPHI + ip, float(loc[2])
RUN [Read pt86a.py] cd /c/Projects/ReST-GI && cat experiments/ecro86a/pt86a.py
RUN [Read closure83_dr.py] ls experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 && wc -l experiments/fnto_compiled83/closure83_dr.py && cat experiments/fnto_compiled83/closure83_dr.py
  -> experiments/fnto_compiled81:
bench81.py
closure81.py
closure81_dr.py
compress81.py
contract81.py
data81.py
diag81_posthoc.py
energy81.py
eval81.py
fit81.py
gate81.py
layered81.py
protocol
refaudit81.py
render81.py
run81.sh
testsB81.py
viz81.py

experiments/fnto_compiled82:
bench82_posthoc.py
closure ...[cut]...      if self.arm == 'ER':
            tx, ty = wr.x * self.eta, wr.y * self.eta
            r2 = tx * tx + ty * ty
            wex = mi.Vector3f(tx, ty, dr.sqrt(dr.maximum(1.0 - r2, 0.0)))
            ok_exit = ~refr | ((r2 < 1.0) & (wr.z > 0.0))
            wr = dr.select(refr, wex, wr)
        wo = dr.select(dr.eq(sel, 0), mi.warp.square_to_cosine_hemisphere(u2), wr)
        f, pdf = self.eval_pdf(wi, wo, active)
        valid = active & ok_exit & (wo.z > EPS) & (pdf > 0)
        w = dr.select(valid, f / dr.select(valid, pdf, 1.0), 0.0)
        return wo, dr.select(valid, pdf, 0.0), w, valid
RUN [Read closure83_dr.py] cd /c/Projects/ReST-GI && ls experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 && wc -l experiments/fnto_compiled83/closure83_dr.py && cat experiments/fnto_compiled83/closure83_dr.py
RUN [Read quad87.py] cat experiments/gm1_explicit87/quad87.py
  -> """WL87 discrete ER scattering on the WL86 patch x bin state (learning-free; frozen ER closures, Dr.Jit).

Notation (patch-local frame (t, b, n) of scene86; 64 equal-solid-angle bins, DOMEGA = 2 pi / 64; eval(wi, wo) = f(wi, wo) cos(wo)).
ER f is reciprocal (f(a, b) = f(b, a): symmetric GGX lobes, s ...[cut]...         # inside a patch-frame bin and above the local horizon
        cv = dr.select(keep, val * E * (dr.gather(F, qs['invom'], slot) * (1.0 / (P * Mq))), 0.0)
        dr.eval(cv, slot); dr.sync_thread()                                    # evaluate BEFORE the scatter (lazy-scatter defect)
        for c in range(3):
            dr.scatter_reduce(dr.ReduceOp.Add, buf[c], cv[c], slot)
        dr.eval(buf)
    dr.sync_thread()
    q = np.stack([np.asarray(x, np.float64) for x in buf], 1).reshape(N, NB, 3)
    ql = np.stack([np.asarray(x, np.float64) for x in bl], 1)
    return q + ql[:, None, :]
RUN [Read quad87.py] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/quad87.py
RUN [Read xport87.py, fx87.py, pt87.py] cat experiments/gm1_explicit87/xport87.py experiments/gm1_explicit87/fx87.py && sed -n 1,147p experiments/gm1_explicit87/pt87.py
  -> """WL87 explicit (learning-free) patch x bin transport on the WL86 geometry state.

Primary comparator EXP (receiver gather on the deployed state):
  L_1 = q                                           once-scattered table (quad87.qtable)
  L_{k+1}[i, bo] = sum_r W_ray[m_i, r, bo] L_k[hit_r, dbin_r]   ...[cut]...  F(1.0), F(0.0))
        a = ctx.accum(out, slot, ns)
        acc = a if acc is None else {k_: acc[k_] + v for k_, v in a.items()}
        done += bb; launch += 1
    cnt = np.maximum(acc['cnt'].reshape(ctx.N, NB), 1)[..., None]
    return np.stack([np.stack([acc[f'o{k}_{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / cnt for k in range(2, D)])


def table_fill(ctx, lights, spp, seed):
    """Deployable physical table: path-traced 2..5-event field at `spp` samples per (patch, bin) slot (pt86.Ctx.field, one pass)."""
    f, cnt = ctx.field(lights, spp, seed)
    return f.astype(np.float64)
RUN [Read xport87.py, fx87.py, pt87.py] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/xport87.py experiments/gm1_explicit87/fx87.py && sed -n 1,147p experiments/gm1_explicit87/pt87.py
RUN [Read an94.py] cat experiments/ecro_mat94/an94.py
  -> """WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.

  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints F ...[cut]...  'final.json'))
    import viz94
    files = viz94.figures(R, outd, B, Cs, L)
    man = []
    for f in files:
        man.append({'file': os.path.basename(f), 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest()})
    jd({'worklog': 94, 'files': man, 'source_root': R}, os.path.join(outd, 'manifest.json'))
    print('FINAL', json.dumps(L), flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if a[0] == 'stageB':
        stageB(a[1])
    elif a[0] == 'stageC':
        stageC(a[1], a[2])
    elif a[0] == 'final':
        final(a[1], a[2])
RUN [Read an94.py] cd /c/Projects/ReST-GI && cat experiments/ecro_mat94/an94.py
RUN [Grep for an94.W consumers and hard-coded weights] cd /c/Projects/ReST-GI/experiments && grep -rn "import an94\|an94\b\|AN\.W\|AN\.ex\|A94\.\|\.W\[None\|NPHI, NZ = \|NB, NPHI, NZ\|// 8) + 0.5\|//8\|// NPHI) + 0.5\|// S.NPHI + 0.5\|// S.NPHI) + 0.5\|// NPHI + 0.5" --include=*.py ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a ecro86 ecro86a gm1_explicit87 gm1_relation88 gm1_relation89 gm1_gap90 gm1_c3urx1_91 2>/
  -> ecro_attr92/diag92.py:119:    return ((np.arange(S.NB) // S.NPHI) + 0.5) / S.NZ
ecro_mat94/an94.py:3:  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
ecro_mat94/an94.py:4:  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints FIT / ...[cut]... nk66
fnto_rank75
fnto_repr68
fnto_reuse58
fnto_rule57
fnto_sel56
fnto_smoke47
fnto_spatial49
fnto_src73
fnto_srcid78
fnto_sym79
fnto_visual_review
fnto_walker71c
full_recompute_benchmark
gm1_c3urx1_91
gm1_char85
gm1_explicit87
gm1_gap90
gm1_relation88
gm1_relation89
neural_recompute_cost
phys_ref53
radiometric_relation_state_prototype
relational_residual_prototype
rna_baseline_quality
rna_reproduction
rna_selective_reconstruction_lego
rna_surface_ownership
rna_surface_ownership_b2
rna_surface_ownership_lego
selective_refit_oracle
server_setup
spt_lego
sunreuse70
surface_owned_radiance
xbench69
RUN [Find all AN. references in WL94-96A code] cd /c/Projects/ReST-GI/experiments && ls audit97 ../results/audit97 ../docs/audits 2>/dev/null; grep -n "AN\.\|import an94\|as AN" ecro_mat94/posthoc94.py ecro_mat94/viz94.py ecro_mat94/c94.py ecro_mat94/mat94.py ecro_mat94r/*.py ecro_exp94b/*.py ecro_q94c/*.py ecro_s94d/*.py ecro_attr95a/a95.py ecro_util96a/*.py | cut -c1-260
  -> ../docs/audits:
wl97_evidence_reliability

../results/audit97:
B
C2
MAP

audit97:
B
C2
MAP
ecro_mat94/posthoc94.py:7:import an94 as AN
ecro_mat94/posthoc94.py:8:F = AN.fref(R); M = AN.masks(R)
ecro_mat94/viz94.py:5:import an94 as AN
ecro_mat94/viz94.py:38:    M = AN.masks(R)
ecro_mat94/viz94.py:39:  ...[cut]... :19:import an94 as AN  # noqa: E402
ecro_util96a/w96a.py:358:    M = AN.masks(R)
ecro_util96a/w96a.py:359:    ref = AN.load_ref(R, 'none'); tot = ref['tot'].mean(0)
ecro_util96a/w96a.py:368:            e = AN.eh(ref['h'], X[k], tot, m)
ecro_util96a/w96a.py:700:        M = AN.masks(R); ref = AN.load_ref(R, 'none'); tot = ref['tot'].mean(0)
ecro_util96a/w96a.py:747:    M = AN.masks(R); ref = AN.load_ref(R, 'none')
ecro_util96a/w96a.py:748:    h = AN.lum(ref['h'].mean(0)); tot = AN.lum(ref['tot'].mean(0))
ecro_util96a/w96a.py:750:    X = {k: AN.lum(load_reps(R, k + sfx, n).mean(0)) for k in show}
RUN [Grep bin-layout usages across WL92-96A] cd /c/Projects/ReST-GI/experiments && wc -l ecro_attr92/*.py ecro_own92a/*.py ecro_cf92b/*.py ecro_stage93*/*.py ecro_local93d/*.py ecro_mat94r/*.py ecro_exp94b/*.py ecro_q94c/*.py ecro_s94d/*.py | tail -50; grep -n "NPHI\|NZ\b\|DOMEGA\|// 16\|// 8\|reshape(.*, 4, 16\|reshape(.*8, 8\|view(.*4, 16" ecro_attr92/*.py ecro_own92a/*.py ecro_cf92b/*.py ecro_stage93*/*.py ecro_local93d/*.py ecro_mat94/mat94.py ecro_mat94/c94.py ecro_mat94r/*.py ecro_exp94b/*.py ecro_q94c/*.py ecro_s94d/*.py ecro_attr95
  -> 380 ecro_attr92/analyze92.py
   545 ecro_attr92/diag92.py
    54 ecro_attr92/fx92.py
   276 ecro_attr92/interv92.py
   113 ecro_attr92/tests92.py
   181 ecro_attr92/viz92.py
   199 ecro_own92a/analyze92a.py
   471 ecro_own92a/own92a.py
    82 ecro_own92a/tests92a.py
   138 ecro_own92a/viz92a.py
   2 ...[cut]... .W as coded (8 x 8)
ecro_attr95a/a95.py:63:    z = ((k // S.KAPPA) // S.NPHI + u[2, 0]) / S.NZ
ecro_attr95a/a95.py:64:    return (z * S.DOMEGA / S.KAPPA).astype(np.float64)
ecro_attr95a/a95.py:179:    T['T_BINS'] = {'pass': S.NZ == 4 and S.NPHI == 16 and S.NB == 64 and agree >= 0.999 and abs(WT.sum() - math.pi) < 1e-9,
ecro_util96a/w96a.py:32:ZB = ((np.arange(NB) // 16) + 0.5) / 4
ecro_util96a/w96a.py:514:    o_r, o_f, o_s = S.stationary_rects, S.box_faces, (S.SPH_NPHI, S.SPH_NZ)
ecro_util96a/w96a.py:518:    S.SPH_NPHI = max(2, int(round(o_s[0] * s))); S.SPH_NZ = max(1, int(round(o_s[1] * s)))
NOTE: Now a95.py in full.
RUN [Read a95.py first half] cd /c/Projects/ReST-GI/experiments && sed -n 1,330p ecro_attr95a/a95.py
  -> """WL95A (protocol ecro_attr95a): sharp sender representability and flight attribution smoke on CK2 'none' (learning-free diagnostics).
Production code (model86 / ecro86 / pt86 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python a95.py stage0 <R> <R92A>     contract tests,  ...[cut]... 'passive_all': bool(all(p['passive'] for p in per)), 'nonneg': bool(min(p['min_entry'] for p in per) >= 0),
                           'mass_mean': float(np.mean([p['mass'] for p in per]))}
    out['sender'] = sm; out['operator'] = om
    # propagation: deployed + pooled draws
    rw0 = rw_of(u0, S)
    TAB = [k_ for k_ in snd]                                   # table senders
    RAY = ['X1c', 'X1']
    ex = {}; tabs_mean = {}; tabs_dep = {}

    def deliver(hit, dbin, seg, rw, store):
        res = {}
        for k_ in TAB:
            Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
RUN [Inspect WL95A results and operator metrics] ls results/ecro_attr95a results/ecro_attr95a/* | head -60; python - <<'EOF'
import json
j=json.load(open('results/ecro_attr95a/main.json')) if __import__('os').path.exists('results/ecro_attr95a/main.json') else None
if j:
    print(json.dumps(j['operator'],indent=1))
EOF
  -> results/ecro_attr95a/an.json
results/ecro_attr95a/chain.status
results/ecro_attr95a/deliv.npz
results/ecro_attr95a/gpu.txt
results/ecro_attr95a/main.json
results/ecro_attr95a/ops95a.npz
results/ecro_attr95a/s0.json

results/ecro_attr95a:
an.json
chain.status
deliv.npz
gpu.txt
main.json
ops95a.npz
s0 ...[cut]...  },
 "A4": {
  "op_flux_l1": 1.8086370180227094e-08,
  "mass": 0.9999999995463732,
  "albedo_max": 0.8246719744363133,
  "albedo_mean": 0.6349271624965894,
  "min_entry": 0.0009212702598019495,
  "passive": true,
  "nonneg": true
 },
 "A0_s8601": {
  "op_flux_l1_mean": 1.0179047599119777,
  "albedo_max": 2.263723684963198,
  "passive_all": false,
  "nonneg": true,
  "mass_mean": 0.8689575677345066
 },
 "A0_s8602": {
  "op_flux_l1_mean": 0.9870053594284064,
  "albedo_max": 3.077449703949969,
  "passive_all": false,
  "nonneg": true,
  "mass_mean": 0.8349744492310064
 }
}
RUN [Inspect WL95A results and operator metrics] cd /c/Projects/ReST-GI && ls results/ecro_attr95a results/ecro_attr95a/* | head -60; python - <<'EOF'
import json
j=json.load(open('results/ecro_attr95a/main.json')) if __import__('os').path.exists('results/ecro_attr95a/main.json') else None
if j:
    print(json.dumps(j['operator'],indent=1))
EOF
RUN [Read a95.py second half] cd /c/Projects/ReST-GI/experiments && sed -n 330,600p ecro_attr95a/a95.py
  -> Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
            res[k_] = Dt
        for k_, cen in (('X1c', True), ('X1', False)):
            L, nr = x1_rays(PT, ctx, lt, hit, seg, cen)
            res[k_] = receive(flight_ray(L, rw, N, S), Mrcv)
            store['rays_S1'] = nr
        retu ...[cut]... et_xticks([]); ax[i, f_].set_yticks([])
            if f_ == 0:
                ax[i, f_].set_ylabel(t, fontsize=7)
            if i == 0:
                ax[i, f_].set_title(f'box A face {f_}' + (' (panel-facing)' if f_ == 0 else ''), fontsize=7)
    fig.suptitle(f'WL95A box-A delivered S1-mediated exitance per patch (shared absolute scale 0..{vmax:.3g})', fontsize=9)
    fn = os.path.join(outd, 'figA3_footprint.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # F4 renders
    M = AN.masks(R)
    refimg = AN.load_ref(R, 'none'); h = refimg['h'].mean(0)
RUN [Dump WL95A analysis numbers] python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(json.dumps(a['flight'],indent=1)); print(a['verdicts']); print(json.dumps(a['rank16'],indent=1)); print(json.dumps(a['components'],indent=1)); print('floor',a['floor_E_H'])
for k,v in a['delivery'].items():
    if 'boxA' in k or 'panel_face' in k:
        print(k, {q: round(v[q],4) for q in ('ratio','abs_rel_l1','ref_half_noise','corr','peak_amp_ratio') if q in v}, 'se', v.get('se_abs_rel_l1'))
for k,v in a['ima
  -> {
 "n0": 0.05926934169469287,
 "E0": 0.46446359806088566,
 "Rs": 0.024175660096948215,
 "Rs_se": 0.034055179336480924,
 "Ra": 0.23188168674623544,
 "Ra_se": 0.046620248374830725,
 "Rp": 0.1571067426298993,
 "Rp_se": 0.06009806987301328,
 "Rq_incident": 0.06643679185456292,
 "Rq_incident_se ...[cut]... 229102,
"corr": -0.3339030580785568,
"corr_halves": 0.8673728501887797,
"rel_l1": 0.5977259301162438,
"half_noise_rel_l1": 0.0692607652362704
}
},
"sender_ck2_W94": {
"8601": {
"rel_l1_flux_W94": 1.0746493683253195,
"wl94d": 1.0746492201948834
},
"8602": {
"rel_l1_flux_W94": 1.2207934125201385,
"wl94d": 1.2207932828895183
}
}
},
"T_OPS": {
"pass": true,
"regenerated_rank": 32,
"regen": {
"op_l1": 0.23305602509523055,
"val_field_rel_l1": 0.1458158449362617
},
"wl95": {
"op_l1": 0.23305602509523055,
"val": 0.1458158449362617
},
"M64_nmf_npz_equals_mq_ext": true
}
}
RUN [Dump WL95A analysis numbers] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(json.dumps(a['flight'],indent=1)); print(a['verdicts']); print(json.dumps(a['rank16'],indent=1)); print(json.dumps(a['components'],indent=1)); print('floor',a['floor_E_H'])
for k,v in a['delivery'].items():
    if 'boxA' in k or 'panel_face' in k:
        print(k, {q: round(v[q],4) for q in ('ratio','abs_rel_l1','ref_half_noise','corr','peak_amp_ratio') if q in v}, 'se', v.get('se_abs_r
RUN [Per-draw delivery stats and available local artifacts] python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_attr95a/an.json'))
for k in ('A4','M64_spool','q256','X1c','X1','A0_s8601','A0_s8602'):
    print(k, json.dumps(a['delivery'][f'pool|{k}|boxA']['per_draw']))
z=np.load('results/ecro_attr95a/deliv.npz'); print(list(z.keys())[:40])
o=np.load('results/ecro_attr95a/ops95a.npz'); print({k:o[k].shape for k in o})
EOF
ls results/ecro_mat94 results/ecro_s94d results/ecro_util96a 2>/dev/null | head -80
  -> A4 {"ratio": {"mean": 1.544085478109459, "median": 1.5144533930162294, "p90": 1.9050564845486855, "worst": 1.4216002834854775}, "abs_rel_l1": {"mean": 0.7614863909731039, "median": 0.7261669512279727, "p90": 1.1483439683082886, "worst": 1.4499647791465888}}
M64_spool {"ratio": {"mean": 1.4948638661 ...[cut]... , 'draws|A0_s8601', 'draws|A0_s8602', 'draws|M64_spool', 'draws|q256', 'draws|X1c', 'draws|X1', 'ref_a', 'ref_b']
{'A1': (64, 64, 3), 'A2': (64, 64, 3), 'A3': (64, 64, 3), 'A4': (64, 64, 3)}
results/ecro_mat94:
chain.status
final.json
gpu.txt
models
pop
posthoc94.json
pre.json
reused_refs.txt
select94.json
stageB.json
stageB_rho.json
stageC_s8601.json

results/ecro_s94d:
audit.json
chain.status
final_s.json
gpu.txt
lam.json
models
pop
repro.json
s_s8601.json
s_s8602.json
sender

results/ecro_util96a:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz
RUN [Per-draw delivery stats and available local artifacts] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_attr95a/an.json'))
for k in ('A4','M64_spool','q256','X1c','X1','A0_s8601','A0_s8602'):
    print(k, json.dumps(a['delivery'][f'pool|{k}|boxA']['per_draw']))
z=np.load('results/ecro_attr95a/deliv.npz'); print(list(z.keys())[:40])
o=np.load('results/ecro_attr95a/ops95a.npz'); print({k:o[k].shape for k in o})
EOF
ls results/ecro_mat94 results/ecro_s94d results/ecro_util96a 2>/dev/null | head -80
RUN [Inspect material states] ls experiments/ecro86/data/ && python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz', allow_pickle=True)
print({k:z[k].shape for k in z})
st=z['states']; np.set_printoptions(precision=4, suppress=True, linewidth=200)
print('lambert',z['lambert'],'albedo',z['albedo'])
for m in range(3):
    s=st[m]
    print('mat',m,'lam',s[:24].reshape(3,8)); print(' au',s[24:31],'av',s[31:38]); print(' f0',s[38:45]); print(' pr',s[45:53]); print(' s',s[53:68].reshape(3,5),'eta',s[68
  -> mats86.json
mats86.npz
{'states': (6, 69), 'desc': (6, 10), 'lambert': (6,), 'albedo': (6,)}
lambert [0 0 0 1 1 1] albedo [0.  0.  0.  0.7 0.5 0.6]
mat 0 lam [[0.0746 0.2598 0.101  0.0205 0.0001 0.5171 0.0266 0.0004]
 [0.09   0.0645 0.1637 0.1694 0.0001 0.4848 0.0274 0.    ]
 [0.0491 0.0067 0.06 ...[cut]... .5647 0.0795 0.0001]
 s [[0.6461 0.9316 0.9985 0.9987 1.    ]
 [0.3245 0.7964 0.9985 0.9828 0.9806]
 [0.434  0.8379 0.9933 0.9976 1.    ]] eta 2.1186056
[[0.4335 1.3353 0.0433 0.9331 0.9802 0.9639 0.8957 0.958  0.6768 0.3323]
 [0.0507 1.3335 0.0238 0.2641 0.1061 0.1631 0.464  0.9276 0.8713 0.0518]
 [0.1314 1.6967 0.0244 0.4583 0.977  0.2454 0.8797 0.4449 0.7423 0.1139]
 [0.3    1.5    0.5    0.5    0.5    0.5    0.5    0.5    0.5    0.3   ]
 [0.3    1.5    0.5    0.5    0.5    0.5    0.5    0.5    0.5    0.3   ]
 [0.3    1.5    0.5    0.5    0.5    0.5    0.5    0.5    0.5    0.3   ]]
RUN [Inspect material states] cd /c/Projects/ReST-GI && ls experiments/ecro86/data/ && python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz', allow_pickle=True)
print({k:z[k].shape for k in z})
st=z['states']; np.set_printoptions(precision=4, suppress=True, linewidth=200)
print('lambert',z['lambert'],'albedo',z['albedo'])
for m in range(3):
    s=st[m]
    print('mat',m,'lam',s[:24].reshape(3,8)); print(' au',s[24:31],'av',s[31:38]); print(' f0',s[38:45]); print(' pr',s[45:53]); print(' s',s[53:6
RUN [Read mat94.py first part] cd /c/Projects/ReST-GI/experiments && sed -n 1,200p ecro_mat94/mat94.py
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]... t()}
    return PT, S, A


def set_pmat(PT, ctx, pm):
    ctx.G.PMAT[:] = pm
    ctx.PMATU = PT.U(np.asarray(pm, np.uint32))


def ctx_ck2(R):
    """Ctx of the CK2 scene (geometry, light set, camera of the WL93D addendum rev. 2), default material map."""
    import fx92
    fx92.inject(R)
    PT, S, A = install()
    FX = fx92.FX
    FX[SC['tag']] = {'T0': SC['geom'], 'T1': SC['geom'], 'moved': [], 'light_set': SC['light_set']}
    FX['light_sets'][SC['light_set']] = SC['lights']
    FX['camera'] = SC['camera']
    ctx = PT.Ctx(SC['geom'])
    return PT, S, A, FX, ctx, PT.Lights(SC['lights'])
RUN [Read mat94.py middle part] cd /c/Projects/ReST-GI/experiments && sed -n 200,420p ecro_mat94/mat94.py
  -> # ------------------------------------------------------------------------------------------------ models / operators
def extend_buffer(model):
    import torch
    if model.mat_states.shape[0] == 6:
        model.mat_states = torch.cat([model.mat_states, torch.as_tensor(new_rows(), device=model.mat ...[cut]...      'pid': pid_, 'mirror_patch': np.where(roi, mpid, -1), 'T': tgt_p, 'T_panel_face': face == PANEL_FACE,
            'T_other_faces': tgt_p & (face != PANEL_FACE), 'panel_facing_bins': pfb, 'S1': G.PSID == 5, 'PSID': G.PSID.copy(),
            'PMAT_default': default_pmat(G)}


# ------------------------------------------------------------------------------------------------ Stage A tests
def pre(R, R92A, R92B, R93D, RCK):
    import torch
    import ecro86 as EC
    import model86 as M86
    T = {}
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    dr, mi, CD, U, F = PT.dr, PT.mi, PT.CD, PT.U, PT.F
RUN [Inspect own92a training targets] cd /c/Projects/ReST-GI/experiments && sed -n 1,60p ecro_own92a/own92a.py && grep -n "e2_\|e3_\|use_single\|use_global\|loss\|def " ecro_own92a/own92a.py | cut -c1-220
  -> """WL92A: ECRO physical component supervision - controlled generator learning (protocol ecro_own92a).

  python own92a.py excl     <R> <R86>                       training exclusion list for E4 (geometry / light positions only)
  python own92a.py gen      <R> <R86> <split> <start> <count>  component ...[cut]...  fieldeval(R, R86):
254:                    Bs = mdl.apply(f, gs, s, use_global=False); Bg = mdl.apply(f, gs, s, use_single=False); B = Bs + Bg
279:                              'single_plus_global_minus_B': float((mdl.apply(f, gs, s, use_global=False) + mdl.apply(f, gs, s, use_single=False) - mdl.apply(f, gs, s)).abs().max() / float(mdl.apply(f, gs, s).abs().ma
302:def mtables(R, R86, tag):
325:            Bs = mdl.apply(f, gs, s, use_global=False)[0]; Bg = mdl.apply(f, gs, s, use_single=False)[0]
345:def local(R, R86):
382:def timing(R, R86):
394:    def tm(fn, n=10):
424:def ladder(R, R86):
RUN [Read own92a gen, objective, train, local] cd /c/Projects/ReST-GI/experiments && sed -n 79,126p ecro_own92a/own92a.py && sed -n 160,236p ecro_own92a/own92a.py && sed -n 345,382p ecro_own92a/own92a.py
  -> def gen(R, R86, split, start, count):
    import fx92
    import pt86a as A
    PT, S = fx92.PT, fx92.S
    TR = S.P['training']
    start, count = int(start), int(count)
    seed = TR['splits'][split]['seed']
    spp = TR['splits'][split].get('target_spp', TR['target_spp']); off = TR['splits'][spli ...[cut]... type(np.float64)[sel]; qq = q[sel]
                o[str(m)]['Bq_vs_q_rel_l1'] = float(np.abs(Bq - qq).sum() / max(qq.sum(), 1e-30)); o[str(m)]['Bq_vs_q_flux'] = float(Bq.sum() / max(qq.sum(), 1e-30))
            res[tag][name] = o
            print('local', tag, name, {m: {k: round(v, 3) for k, v in x.items() if k in ('fro', 'peak', 'mass', 'Bq_vs_q_rel_l1', 'Bq_vs_q_flux')} for m, x in o.items()}, flush=True)
    jd(res, os.path.join(R, 'results', 'local92a.json'))


# ------------------------------------------------------------------------------------------------ runtime
def timing(R, R86):
RUN [List worklogs and grep avoidable/passivity claims] ls docs/worklogs | sed -n '/^8[6-9]\|^9/p' && grep -n -i "avoidable\|passiv\|albedo" docs/worklogs/9[4-6]*_WORKLOG_*.md docs/worklogs/9*A_*.md 2>/dev/null | cut -c1-420 | head -60
  -> 86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md
86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md
89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_AND_DISCRIMI ...[cut]... tors and M64 are passive (≤ 0.96).
docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:143:- **Learned sender (A).** On CK2 it is a large error source in delivery: 0.52× / 1.63× flux and a misplaced footprint. It is seed-signed, as in WL94C/94D, and non-passive. It is far from M64 at the field level, and slightly worse than the rank-16 bounds in rel L1 but not in peak. **Avoidable within rank 16 on this workload?** Not 
docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:161:  | arm | passive | flux vs A4 | map vs A4 | ROI E_H vs A4 |
RUN [List worklogs and grep avoidable/passivity claims] cd /c/Projects/ReST-GI && ls docs/worklogs | sed -n '/^8[6-9]\|^9/p' && grep -n -i "avoidable\|passiv\|albedo" docs/worklogs/9[4-6]*_WORKLOG_*.md docs/worklogs/9*A_*.md 2>/dev/null | cut -c1-420 | head -60
RUN [Read WL95A worklog key sections] sed -n 1,60p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-900; sed -n 110,160p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-1200
  -> # Worklog — WL95A: Sharp Sender Representability & Flight Attribution Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL95A — Sharp Sender Representability & Flight Attribution Smoke" (user). It is a suffix of WL95 and a bounded, learning-free attribution batch.
- **Proposition tested.** "The ...[cut]... t with WL95's W94-weighted objectives (A3) or flux-Frobenius weights (A1). No grazing-specific bound exists.
- **Evidence for an actual angular representation floor.** On this workload, yes for the **binned outgoing sender table**: its expected delivery is biased independently of sampling, and the exact-direction evaluation closes the gap. This is evidence about the patch × 64-bin carrier for a grazing sharp lobe. It is not a statement about other bin layouts, adaptive bases or other scenes.

## E. Architecture Verdict (predeclared rules)

- **RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD.**
RUN [Read WL95A worklog key sections] cd /c/Projects/ReST-GI && sed -n 1,60p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-900; sed -n 110,160p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-1200
RUN [Read W consumers in b94.py and s94.py] cd /c/Projects/ReST-GI/experiments && sed -n 40,75p ecro_exp94b/b94.py && echo ---- && sed -n 100,200p ecro_s94d/s94.py && echo ---- && sed -n 270,330p ecro_s94d/s94.py
  -> # ------------------------------------------------------------------------------------------------ Stage A: split re-render
def agen(R, R86, which, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = C94.sel_files(R, 'train'); asg = R94.assignments(len(files), ' ...[cut]...           w = sel.float().view(1, -1, 1, 3)
            o[nm] = [float((X * w).sum()), float((T_ * w).sum())]
        return o
    with torch.no_grad():
        for split, assign in (('train', 'C'), ('val', 'orig')):
            its = R94.mix_items(R, R86, 'train', 'C') if split == 'train' else C94.load_items(R, R86, 'val', 'orig')
            acc = {k: 0.0 for k in ('l1', 'm', 'l', 'pl', 'pm', 'dl1', 'dm', 'dl', 'bl1', 'bm')}
            op = {'train_bins': [0.0, 0.0], 'heldout_bins': [0.0, 0.0]}
            for it in its:
                gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
RUN [Read w96a.py header and primitives] cd /c/Projects/ReST-GI/experiments && sed -n 1,130p ecro_util96a/w96a.py
  -> """WL96A (protocol ecro_util96a): learned higher-order transport utility ceiling on CK2 'none' (learning-free diagnostics).
Production code (pt86 / pt86a / ecro86 / model86 / quad87 / xport87 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python w96a.py stage0 <R> <R95A>      ...[cut]... return op, torch.as_tensor(pm[..., None].astype(np.float32), device=X.dev), st


def tail(op, mp, L2):
    """e3_p_exp = T_p(L2 + L3 + L4), e3_o_exp = T_o(L2 + L3 + L4) (last gather split by the material of s1); orders for the tests."""
    import torch
    import xport87 as X
    L2 = torch.as_tensor(np.asarray(L2, np.float32), device=X.dev)
    L3 = op.apply(L2); L4 = op.apply(L3); L5 = op.apply(L4)
    g = op.gather(L2 + L3 + L4)
    tp = op.scatter(g * mp); to = op.scatter(g * (1.0 - mp))
    return tp.double().cpu().numpy(), to.double().cpu().numpy(), (L3, L4, L5)


def realizations(ctx):
RUN [Read w96a.py tails and Stage 1 analysis] cd /c/Projects/ReST-GI/experiments && sed -n 130,420p ecro_util96a/w96a.py
  -> def realizations(ctx):
    import fx92
    import xport87 as X
    E87 = fx92.E87
    yield 'deployed', X.realization_us(86001), E87['seed_ray']
    for d in range(R_POOL):
        yield f'pool{d}', X.realization_us(960000 + d), E87['seed_ray'] + 1000 * (d + 1)


# ---------------------------------- ...[cut]... ------------ Stage 2 (conditional)
def fit_lights(geom, n=160, seed=96200):
    import scene86 as S
    rng = np.random.default_rng(seed)
    ck = np.array(MT.SC['lights'][0]['pos'])
    out = []
    while len(out) < n:
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, geom) and np.linalg.norm(np.array(pos) - ck) >= 0.3:
            out.append([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}])
    return out


def s2ref(R, i0, i1):
    PT, S, A, FX, ctx, lt = setup(R)
    L = fit_lights(MT.SC['geom'], 8 if DEV else 160)
RUN [Dump WL96A results] python - <<'EOF'
import json
f=json.load(open('results/ecro_util96a/final.json')); 
def show(o,d=0,maxd=2):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<maxd: print(' '*d+k+':'); show(v,d+1,maxd)
            else: print(' '*d+k+':',str(v)[:300])
show(f,0,2)
s1=json.load(open('results/ecro_util96a/s1.json')); print(json.dumps(s1['decision'],indent=0))
for k in ('F|ROI_T','Z|ROI_T','XD|ROI_T','XP|ROI_T','XOD|ROI_T','XOP|ROI_T','XSD|ROI_T',
  -> stage0: True
S1: S1_NO_HEADROOM
S1_ext: False
S2: None
accuracy_confirmed: False
cost_met: False
U:
 1.0:
  LAM24: {'edit': 0.9321302015681585, 'source': 1.040492233086946}
  LAM64: {'edit': 0.9209615109936242, 'source': 1.038301742024448}
  ZERO: {'edit': 1.038028573853174, 'source': 1.04 ...[cut]... _l1': 0.0411, 'half_vs_half_rel_l1': 0.0579}
XD_cross_atob|e3|all {'table_rel_l1': 0.0395, 'half_vs_half_rel_l1': 0.0496}
XD_cross_btoa|e3_o|boxA {'table_rel_l1': 0.0438, 'half_vs_half_rel_l1': 0.0629}
XD_cross_btoa|e3_o|all {'table_rel_l1': 0.0394, 'half_vs_half_rel_l1': 0.0498}
XD_cross_btoa|e3_p|boxA {'table_rel_l1': 0.0606, 'half_vs_half_rel_l1': 0.0703}
XD_cross_btoa|e3_p|all {'table_rel_l1': 0.253, 'half_vs_half_rel_l1': 0.2002}
XD_cross_btoa|e3|boxA {'table_rel_l1': 0.0409, 'half_vs_half_rel_l1': 0.0578}
XD_cross_btoa|e3|all {'table_rel_l1': 0.0389, 'half_vs_half_rel_l1': 0.0496}
RUN [Dump WL96A results] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
f=json.load(open('results/ecro_util96a/final.json')); 
def show(o,d=0,maxd=2):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<maxd: print(' '*d+k+':'); show(v,d+1,maxd)
            else: print(' '*d+k+':',str(v)[:300])
show(f,0,2)
s1=json.load(open('results/ecro_util96a/s1.json')); print(json.dumps(s1['decision'],indent=0))
for k in ('F|ROI_T','Z|ROI_T','XD|ROI_T','XP|ROI_T','XOD|ROI_T
RUN [Read CK2 scene definition] cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -60 && python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(P['fixture']['panel'],P['fixture']['sphere'],P['fixture']['camera'])"
  -> {
 "tag": "CK2",
 "targets": [
  "A"
 ],
 "light_set": "LCK2",
 "camera": {
  "origin": [
   0.3,
   -1.6,
   1.3
  ],
  "target": [
   -1.3,
   0.45,
   1.0
  ],
  "fov_deg": 32.0,
  "res": 512
 },
 "geom": {
  "A": {
   "c": [
    -1.08,
    0.6,
    1.0
   ],
   "h": [
    0.2,
    0.45,
    0.45
   ],
   "yaw": 0.15
  },
  "B": {
   "c": [
    1.4,
    1.5,
    0.5
   ],
   "h": [
    0.3,
    0.25,
    0.5
   ],
   "yaw": 0.0
  }
 },
 "lights": [
  {
   "pos": [
    -1.45,
    -0.6,
    1.7
   ],
   "rgb": [
    30.0,
    30.0,
    30.0
   ]
{'center': [-1.5, 0.3, 1.0], 'half': [0.6, 0.7], 'normal': '+x', 'material': 'S1 (WL85 fixture material 1, test83_hard #97)'} {'center': [0.9, 0.4, 0.4], 'radius': 0.4, 'material': 'S2 (WL85 fixture material 2, test83_hard #41)'} {'origin': [0.4, -6.2, 1.7], 'target': [0.0, 0.5, 1.2], 'fov_deg': 40.0, 'res': 512}
RUN [Read CK2 scene definition] cd /c/Projects/ReST-GI && cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -60 && python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(P['fixture']['panel'],P['fixture']['sphere'],P['fixture']['camera'])"
RUN [Write and run local numpy check A1] mkdir -p /c/Projects/ReST-GI/experiments/audit97/A /c/Projects/ReST-GI/results/audit97/A && cd /c/Projects/ReST-GI && cat > experiments/audit97/A/a1_bins_m64_deliv.py <<'EOF'
"""WL97 audit, Track A, check A1 (numpy only, local).
(1) bin layout / weights: true layout from scene86 (NZ x NPHI), exact per-bin cos-weighted solid angle, an94.W as coded;
(2) M64 (S1) stored in results/ecro_attr95a/ops95a.npz:A4 : directional albedo under W_T, reciprocity symmetry, unconstrained rank bounds;
(3) WL95A d
  -> Exit code 2
/usr/bin/bash: -c: line 96: unexpected EOF while looking for matching `''
WRITE c:\Projects\ReST-GI\experiments\audit97\A\a1_bins_m64_deliv.py
RUN [Run local numpy check A1] python experiments/audit97/A/a1_bins_m64_deliv.py
  -> {
 "NZ": 4,
 "NPHI": 16,
 "NB": 64,
 "DOMEGA": 0.09817477042468103,
 "WT_equals_exact_bin_integral_max_abs": 0.0,
 "sum_WT": 3.141592653589793,
 "sum_W94": 3.141592653589793,
 "W94_over_WT_by_ring_first_half_azimuth": [
  0.5,
  0.8333333333333334,
  0.9,
  0.9285714285714286
 ],
 "W94 ...[cut]... .012, 1.001, 0.977, 1.024), 'sphere_ER': (0.809, 0.019, 0.802, 0.242, 0.453), 'all_lambert': (0.408, 0.002, 0.392, 0.859, 0.889), 'all_non_panel': (0.537, 0.003, 0.524, 0.884, 0.916)}
A0_s8602 {'boxA': (1.63, 0.014, 1.534, 0.965, 0.996), 'boxA_panel_face': (1.55, 0.015, 1.387, 0.811, 0.821), 'lambert_walls_ceiling': (0.53, 0.005, 0.47, 1.045, 1.114), 'boxB': (1.918, 0.049, 5.257, 0.958, 1.477), 'floor_ER': (1.289, 0.027, 1.421, 0.737, 0.885), 'sphere_ER': (1.88, 0.035, 1.916, 0.997, 1.071), 'all_lambert': (1.008, 0.007, 0.942, 1.01, 1.054), 'all_non_panel': (1.07, 0.008, 1.047, 0.951, 1.016)}
WRITE c:\Projects\ReST-GI\experiments\audit97\A\a1_bins_m64_deliv.py
RUN [Run local numpy check A1] cd /c/Projects/ReST-GI && python experiments/audit97/A/a1_bins_m64_deliv.py
WRITE c:\Projects\ReST-GI\experiments\audit97\A\er_np.py
WRITE c:\Projects\ReST-GI\experiments\audit97\A\a2_m64_er.py
RUN [Run numpy ER port vs M64 check] time python experiments/audit97/A/a2_m64_er.py
  -> {
 "mats86_sha256_16": "d3c5be854d80665b",
 "convention": {
  "lambert_eval_equals_rho_over_pi_cos_wo_max_rel": 1.702989871343874e-08,
  "reciprocity_fr_max_rel": {
   "0": 5.962460040001899e-15,
   "1": 4.004181641914753e-15,
   "2": 4.847118422465765e-15
  }
 },
 "timing_s_m64": 122.5297 ...[cut]... olid_angle_mean_operator_rel_l1": 0.07340003870040832,
  "ring0_row_flux_ratio_solid_angle_mean_over_true": 1.7606021884227292
 },
 "flux_conservation_mat0": {
  "physical_albedo_max": 0.5290747621591597,
  "table_albedo_single_direction_max": 0.5218467275416103,
  "ratio_min": 0.9855343955138098,
  "ratio_max": 1.049892399926281
 },
 "flux_conservation_mat2": {
  "physical_albedo_max": 0.6766840073478599,
  "table_albedo_single_direction_max": 0.6516674073913561,
  "ratio_min": 0.9589272249850339,
  "ratio_max": 1.1413061330204735
 }
}

real	3m0.764s
user	0m0.015s
sys	0m0.000s
WRITE c:\Projects\ReST-GI\experiments\audit97\A\a2_m64_er.py
RUN [Run numpy ER port vs M64 check] cd /c/Projects/ReST-GI && time python experiments/audit97/A/a2_m64_er.py
RUN [Read diag92 local_matrices and fx92] cd /c/Projects/ReST-GI/experiments && sed -n 100,200p ecro_attr92/diag92.py && cat ecro_attr92/fx92.py | head -60
  -> eps = 1e-12
    sc = math.sqrt(max(V.mean(), 1e-30) / r)
    for t in range(restarts):
        rng = np.random.default_rng([seed, t])
        W = rng.random((V.shape[0], r)) * sc + eps; H = rng.random((r, V.shape[1])) * sc + eps
        for _ in range(iters):
            H *= (W.T @ V) / (W.T @ W @  ...[cut]... e, P92['e4']['sources']['E3b']).items():
        FX['light_sets'][k] = v
    return name


def cfg_ls(tag):
    """'E2_T1' or 'E1_T1_LMIX' -> (geometry, light-set name, edit name, cfg)."""
    p = tag.split('_')
    e, t = p[0], p[1]
    return FX[e][t], (p[2] if len(p) > 2 else FX[e]['light_set']), e, f'{e}_{t}'


def mask_file(R, R86, tag):
    p = tag.split('_')
    name = p[0] + ('_' + p[2] if len(p) > 2 else '')
    for root in (R, R86):
        fn = os.path.join(root, 'results', 'masks', name + '.npz')
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(name)
RUN [Block comparison and quadrature convergence check] python -c "
import json;P=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'));print(P['explicit'])" && cat > experiments/audit97/A/a2b_convergence.py <<'PYEOF'
"""WL97 audit, Track A, check A2b: (i) block comparison mine vs stored M64 by (ring_in, ring_out); (ii) convergence of my quadrature for grazing
incident directions (n x n strata per outgoing bin). numpy only."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = 
  -> {'M_ray': 65536, 'seed_ray': 87001, 'M_dense': 16384, 'seed_dense': 87002, 'dense_cells': '8 x 8 strata per incident bin (4,096 cells, cell centres)', 'P': 16, 'Mq': 1024, 'seed_q': 87003, 'seed_c3': 87004, 'orders': 4, 'not_imposed': 'no renormalisation, clamping or energy correction of any table'} ...[cut]... 77695
   }
  },
  "0.4": {
   "24": {
    "albedo_true": 0.7046204484493142,
    "table_flux": 0.7105030146120611,
    "ring0_ring0_sum": 5.325885123216865
   },
   "48": {
    "albedo_true": 0.7046080786406755,
    "table_flux": 0.7105093999450842,
    "ring0_ring0_sum": 5.32631594512943
   },
   "96": {
    "albedo_true": 0.7046064941796951,
    "table_flux": 0.7105123543729558,
    "ring0_ring0_sum": 5.326460186426664
   },
   "192": {
    "albedo_true": 0.7046059564000606,
    "table_flux": 0.7105129744228013,
    "ring0_ring0_sum": 5.326487132185826
   }
  }
 }
}
RUN [Block comparison and quadrature convergence check] cd /c/Projects/ReST-GI && python -c "
import json;P=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'));print(P['explicit'])" && cat > experiments/audit97/A/a2b_convergence.py <<'PYEOF'
"""WL97 audit, Track A, check A2b: (i) block comparison mine vs stored M64 by (ring_in, ring_out); (ii) convergence of my quadrature for grazing
incident directions (n x n strata per outgoing bin). numpy only."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.a
RUN [Inspect server run directories and GPU state] ssh -o ConnectTimeout=20 LabServer63 'ls ~ | grep rest_ | head -50; ls ~/rest_x97 2>/dev/null; cat ~/rest_x95a/env.sh; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; ls ~/rest_x96a; ls ~/rest_x95a/results | head -30'
  -> rest_8dna_baseline
rest_arch77a
rest_ato74
rest_ato74_design
rest_ato74_dev
rest_attrib71b
rest_b4rep
rest_closure80d
rest_cmp72
rest_cmp72_dev
rest_compiled80c
rest_compiled81
rest_compiled81_dev
rest_compiled82
rest_compiled82_dev
rest_compiled82_probe
rest_compiled83
rest_compiled83_dev
rest_comp ...[cut]... 5_look
rest_orbit80
rest_orbit80_dev
rest_rank75
rest_rank75L
rest_rank75_dev
rest_rank75_scratch
rest_rel88
rest_rel88dev
rest_rel89
B
C1
C2
D
E1
E2
E3
E4
MAP
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
2596 MiB, 12288 MiB, 51 %
chain.out
chain.status
code
data
env.sh
figw3_posthoc.py
results
an.json
deliv.npz
eval95a
fields
lists
logs
main.json
ops95a.npz
pre
renders
s0.json
senders.npz
tables
wl94d
wl95_calc
RUN [Inspect server code checkout and WL96A results layout] ssh -o ConnectTimeout=20 LabServer63 'ls ~/rest_x96a/code; ls ~/rest_x96a/code/experiments | head -80; ls ~/rest_x96a/code/experiments/ecro86/data; ls ~/rest_x96a/results ~/rest_x96a/results/fields ~/rest_x96a/results/tables ~/rest_x96a/results/pre | head -60; cat ~/rest_x96a/code/experiments/ecro_util96a/run96a.sh | head -60; mkdir -p ~/rest_x97/A && ls -la ~/rest_x97/A'
  -> AGENTS.md
COMMIT
experiments
8dna_deformation_replication
baseline_sup67
dynamic_transport_failure
dynamic_transport_validity
ecro86
ecro86a
ecro_attr92
ecro_attr95a
ecro_cf92b
ecro_exp94b
ecro_local93d
ecro_mat94
ecro_mat94r
ecro_own92a
ecro_q94c
ecro_s94d
ecro_stage93
ecro_stage93a
ecro_stage93b
e ...[cut]... (rule: S1_UNRESOLVED)"
  pool 3 FX0 "$A fext $R 0 $DEV" FX1 "$A fext $R 1 $DEV" FX2 "$A fext $R 2 $DEV" FX3 "$A fext $R 3 $DEV" FX4 "$A fext $R 4 $DEV" FX5 "$A fext $R 5 $DEV"   # CPU-bound accumulation (probe ~3 % GPU)
  run T2 "$A tails $R --ext $DEV"
  renders e "$O/lists/r96a_ext.txt"
  run N2 "$A an $R --ext $DEV"
  S1=$(grep '^S1 ' $O/logs/N2.log | tail -1 | cut -d' ' -f2); log "S1_EXT $S1"
fi
if [ "$S1" = S1_HEADROOM ] || [ -n "$DEV" ]; then                # dev exercises the Stage 2 path regardless
total 8
drwxr-xr-x  2 nd nd 4096 Nov 15 17:59 .
drwxr-xr-x 12 nd nd 4096 Nov 15 17:59 ..
WRITE c:\Projects\ReST-GI\experiments\audit97\A\a3_ck2_delivery.py
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
