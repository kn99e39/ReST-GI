"""WL67 W2 analysis (protocol baseline_sup67_v1): teaset common light, PT Full Recompute ladder vs native RNA / 8DNA outputs.

python b67_w2_analyze.py <w2 inputs dir (WL22 copies, sha manifest)> <pt out dir> <analysis out dir>
All arms are scored by the unchanged WL22 cross_backbone_eval.refit_block on the WL22 ROIs and GT mean.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '8dna_deformation_replication'))
import numpy as np  # noqa: E402

LADDER = (1, 4, 16, 64, 256, 1024)
# Inherited WL27 (RTX 5080, results/neural_recompute_cost/v1_d2d6432) and WL22 records; never re-labelled as LabServer69 timings.
INHERITED = dict(
    hardware='RTX 5080 16 GB (personal PC), WL27',
    rebuild_8dna_full_schedule_s=6354, rebuild_8dna_first_rule_checkpoint_envmap_s=2333, rebuild_8dna_common_light_first_rule_s=3810,
    rebuild_rna_full_schedule_s=11648, rna_target_rendering_s=8450.9,
    infer_8dna_1spp_s=0.159, infer_8dna_1024spp_s=100.7, infer_rna_network_s=3.04, infer_rna_inputs_s=5.0, infer_rna_full_call_s=6.8)


def exr(path):
    import ednalib as L
    return L.load_exr(path)


def main():
    import ednalib as L
    import cross_backbone_eval as CB
    from teaset_frozen_eval import load_rois
    inp, pt, out = (Path(a) for a in sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
    L.init_upstream()
    states = ('T0', 'T3')
    gt = {s: (exr(inp / f'frozen/{s}/gt_A.exr'), exr(inp / f'frozen/{s}/gt_B.exr')) for s in states}
    rois = load_rois(inp / 'gt_design', states)
    flat = lambda a: a.reshape(-1, 3)
    res = dict(protocol='baseline_sup67_v1', inherited=INHERITED)

    def score(t0, t3):
        b = CB.refit_block(gt, rois, t0, t3); g3 = flat(0.5 * (gt['T3'][0] + gt['T3'][1]))
        b['T3_whole_image'] = L.pixel_metrics(flat(t3), g3); return b

    arms = {
        'RNA_frozen': (np.load(inp / 'rna/T0_canonical.npy'), np.load(inp / 'rna/T3_canonical.npy')),
        'RNA_refit': (np.load(inp / 'rna/T0_canonical.npy'), np.load(inp / 'rna/refit_T3_current.npy')),
        '8DNA_frozen_WL22': (exr(inp / 'frozen/T0/8dna_attached.exr'), exr(inp / 'frozen/T3/8dna_attached.exr')),
        '8DNA_refit': (exr(inp / 'refit/T0_retrain_at_T0.exr'), exr(inp / 'refit/T3_refit_at_T3.exr')),
    }
    if (pt / '8dna_attached_T3.npy').exists():
        arms['8DNA_frozen_reexecuted_69'] = (np.load(pt / '8dna_attached_T0.npy'), np.load(pt / '8dna_attached_T3.npy'))
    sc = {k: score(np.asarray(a, np.float32), np.asarray(b, np.float32)) for k, (a, b) in arms.items()}
    # reproduction of the WL22 record by this code path (test)
    cb = json.load(open(inp / 'cross_backbone.json'))
    rep = {'RNA_refit': abs(sc['RNA_refit']['refit_T3_interaction_error'] - cb['refit_rna_common_light']['refit_T3_interaction_error']),
           '8DNA_refit': abs(sc['8DNA_refit']['refit_T3_interaction_error'] - cb['refit_8dna_common_light']['refit_T3_interaction_error']),
           'RNA_refit_gain': abs(sc['RNA_refit']['gain'] - cb['refit_rna_common_light']['gain']),
           '8DNA_refit_gain': abs(sc['8DNA_refit']['gain'] - cb['refit_8dna_common_light']['gain'])}
    res['wl22_reproduction_abs_diff'] = rep; assert max(rep.values()) <= 1e-6, rep
    for r in (0, 1):
        for spp in LADDER:
            sc[f'PT|{spp}|r{r}'] = score(np.load(pt / f'pt_T0_spp{spp}_r{r}.npy'), np.load(pt / f'pt_T3_spp{spp}_r{r}.npy'))
    res['scores'] = sc
    # cross-machine check
    m, _ = rois['T3']['interaction']; g3 = flat(0.5 * (gt['T3'][0] + gt['T3'][1])); x = flat(np.load(pt / 'pt_T3_spp4096_xcheck.npy'))
    ab = L.pixel_metrics(flat(gt['T3'][0])[m], flat(gt['T3'][1])[m])['display_mae']; e = L.pixel_metrics(x[m], g3[m])['display_mae']
    ratio = float(x[m].mean() / g3[m].mean()); bound = 1.5 * ab * np.sqrt(1 / 4096 + 1 / 65536) / np.sqrt(2 / 32768)
    res['cross_machine_check'] = dict(display_mae_69_vs_gtmean=e, gt_A_vs_B=ab, bound=float(bound), roi_mean_ratio=ratio,
                                      passed=bool(e <= bound and abs(ratio - 1) <= 0.01))
    # quality matching on the interaction ROI (T3), both realizations
    qm = {}
    for arm in arms:
        target = sc[arm]['refit_T3_interaction_error']; hit = None
        for spp in LADDER:
            if all(sc[f'PT|{spp}|r{r}']['refit_T3_interaction_error'] <= target for r in (0, 1)): hit = spp; break
        qm[arm] = dict(arm_T3_roi_error=target, arm_gain=sc[arm]['gain'], arm_recovers=sc[arm]['recovers'], pt_matched_spp=hit)
    res['quality_matching'] = qm
    tl = [json.loads(l) for l in open(pt / 'timings.jsonl')]
    tim = {}
    for st in states:
        for spp in LADDER:
            q = [t for t in tl if t.get('kind') == 'pt' and t['state'] == st and t['spp'] == spp]
            tim[f'{st}|{spp}'] = dict(update=[t['update_seconds'] for t in q], load=[t['load_seconds'] for t in q], render=[t['render_seconds'] for t in q],
                                      median=float(np.median([t['update_seconds'] for t in q])), peers=[t['peers'] for t in q])
    tim['8dna'] = [t for t in tl if t.get('kind', '').startswith('8dna')]
    res['timing'] = tim
    for arm, q in qm.items():
        if q['pt_matched_spp'] is not None: q['pt_update_seconds_T3'] = tim[f"T3|{q['pt_matched_spp']}"]['median']
    json.dump(res, open(out / 'w2_analysis.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else (bool(o) if isinstance(o, np.bool_) else str(o)))
    with open(out / 'w2_scores.csv', 'w') as f:
        f.write('arm,T3_roi_display_mae,T0_same_surface_mae,error_ratio,gain,recovers,T3_whole_display_mae,T3_whole_psnr_db\n')
        for k, v in sc.items():
            f.write(f"{k},{v['refit_T3_interaction_error']:.6f},{v['T0_model_error_same_surface']:.6f},{v['error_ratio']:.4f},{v['gain']:.4f},{v['recovers']},"
                    f"{v['T3_whole_image']['display_mae']:.6f},{v['T3_whole_image']['display_psnr_db']:.3f}\n")
    print(json.dumps(dict(qm=qm, xcheck=res['cross_machine_check'], repro=rep), indent=1, default=str))


if __name__ == '__main__':
    main(); sys.stdout.flush(); __import__('os')._exit(0)
