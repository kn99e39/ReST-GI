"""WL94 POST HOC (descriptive, labelled; after the frozen labels): untextured box-A error split face x bin group, signed bias, S / G parts,
and the e2_p (panel-sender) part, for frozen own s8601 / s8602 and smoke A / B (s8601)."""
import json, os, sys
import numpy as np
R = sys.argv[1]
sys.path.insert(0, os.path.join(R, 'code', 'experiments', 'ecro_mat94'))
import an94 as AN
F = AN.fref(R); M = AN.masks(R)
T, Tp = M['T'], M['T_panel_face']; pfb = M['panel_facing_bins']
out = {}
for mt, fn in (('own_s8601', 'FZ_own_s8601'), ('own_s8602', 'FZ_own_s8602'), ('A_s8601', 'CD_A_s8601'), ('B_s8601', 'CD_B_s8601')):
    tab = np.load(os.path.join(R, 'results', 'tables', fn + '.npz'))
    o = {}
    for arm in ('C0', 'C2_srcS1', 'C2_rcv'):
        for v in ('none', 'bx38'):
            for comp, km, kr in (('B', f'B_{arm}', 'B'), ('S', f'S_{arm}', 'S'), ('G', 'G', 'G')):
                X = tab[f'{v}|{km}'].astype(np.float64).sum(2); Fr = F[v][kr].sum(2)
                for fn_, fm in (('panel_face', Tp), ('other_faces', T & ~Tp)):
                    for bn, bm in (('pf_bins', pfb), ('other_bins', ~pfb)):
                        sel = fm[:, None] & bm
                        if sel.sum() == 0:
                            continue
                        x, f = X[sel], Fr[sel]
                        o[f'{arm}|{v}|{comp}|{fn_}|{bn}'] = {'rel_l1': float(np.abs(x - f).sum() / f.sum()), 'signed': float((x - f).sum() / f.sum()),
                                                            'cells': int(sel.sum()), 'ref_share_of_B': float(f.sum() / F[v]['B'].sum(2)[sel].sum())}
    # single-flight from the panel sender vs e2_p at panel-face panel-facing bins (C0 and exact receiver)
    for km in ('S_C0_fromS1', 'S_C2_rcv_fromS1', 'S_C0_fromOther', 'S_C2_rcv_fromOther'):
        kr = 'Sp' if 'S1' in km else 'So'
        X = tab[f'none|{km}'].astype(np.float64).sum(2); Fr = F['none'][kr].sum(2)
        for fn_, fm in (('panel_face', Tp), ('other_faces', T & ~Tp)):
            sel = fm[:, None] & np.ones_like(pfb)
            o[f'{km}|{fn_}'] = {'rel_l1': float(np.abs(X[sel] - Fr[sel]).sum() / max(Fr[sel].sum(), 1e-30)), 'signed': float((X[sel] - Fr[sel]).sum() / max(Fr[sel].sum(), 1e-30)),
                               'ref_share_of_S': float(Fr[sel].sum() / F['none']['S'].sum(2)[sel].sum())}
    out[mt] = o
json.dump(out, open(os.path.join(R, 'results', 'an', 'posthoc94.json'), 'w'), indent=1)
print('POSTHOC done')
