import json, sys, numpy as np
z = np.load('C:/Projects/ReST-GI/results/ecro_attr95a/deliv.npz')
ref = 0.5*(z['ref_a']+z['ref_b'])[1130:1184]
d = json.load(open(sys.argv[1]))
def m(a, r):
    a, r = np.asarray(a), np.asarray(r)
    return f"ratio {a.sum()/r.sum():.3f} relL1 {np.abs(a-r).sum()/r.sum():.3f} corr {np.corrcoef(a,r)[0,1]:.3f} pf {a[:9].sum()/r[:9].sum():.3f}/{np.abs(a[:9]-r[:9]).sum()/r[:9].sum():.3f}"
for ln, res in d['res'].items():
    print('==', ln, 'HP', d['HP'], 'NR', d['NR'])
    R = np.asarray(res['REF'])
    if ln == 'CK2':
        print('REF vs pathtraced :', m(R, ref))
        for k in ['X1', 'X1c', 'M64_spool', 'q256']:
            print(f'pooled {k:9s} vs PT :', m(z['draws|'+k].mean(0)[1130:1184], ref))
    for k, v in res.items():
        print(f'{k:12s} vs REF:', m(v, R), ('| vs PT: ' + m(v, ref)) if ln == 'CK2' else '')
