import numpy as np, json, os, sys
H=os.path.expanduser('~')
a=np.load(H+'/rest_x96aprobe/results/tables/T96A.npz'); b=np.load(H+'/rest_x96a/results/tables/T96A.npz')
print('keys equal', sorted(a.files)==sorted(b.files), len(a.files))
worst=[]
for k in a.files:
    x=a[k].astype(np.float64); y=b[k].astype(np.float64)
    if x.shape!=y.shape: print('shape', k, x.shape, y.shape); continue
    d=np.abs(x-y); worst.append((float(d.sum()/max(np.abs(y).sum(),1e-30)), float(d.max()/max(np.abs(y).max(),1e-30)), k, bool(np.array_equal(a[k],b[k]))))
worst.sort(reverse=True)
for w in worst[:8]: print('relL1 %.3e relmax %.3e %s bitwise=%s'%w)
print('bitwise-equal keys', sum(w[3] for w in worst), 'of', len(worst))
a=np.load(H+'/rest_x96aprobe/results/tails.npz'); b=np.load(H+'/rest_x96a/results/tails.npz')
for k in a.files:
    if k in b.files and a[k].shape==b[k].shape and a[k].dtype.kind=='f':
        x=a[k].astype(np.float64); y=b[k].astype(np.float64)
        print('tails.npz', k, 'relL1 %.3e'%(np.abs(x-y).sum()/max(np.abs(y).sum(),1e-30)), np.array_equal(a[k],b[k]))
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,p+'/'+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,p+'/'+str(i))
    else: yield p,o
ja=dict(walk(json.load(open(H+'/rest_x96aprobe/results/tails.json')))); jb=dict(walk(json.load(open(H+'/rest_x96a/results/tails.json'))))
diffs=[(abs(ja[k]-jb[k])/max(abs(jb[k]),1e-12),k,ja[k],jb[k]) for k in ja if k in jb and isinstance(ja[k],float) and isinstance(jb[k],float) and ja[k]!=jb[k]]
diffs.sort(reverse=True)
print('tails.json float leaves', sum(isinstance(v,float) for v in jb.values()), 'differing', len(diffs))
for d in diffs[:12]: print('  rel %.2e %s probe=%.6g final=%.6g'%d)
ca=dict(walk(json.load(open(H+'/rest_x96aprobe/results/cost_s1.0.json')))); cb=dict(walk(json.load(open(H+'/rest_x96a/results/cost_s1.0.json'))))
for k in cb:
    if any(s in k for s in ('U_','ratio','wall_ms','edit_ms','source_ms','tail')) and isinstance(cb[k],(int,float)) and k in ca and len(k)<70:
        print('  cost', k, 'probe=%.5g final=%.5g'%(ca[k],cb[k]))
