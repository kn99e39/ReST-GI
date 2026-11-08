"""WL80 Stage 2: oracle representability gate (LABELLED ORACLES). The WL79 Stage 1 oracle code (oracle79.oracle / closure, unchanged
arithmetic: per-source T, float64 FISTA 3,000 + 3,000-iteration check, fitting replicas 0-3, scoring held-out replicas 4-7, WL79 operations
r90 / r180 / mx / sw / id+z / r90+z, groups ALL / BND / INT, eval68 estimators) with only the frozen basis swapped.
  G0 = the frozen G2 basis: the WL79 oracle record (same code, data and basis; sha256 recorded) is reused, not recomputed.
  G1 = the selected orbit-balanced basis (basis80.select).  Non-selected restarts: descriptive, operations r90 and r180 only.
    python oracle80.py run G1 | run B80_r<k> | decide"""
import sys,os,json
import torch
from common80 import CFG,ROOT,R79,environment,load,write_new,sha
RG=environment()
import ato74 as AT
import oracle79 as O9
S2=CFG.get('stage2',{})

def basis_of(tag):
    f=ROOT/'basis'/('G1.pt' if tag=='G1' else f'{tag}.pt');b=load(f)
    return b['in_logits'].double(),b['out_logits'].double(),dict(id=tag,sha256=sha(f))

def run(tag):
    d=ROOT/'stage2'/tag;d.mkdir(parents=True,exist_ok=True)
    O9.ROOT=d;O9.basis64=lambda:basis_of(tag)
    if tag!='G1':O9.S1=dict(O9.S1,ops=S2['descriptive_ops'])
    if not (d/'stage1/closure.json').exists():(d/'stage1').mkdir(exist_ok=True);O9.closure()
    O9.oracle()

def decide():
    out=ROOT/'stage2/gate.json'
    if out.exists():raise FileExistsError('immutable')
    f0=R79/'stage1/oracle.json';assert sha(f0)==S2['G0_record_sha256'],'WL79 G0 record changed'
    r0=json.loads(f0.read_text());r1=json.loads((ROOT/'stage2/G1/stage1/oracle.json').read_text());gc=S2['gate']
    hi=lambda r,k,m:max(c['hi'] for c in r['groups'][k]['pooled'][m]);rows={};fail=[]
    for gn in gc['groups']:
        for m in gc['metrics']:   # (b) original-orientation preservation vs G0
            v=dict(G0=hi(r0,f'id:{gn}',m),G1=hi(r1,f'id:{gn}',m));v['ratio']=v['G1']/v['G0'];v['ok']=v['ratio']<=gc['factor'];rows[f'original:{gn}:{m}']=v
            if not v['ok']:fail.append(('original',gn,m))
        for m,th in gc['wl59'].items():
            if hi(r0,f'id:{gn}',m)<=th and hi(r1,f'id:{gn}',m)>th:fail.append(('original_wl59',gn,m))
        for opn in S2['ops']:
            for m in gc['metrics']:   # (a) WL79 rule on G1; (c) transformed vs the G0 original level
                v=dict(G1_orig=hi(r1,f'id:{gn}',m),G1_trans=hi(r1,f'{opn}:{gn}',m),G0_orig=hi(r0,f'id:{gn}',m),G0_trans=hi(r0,f'{opn}:{gn}',m))
                v['ratio_own']=v['G1_trans']/v['G1_orig'];v['ratio_vs_G0_orig']=v['G1_trans']/v['G0_orig'];v['ok']=v['ratio_own']<=gc['factor'] and v['ratio_vs_G0_orig']<=gc['factor']
                rows[f'{opn}:{gn}:{m}']=v
                if not v['ok']:fail.append((opn,gn,m))
            for m,th in gc['wl59'].items():
                if hi(r1,f'id:{gn}',m)<=th and hi(r1,f'{opn}:{gn}',m)>th:fail.append((opn,gn,f'wl59_{m}'))
    conv=max(abs(v['rel_change']) for v in r1['convergence'].values());sel=json.loads((ROOT/'basis/select.json').read_text())
    s0=json.loads((ROOT/'stage0/stage0.json').read_text());passed=not fail and s0['valid'] and conv<=gc['convex_change_max']
    cls=('RANK-16 ORBIT REPRESENTABILITY SUPPORTED' if passed else 'ORBIT-BALANCED BASIS INADEQUATE (stop before learned training)')
    if not s0['valid']:cls='STAGE 0 CONTRACT INVALID (stop)'
    write_new(out,dict(rows=rows,failures=fail,passed=passed,classification=cls,G1_convergence_max=conv,restart_spread=sel['restart_spread'],selected_restart=sel['selected_restart'],
                       G0_record=str(f0),stage0_valid=s0['valid']))
    print('STAGE2_GATE',cls,fail[:12],conv,flush=True)

if __name__=='__main__':
    {'run':run,'decide':decide}[sys.argv[1]](*sys.argv[2:]);sys.stdout.flush();os._exit(0)
