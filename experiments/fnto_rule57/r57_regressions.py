"""Historical focused regressions, with every newly written output isolated in WL57."""
import os,sys,subprocess,shutil
from pathlib import Path
from r57_data import ROOT,OUT,dump
import r57_geometry as RG

def main():
    out=OUT/'tests';out.mkdir(parents=True,exist_ok=True);base=RG.HERE.parent
    cases=[('wl53',base/'phys_ref53/tests/test_p53.py'),('wl54',base/'fnto_c54/tests/test_c54.py'),('wl55',base/'fnto_k55/tests/test_k55.py'),('wl56',base/'fnto_sel56/tests/test_s56.py')]
    records=[]
    for name,p in cases:
        with open(out/f'{name}.log','w') as f:r=subprocess.run([sys.executable,str(p),str(out/f'{name}.json')],stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(name=name,returncode=r.returncode));print(records[-1],flush=True)
    # WL54 runner expects this short alias; no historical file is changed.
    md=out/'masks';md.mkdir(exist_ok=True)
    shutil.copyfile(Path.home()/'rest_wl54/out/cycles/ccam_floor_lam_L.json',md/'ccam_floor_lam.json')
    import importlib.util
    p=base/'fnto_c54/tests/test_c54.py';sp=importlib.util.spec_from_file_location('historical_masks',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    ok=m.comparison_masks(md);dump(dict(passed=bool(ok),results=m.RES),out/'wl54_actual_masks.json');records.append(dict(name='wl54_actual_masks',returncode=0 if ok else 1))
    dump(records,out/'regressions.json');assert all(r['returncode']==0 for r in records)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
