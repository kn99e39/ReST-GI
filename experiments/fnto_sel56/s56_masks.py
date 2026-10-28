"""Run the historical WL54 mask check with its actual geometry filename; preserve skipped regression record."""
import sys, shutil, importlib.util
from pathlib import Path
import s56 as S

if __name__=='__main__':
    md=S.OUT/'regression_masks';md.mkdir(exist_ok=True)
    shutil.copyfile(Path.home()/'rest_wl54/out/cycles/ccam_floor_lam_L.json',md/'ccam_floor_lam.json')
    p=Path(__file__).resolve().parents[1]/'fnto_c54/tests/test_c54.py'
    spec=importlib.util.spec_from_file_location('historical_mask_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ok=m.comparison_masks(md)
    S.dump(dict(passed=int(ok),total=1,results=m.RES),S.OUT/'tests/wl54_masks_actual.json')
    assert ok and not any('skipped' in x for x in m.RES)
    sys.stdout.flush();__import__('os')._exit(0)
