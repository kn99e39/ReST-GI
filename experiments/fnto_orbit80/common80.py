"""WL80 shared paths. Imports the WL68-WL79 modules unchanged (fnto_sym79 for the exact symmetry contract, fnto_srcid78 for W+UV-Q and the
WL78 evaluation machinery); their server roots are read-only inputs: ~/rest_repr68, ~/rest_cover71, ~/rest_cmp72, ~/rest_src73 (S16-R fit
targets, WL73 references, AR), ~/rest_ato74 (G2 basis), ~/rest_rank75 (A75), ~/rest_rank75L, ~/rest_srcid78 (Q78), ~/rest_arch77a (WL77A Part
0: development only), ~/rest_sym79 (WL79 Stage 0 / 1 records; the WL79 Stage 5 population was never referenced or scored).
Evidence root: ~/rest_orbit80 (R80_ROOT)."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R80_ROOT',HOME/'rest_orbit80'))
for k in ('R78_ROOT','R79_ROOT'):os.environ.pop(k,None)     # earlier batches' modules always resolve their own (read-only) roots
for d in ('fnto_sym79','fnto_srcid78'):
    p=str(HERE.parent/d)
    if p not in sys.path:sys.path.insert(0,p)
import common79 as C79  # noqa: E402  (-> common78 -> common75 -> ... ; adds the WL68-WL75 directories)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R79=C79.ROOT;R78=C79.R78;R75=C79.R75;R75L=C79.R75L;R77A=C79.R77A;R74=C79.R74;R73=C79.R73;R72=C79.R72;R71=C79.R71;R68=C79.R68;INP68=C79.INP68
PROTO=HERE/'protocol/orbit80_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}

def environment():
    import common78
    common78.environment=lambda:C68.environment(ROOT/'repo')   # WL78 / WL79 modules use this repository copy
    C79.environment=lambda:C68.environment(ROOT/'repo')
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_orbit80'

def load(p,mmap=False):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
