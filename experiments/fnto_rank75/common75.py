"""WL75 shared paths. Reuses the WL68/71/72/73/74 modules unchanged; their server roots are read-only inputs:
~/rest_repr68, ~/rest_cover71, ~/rest_cmp72, ~/rest_src73 (S16-R targets, WL73 references, AR controls), ~/rest_ato74 (G0/G2 bases,
Stage 4 population ports4.json, cf1 geometry). Evidence root: ~/rest_rank75 (R75_ROOT)."""
import os,sys,json
from pathlib import Path
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R75_ROOT',HOME/'rest_rank75'))
os.environ['R74_ROOT']=str(HOME/'rest_ato74')     # WL74 root (read-only): bases, ports4.json, inputs/geometry_cf1.pt
p=str(HERE.parent/'fnto_ato74')
if p not in sys.path:sys.path.insert(0,p)
import common74 as C74  # noqa: E402  (adds fnto_src73, fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R74=C74.ROOT;R73=C74.R73;R72=C74.R72;R71=C74.R71;R68=C74.R68;INP68=C74.INP68
PROTO=HERE/'protocol/rank75_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}
TRAIN=('rest','pt0')

def environment():
    """WL59 runtime/reference modules from this batch's deployed repository copy (ROOT/repo)."""
    return C68.environment(ROOT/'repo')

def dev():
    """True outside the evidence root (dev smoke)."""
    return ROOT!=HOME/'rest_rank75'
