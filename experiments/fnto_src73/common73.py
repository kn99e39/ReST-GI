"""WL73 shared paths. Reuses the WL68/WL71/WL72 modules unchanged; their server roots are read-only inputs:
~/rest_repr68 (WL66/68), ~/rest_cover71 (WL71 S16, FIN71, D1 runs), ~/rest_cmp72 (WL72 A/B runs, pt0 symmetry images, CONF72),
~/rest_drift_smoke (WL71A). Evidence root: ~/rest_src73 (R73_ROOT)."""
import os,sys,json
from pathlib import Path
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R73_ROOT',HOME/'rest_src73'))
os.environ['R72_ROOT']=str(HOME/'rest_cmp72')     # WL72 root (read-only): A/B runs, geom/pt0v_*, te2 geometry, CONF72 bundles
os.environ['R71_ROOT']=str(HOME/'rest_cover71')   # WL71 root (read-only): ports, fit_targets71, bundles, D1 runs
for d in ('fnto_cmp72',):
    p=str(HERE.parent/d)
    if p not in sys.path:sys.path.insert(0,p)
import common72 as C72  # noqa: E402  (adds fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R68=C72.R68;INP68=C72.INP68;R71=C72.R71;R72=C72.ROOT
PROTO=HERE/'protocol/src73_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}
TRAIN=('rest','pt0')

def environment():
    """WL59 runtime/reference modules from this batch's deployed repository copy (ROOT/repo); te0 registered as in WL57."""
    return C68.environment(ROOT/'repo')

def dev():
    """True outside the evidence root (dev smoke)."""
    return ROOT!=HOME/'rest_src73'

def seed(kind,g,slot,a,rep):
    """WL73 reference seeds: training 733/734xxxxxx (rest 0, pt0 1), confirmation 736/737xxxxxx (rest 0, te0 1) (the dev
    root reuses them with 0.5M photons); disjoint from WL66 (66x), WL68 (683/688), WL71 (713-715/719), WL72 (723-724)."""
    base={'train':733000000,'conf':736000000}[kind]
    return base+g*1000000+slot*10000+a*100+rep
