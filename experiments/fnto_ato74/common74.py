"""WL74 shared paths. Reuses WL68/71/72/73 modules unchanged; ~/rest_repr68, ~/rest_cover71, ~/rest_cmp72, ~/rest_src73 are read-only
inputs (WL73 references, fit targets, runs). Evidence root: ~/rest_ato74 (R74_ROOT)."""
import os,sys,json
from pathlib import Path
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R74_ROOT',HOME/'rest_ato74'))
os.environ['R73_ROOT']=str(HOME/'rest_src73')   # WL73 root (read-only)
p=str(HERE.parent/'fnto_src73')
if p not in sys.path:sys.path.insert(0,p)
import common73 as C73  # noqa: E402  (adds fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R68=C73.R68;INP68=C73.INP68;R71=C73.R71;R72=C73.R72;R73=C73.ROOT
PROTO=HERE/'protocol/ato74_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}

def environment():
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_ato74'

def seed(g,slot,a,rep):
    """WL74 Stage 4 confirmation seeds 743/744xxxxxx (rest 0, pose 1); disjoint from WL66-73."""
    return 743000000+g*1000000+slot*10000+a*100+rep
