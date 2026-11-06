"""WL76 shared paths. Reuses the WL68/71/72/73/74/75 modules unchanged; their roots are read-only inputs:
~/rest_repr68, ~/rest_cover71, ~/rest_cmp72, ~/rest_src73 (S16-R targets, WL73 references, AR controls), ~/rest_ato74 (G2 basis),
~/rest_rank75 (A75 controls; copied from the local WL75 root). Evidence root: ~/rest_fia76 (R76_ROOT)."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R76_ROOT',HOME/'rest_fia76'))
os.environ.setdefault('R75_ROOT',str(HOME/'rest_rank75'))
p=str(HERE.parent/'fnto_rank75')
if p not in sys.path:sys.path.insert(0,p)
import common75 as C75  # noqa: E402  (adds fnto_ato74, fnto_src73, fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R75=C75.ROOT;R74=C75.R74;R73=C75.R73;R72=C75.R72;R71=C75.R71;R68=C75.R68;INP68=C75.INP68
PROTO=HERE/'protocol/fia76_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}
TRAIN=('rest','pt0')

def environment():
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_fia76'

def seed(kind,g,slot,a,rep):
    """WL76 seeds 761-769xxxxxx (disjoint from WL66-75): physics validation 761, first-event quadrature 762, confirmation 766/767."""
    base={'phys':761000000,'quad':762000000,'conf':766000000}[kind]
    return base+g*1000000+slot*10000+a*100+rep
