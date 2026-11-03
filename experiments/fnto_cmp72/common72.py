"""WL72 shared paths and helpers. Reuses WL68/WL71 modules unchanged (read-only inputs: ~/rest_repr68, ~/rest_cover71, ~/rest_drift_smoke)."""
import os,sys,json
from pathlib import Path
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R72_ROOT',HOME/'rest_cmp72'))
os.environ.setdefault('R71_ROOT',str(HOME/'rest_cover71'))   # fit71/common71 read WL71 fit targets / ports there (read-only)
for d in ('fnto_cover71','fnto_repr68'):
    p=str(HERE.parent/d)
    if p not in sys.path:sys.path.insert(0,p)
import common71 as C71  # noqa: E402
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R68=C71.R68;INP68=R68/'inputs';R71=C71.ROOT
CFG=json.loads((HERE/'protocol/cmp72_v1.json').read_text()) if (HERE/'protocol/cmp72_v1.json').exists() else {}

def environment():
    return C68.environment(ROOT/'repo')

def seed(g,slot,a,rep):
    """WL72 confirmation seeds: 723xxxxxx, disjoint from WL66/68/71 (66x/683/688/713-715) and smoke ranges."""
    return 723000000+g*1000000+slot*10000+a*100+rep
