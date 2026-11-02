"""WL71 shared paths. Reuses the WL68 modules (family, estimator, evaluator) unchanged; WL68 server files are read-only inputs.
Server layout: ~/rest_cover71 (this batch, evidence root), ~/rest_repr68 (WL68 inputs, runs and held-out bundles; never written)."""
import os,sys,json
from pathlib import Path
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R71_ROOT',HOME/'rest_cover71'))
R68=Path(os.environ.get('R71_R68',HOME/'rest_repr68'))
os.environ['R68_ROOT']=str(R68)          # common68.INP -> WL68 inputs (read-only)
sys.path.insert(0,str(HERE.parent/'fnto_repr68'))
from common68 import sha,tensor_sha,write_new,save_new,NBIN,PT0,verify_sources  # noqa: E402
import common68 as C68  # noqa: E402
INP68=R68/'inputs'
CFG=json.loads((HERE/'protocol/cover71_v1.json').read_text()) if (HERE/'protocol/cover71_v1.json').exists() else {}
TRAIN_GEOMS=('rest','pt0')

def environment():
    """WL59 runtime/reference modules from this batch's deployed repository copy (ROOT/repo)."""
    return C68.environment(ROOT/'repo')

def seed(g,slot,a,rep):
    """WL71 reference seeds: disjoint from WL66 (66xxxxxxx), WL68 (683xxxxxx/688xxxxxx) and dev smoke (719xxxxxx)."""
    return 713000000+g*1000000+slot*10000+a*100+rep
