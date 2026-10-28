"""Pre-result CPU reporting correction: primary fine fidelity cannot be ignored."""
import sys,os,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r57_report import quality_noninferior

if __name__=='__main__':
    d={'poses':{'p':{'metrics':{'coarse':{'mean':[.05]*3},'fine16':{'mean':[.05]*3}}}}};r=copy.deepcopy(d)
    assert quality_noninferior(d,r)
    r['poses']['p']['metrics']['fine16']['mean'][1]=.08;assert not quality_noninferior(d,r)
    r['poses']['p']['metrics']['fine16']['mean'][1]=.01;assert quality_noninferior(d,r)
    r['poses']['p']['metrics']['coarse']['mean'][2]=.08;assert not quality_noninferior(d,r)
    Path(sys.argv[1]).write_text(json.dumps(dict(passed=4,total=4,scope='Same .02 criterion enforced at BOTH preregistered coarse and fine16 primary action resolutions; substantial R improvement is allowed.'),indent=1)+'\n')
    sys.stdout.flush();os._exit(0)
