"""Local detached orchestration; no agent delegation or remote code upload."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent


def main(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    status=root/'chain.status'
    def log(s):
        with status.open('a',encoding='utf8') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s+'\n')
        print(s,flush=True)
    def command(script,*args):return [sys.executable,'-u',str(HERE/script),str(root),*map(str,args)]
    def run(label,args):
        log('START '+label)
        path=root/(label+'.log')
        with path.open('a',encoding='utf8') as f:
            p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
        log('END '+label+' rc='+str(p.returncode))
        if p.returncode:raise RuntimeError(label)
    log('CHAIN_START')
    try:
        for sp in ['train','val','test']:run('near_gen_'+sp,[sys.executable,'-u',str(HERE/'run100.py'),'gen',str(root),'near',sp])
        nf=(root/'near_train.log').open('a',encoding='utf8')
        near=subprocess.Popen(command('train100.py','near'),stdout=nf,stderr=subprocess.STDOUT,cwd=REPO)
        log('START near_train pid='+str(near.pid))
        for sp in ['train','val','test']:run('sss_gen_'+sp,[sys.executable,'-u',str(HERE/'run100.py'),'gen',str(root),'sss',sp])
        run('sss_train',command('train100.py','sss'))
        rc=near.wait();nf.close();log('END near_train rc='+str(rc))
        if rc:raise RuntimeError('near_train')
        for case in ['near','sss']:
            run(case+'_references',[sys.executable,'-u',str(HERE/'run100.py'),'render',str(root),case])
            run(case+'_predictions',[sys.executable,'-u',str(HERE/'run100.py'),'tables',str(root),case])
        log('TRAIN_AND_RENDER_COMPLETE')
    except Exception as e:
        log('CHAIN_FAILED '+repr(e));raise


if __name__=='__main__':main(sys.argv[1])
