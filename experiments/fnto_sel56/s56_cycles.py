"""Same WL55 Cycles renderer/settings, with this batch's immutable fixture provider."""
import sys
import s56_fixtures as SF
import k55_cycles as KC

if __name__=='__main__':
    from pathlib import Path
    out=Path(sys.argv[sys.argv.index('--')+1]);out.mkdir(parents=True,exist_ok=True)
    KC.KF.scene_def=SF.scene
    KC.DEVICE='CUDA'
    for state in ('rest','dent','rigid'):
        seeds=(5611,5612,5613,5614) if state=='rigid' else (5601,5602,5603,5604)
        KC.st_render(out,state,list(SF.CAMERAS),seeds,1024,4)
