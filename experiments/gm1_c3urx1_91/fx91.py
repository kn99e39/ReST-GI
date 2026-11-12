"""WL91 configuration: the frozen WL86-WL90 code imported unchanged (fx90 -> fx89 -> fx88 -> fx87 -> pt86); WL89 workloads, sources and references reused."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gm1_gap90'))
sys.path.insert(0, HERE)
import fx90  # noqa: E402

fx89, fx88 = fx90.fx89, fx90.fx88
PT, Q = fx90.PT, fx90.Q
FX = PT.FX
P91 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_c3urx1_91.json')))
P89, E87 = fx90.P89, fx90.E87
