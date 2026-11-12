"""WL90 configuration: the frozen WL86-WL89 code imported unchanged (fx89 -> fx88 -> fx87 -> pt86); WL89 workloads (W1 = V3, W2 = X2) reused."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gm1_relation89'))
sys.path.insert(0, HERE)
import fx89  # noqa: E402

fx88 = fx89.fx88
PT, Q = fx89.PT, fx89.Q
FX = PT.FX
P90 = json.load(open(os.path.join(HERE, 'protocol', 'gm1_gap90.json')))
E87 = fx88.fx87.P87['explicit']
P89 = fx89.P89
