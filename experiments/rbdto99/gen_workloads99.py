"""Writes protocol/workloads99.json (frozen with protocol rbdto99_v1). Candidate workload definitions for Stage 0 qualification."""
import json, os
import numpy as np
import scene99 as S
HERE = os.path.dirname(os.path.abspath(__file__))


def add(a, b):
    return [float(x + y) for x, y in zip(a, b)]


def body_point(pose, p_body):
    M = S.pose_matrix(pose)
    return [float(v) for v in M[:3, :3] @ np.asarray(p_body, float) + M[:3, 3]]


G90 = {"kind": "groove", "prm": {"L": 0.6, "W": 0.35, "angle": 90}}
c0 = [0.0, 0.6, 0.45]
P0 = {"c": c0, "yaw": 0.0, "pitch": 50.0}
P1 = {"c": add(c0, [0.12, -0.05, 0.0]), "yaw": 25.0, "pitch": 50.0}
CAM = {"origin": add(c0, [-0.5, -2.2, 1.85]), "target": add(c0, [0.0, -0.12, 0.14]), "fov_deg": 18.0}
L_retro = add(c0, [0.5, -2.2, 1.85])
d = np.array([0.5, -2.2, 1.85]); d /= np.linalg.norm(d)
BAR = {"c": add(c0, list(0.62 * d + np.array([0.0, -0.05, 0.12]))), "h": [0.32, 0.012, 0.012], "yaw": 0.0}
ang = np.radians(50 + 38)
L_graze = add(c0, [0.5, -2.9 * np.sin(ang), 2.9 * np.cos(ang)])
dg = np.array([0.5, -2.9 * np.sin(ang), 2.9 * np.cos(ang)]); dg /= np.linalg.norm(dg)
BAR_G = {"c": add(c0, list(0.55 * dg + np.array([0.0, 0.0, 0.05]))), "h": [0.32, 0.012, 0.012], "yaw": 0.0}
L_obl = add(c0, [1.6, -1.2, 1.6])
BAR_O = {"c": add(c0, [0.55, -0.45, 0.62]), "h": [0.012, 0.25, 0.012], "yaw": -20.0}
INTR = {"c": body_point(P1, [0.0, 0.0, 0.10]), "h": [0.04, 0.04, 0.04], "yaw": 25.0}

cc = [0.1, 0.5, 0.05]
PC0 = {"c": cc, "yaw": -135.0, "pitch": 0.0}
PC1 = {"c": add(cc, [0.1, 0.05, 0.0]), "yaw": -115.0, "pitch": 0.0}
CAMC = {"origin": add(cc, [-0.25, -2.3, 1.6]), "target": add(cc, [0.0, 0.0, 0.18]), "fov_deg": 16.0}
LC = add(cc, [0.25, -2.3, 1.6])
dc = np.array([0.25, -2.3, 1.6]); dc /= np.linalg.norm(dc)
BAR_C = {"c": add(cc, list(0.6 * dc + np.array([0.0, 0.0, 0.2]))), "h": [0.3, 0.012, 0.012], "yaw": 0.0}

ch = [0.0, 0.5, 0.05]
PH0 = {"c": ch, "yaw": 0.0, "pitch": 0.0}
PH1 = {"c": add(ch, [0.1, 0.0, 0.0]), "yaw": 20.0, "pitch": 0.0}
CAMH = {"origin": add(ch, [-0.6, -1.8, 1.9]), "target": add(ch, [0.0, 0.0, 0.15]), "fov_deg": 18.0}
LH = add(ch, [0.6, -0.9, 2.4])
BAR_H = {"c": add(ch, [0.3, -0.3, 1.1]), "h": [0.012, 0.3, 0.012], "yaw": 0.0}

W = [
 {"id": "A1_g90s1", "type": "A", "role": "primary sharp (development geometry)", "body": dict(G90, mat="S1"), "T0": P0, "T1": P1,
  "lights": [{"pos": L_retro, "rgb": [30, 30, 30]}], "boxes": [BAR], "camera": CAM, "co_check": True, "intrusion": INTR},
 {"id": "A2_g90s1_graze", "type": "A", "role": "sharp grazing (development geometry)", "body": dict(G90, mat="S1"), "T0": P0, "T1": P1,
  "lights": [{"pos": L_graze, "rgb": [30, 30, 30]}], "boxes": [BAR_G], "camera": CAM},
 {"id": "B1_g90s2", "type": "B", "role": "broader reflection (S2)", "body": dict(G90, mat="S2"), "T0": P0, "T1": P1,
  "lights": [{"pos": L_retro, "rgb": [30, 30, 30]}], "boxes": [BAR], "camera": CAM},
 {"id": "C_box", "type": "C", "role": "null control (convex body)", "body": {"kind": "box", "prm": {"A": 0.35}, "mat": "S1"},
  "T0": {"c": [0.0, 0.5, 0.05], "yaw": 20.0}, "T1": {"c": [0.12, 0.45, 0.05], "yaw": 45.0},
  "lights": [{"pos": L_retro, "rgb": [30, 30, 30]}], "boxes": [BAR], "camera": CAM},
 {"id": "A3_corner", "type": "A", "role": "held-out geometry (trihedral corner)", "body": {"kind": "corner", "prm": {"A": 0.4}, "mat": "S1"},
  "T0": PC0, "T1": PC1, "lights": [{"pos": LC, "rgb": [30, 30, 30]}], "boxes": [BAR_C], "camera": CAMC, "co_check": True},
 {"id": "A4_channel", "type": "A", "role": "held-out geometry (U-channel)", "body": {"kind": "channel", "prm": {"L": 0.6, "G": 0.25, "H": 0.3}, "mat": "S1"},
  "T0": PH0, "T1": PH1, "lights": [{"pos": LH, "rgb": [30, 30, 30]}], "boxes": [BAR_H], "camera": CAMH},
 {"id": "A5_g90s1_obl", "type": "A", "role": "held-out illumination (oblique light, other occluder)", "body": dict(G90, mat="S1"), "T0": P0, "T1": P1,
  "lights": [{"pos": L_obl, "rgb": [30, 30, 30]}], "boxes": [BAR_O], "camera": CAM},
]
json.dump({"protocol": "rbdto99_v1", "candidate_budget": 16, "workloads": W}, open(os.path.join(HERE, 'protocol', 'workloads99.json'), 'w'), indent=1)
print(len(W))
