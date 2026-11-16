"""WL99 fixture: GM1-style room + one rigid non-convex ER body + Lambert external boxes (occluders / intruders) + point lights.

numpy only (no renderer import). Units metres, z up. Room x, y in [-2, 2], z in [0, 3], open towards -y (as scene86).
A body is a union of thin plates (closed cuboids) defined in its body frame; its pose is (c, yaw, pitch) -> to_world.
Body material: ER state S1 (sharp, WL85 test83_hard #97, alpha ~0.05) or S2 (moderate, #41, alpha ~0.12); materials are the frozen
WL86 table (experiments/ecro86/data/mats86.npz): 0 B (floor), 1 S1, 2 S2, 3 Lambert 0.7 (walls / ceiling), 4 Lambert 0.5 (boxes).
"""
import math
import numpy as np

X, Y, Z = np.eye(3)
THICK = 0.01                          # plate thickness (m)
MAT_ID = {'B': 0, 'S1': 1, 'S2': 2, 'LW': 3, 'LB': 4}


def rot(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """R = Rz(yaw) Rx(pitch) Ry(roll) (body -> world)."""
    a, b, c = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Ry = np.array([[math.cos(c), 0, math.sin(c)], [0, 1, 0], [-math.sin(c), 0, math.cos(c)]])
    return Rz @ Rx @ Ry


def pose_matrix(pose):
    M = np.eye(4)
    M[:3, :3] = rot(pose.get('yaw', 0.0), pose.get('pitch', 0.0), pose.get('roll', 0.0))
    M[:3, 3] = pose['c']
    return M


def plate(center, n, u, half_u, half_v):
    """A thin plate: centre, outward normal n of its main face, in-plane axis u; returns a 4x4 unit-cube to_body matrix + metadata."""
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    u = np.asarray(u, float); u -= n * (u @ n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    M = np.eye(4)
    M[:3, 0] = u * half_u; M[:3, 1] = v * half_v; M[:3, 2] = n * (THICK / 2)
    M[:3, 3] = np.asarray(center, float) - n * (THICK / 2)       # main face exactly through `center`
    return {'M': M, 'n': n, 'u': u, 'v': v, 'hu': half_u, 'hv': half_v, 'c': np.asarray(center, float)}


# ---------------------------------------------------------------------------------------------------------------- body family
def body_plates(kind, prm):
    """Body-frame plates. Origin on the body's lowest edge / base centre, z up in the body frame; inner faces face the cavity."""
    P = []
    if kind == 'groove':                     # V-groove: two plates sharing the x-axis edge, dihedral `angle`, opening +z
        L, W, ang = prm['L'], prm['W'], math.radians(prm['angle'])
        for s in (-1, 1):
            t = np.array([0, s * math.sin(ang / 2), math.cos(ang / 2)])          # in-plane direction away from the edge
            n = np.array([0, -s * math.cos(ang / 2), math.sin(ang / 2)])         # inner normal (towards the cavity)
            P.append(plate(t * W / 2, n, X, L / 2, W / 2))
    elif kind == 'channel':                  # U-channel: floor plate + two parallel walls (height H, gap G)
        L, G, H = prm['L'], prm['G'], prm['H']
        P.append(plate([0, 0, 0], Z, X, L / 2, G / 2))
        for s in (-1, 1):
            P.append(plate([0, s * G / 2, H / 2], [0, -s, 0], X, L / 2, H / 2))
    elif kind == 'corner':                   # trihedral corner: three mutually orthogonal square plates of side A, cavity +x +y +z
        A = prm['A']
        P.append(plate([A / 2, A / 2, 0], Z, X, A / 2, A / 2))
        P.append(plate([A / 2, 0, A / 2], Y, X, A / 2, A / 2))
        P.append(plate([0, A / 2, A / 2], X, Y, A / 2, A / 2))
    elif kind == 'box':                      # convex null control: a closed cube (one plate of full thickness)
        A = prm['A']
        M = np.eye(4); M[:3, :3] = np.eye(3) * (A / 2); M[:3, 3] = [0, 0, A / 2]
        P.append({'M': M, 'n': Z, 'u': X, 'v': Y, 'hu': A / 2, 'hv': A / 2, 'c': np.array([0, 0, A])})
    else:
        raise ValueError(kind)
    return P


def body_hull_points(plates):
    """Corner points of all plates (body frame) - the convex hull is the conservative self-path region (certificate)."""
    pts = []
    for p in plates:
        M = p['M']
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(M[:3, :3] @ np.array([sx, sy, sz]) + M[:3, 3])
    return np.array(pts)


def box_matrix(box):
    """Lambert box {c, h, yaw} (world) -> unit-cube to_world."""
    M = np.eye(4)
    R = rot(box.get('yaw', 0.0))
    M[:3, :3] = R * np.asarray(box['h'], float)[None, :]
    M[:3, 3] = box['c']
    return M


def box_corners(box):
    M = box_matrix(box)
    return np.array([M[:3, :3] @ np.array([sx, sy, sz]) + M[:3, 3] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# ---------------------------------------------------------------------------------------------------------------- certificate
def hull_planes(pts):
    """Half-space form of the convex hull (scipy-free: brute force over point triples; small point sets only)."""
    from itertools import combinations
    planes = []
    c0 = pts.mean(0)
    for i, j, k in combinations(range(len(pts)), 3):
        n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn; d = n @ pts[i]
        s = pts @ n - d
        if np.all(s <= 1e-7):
            pass
        elif np.all(s >= -1e-7):
            n, d = -n, -d
        else:
            continue
        if not any(np.allclose(n, q[0], atol=1e-6) and abs(d - q[1]) < 1e-6 for q in planes):
            planes.append((n, d))
    return planes


def convex_overlap(ptsA, ptsB):
    """Separating-axis test between two convex point sets using the hull face normals of both and pairwise edge crosses (approx.:
    face normals + coordinate axes; conservative = may report overlap for separated sets, never the reverse)."""
    axes = [p[0] for p in hull_planes(ptsA)] + [p[0] for p in hull_planes(ptsB)] + list(np.eye(3))
    for a in axes:
        pa, pb = ptsA @ a, ptsB @ a
        if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
            return False
    return True


def certificate(cfg):
    """True if no foreign primitive (occluder / intruder box, other lights excluded) intersects the body's convex hull (world)."""
    plates = body_plates(cfg['body']['kind'], cfg['body']['prm'])
    M = pose_matrix(cfg['body']['pose'])
    hull = body_hull_points(plates) @ M[:3, :3].T + M[:3, 3]
    for b in cfg.get('boxes', []):
        if convex_overlap(hull, box_corners(b)):
            return False
    return True
