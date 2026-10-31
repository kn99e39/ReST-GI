"""WL67 blocker evidence that needs no GPU (protocol baseline_sup67_v1, blocker_evidence).

python b67_blockers.py <rna repo> <out json>
(1) RNA native training-camera coverage of the W1 primary face-on queries (geometry of the 8 mm gap aperture, hemisphere cameras);
(2) exact source lines of the RNA native generator settings that the W1 workload would violate.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fnto_sel56'))
import numpy as np  # noqa: E402

import s56_fixtures as SF  # noqa: E402
import k55_common as CM  # noqa: E402


def gap_aperture(sd, lab, face, half=0.05, gap=0.008):
    """Max elevation (deg, from the slab plane) of any straight exterior ray reaching a port centre on a gap face through the side
    aperture: e_max = atan(gap / d_edge), d_edge = distance from the port centre to the nearest slab edge (both slabs share x,y extent).
    The W1 face-on query is at 90 deg; any exterior camera, hemisphere or not, sees these ports only at e <= e_max."""
    lab_off = 0; pcs = []
    for n, k, m, V in sd['comps']:
        P = len(m['port_area'])
        if n == face.split(':')[0]:
            sel = lab[lab_off:lab_off + P] == face
            # x, y of material-frame port centres equal world x, y: WL56 places each rest slab by a z offset only
            pcs = m['port_center'][sel]
        lab_off += P
    d = np.minimum(half - np.abs(pcs[:, 0]), half - np.abs(pcs[:, 1]))
    e = np.degrees(np.arctan2(gap, np.maximum(d, 1e-9)))
    return dict(face=face, n_ports=int(len(pcs)), e_max_deg=dict(min=float(e.min()), median=float(np.median(e)), max=float(e.max())),
                angular_gap_to_face_on_query_deg=dict(min=float(90 - e.max()), median=float(90 - np.median(e)), max=float(90 - e.min())),
                fraction_ports_within_30deg_of_query=float((90 - e <= 30).mean()))


def cite(repo, rel, needles):
    lines = (Path(repo) / rel).read_text().splitlines(); out = []
    for nd in needles:
        hit = [(i + 1, l.strip()) for i, l in enumerate(lines) if nd in l]
        assert hit, (rel, nd)
        out.append(dict(needle=nd, file=rel, lines=hit[:3]))
    return out


def main():
    repo, outp = sys.argv[1], Path(sys.argv[2])
    sd = SF.scene('rest'); lab = CM.labels_of(sd)
    res = dict(protocol='baseline_sup67_v1')
    res['gap_aperture'] = [gap_aperture(sd, lab, f) for f in ('lower:top', 'upper:bot')]
    res['bottom_face'] = ('lower:bot faces -z at z = -14 mm. RNA native hemisphere cameras have z >= look-at z (create_random_camera '
                          "'hemisphere': z in [0, 0.9999) x radius + target); with the look-at at the asset (all shipped configs use [0,0,0] or near it) "
                          'no training view lies below the face plane, so the bot face-on query (-z) is never supervised.')
    res['rna_native_generator'] = cite(repo, 'training_dataset/blender_render_dataset.py', [
        'elif mode == "hemisphere":', 'z = np.random.random() * 0.9999', 'mode="hemisphere", center=[0.5, -0.5, 0.5], radius=camera_radius',
        'blu.add_sun_light(light_energy, light_radius)', 'scene.world.cycles.query_filename = query_file', 'scene.cycles.max_bounces = 32',
        'scene.cycles.device = "CPU"  # GPU not supported', 'scene.cycles.sample_clamp_direct = (', 'self.h5file["position"][i, ...] = pos_out'])
    res['not_executed'] = ('Transparent-BSDF boundary G-buffer (Cycles writes data passes only above pass_alpha_threshold): code-level concern only, '
                           'not executed in WL67 and not used for any verdict.')
    outp.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(outp, 'w'), indent=1)
    print(json.dumps(res['gap_aperture'], indent=1))


if __name__ == '__main__':
    main()
