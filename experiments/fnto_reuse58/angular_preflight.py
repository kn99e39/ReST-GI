"""Weight-independent finite-interface admission test. No historical code is mutated.

The optimal L2 histogram projection is a LOWER bound on any candidate in that
output basis. Identical projected inputs also give a triangle-inequality bound
on the best possible source-swap error of ANY deterministic response rule.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def square_direction(u, v):
    """Independent two-region concentric map; equal projected-solid-angle measure."""
    a, b = np.broadcast_arrays(2 * np.asarray(u) - 1, 2 * np.asarray(v) - 1)
    use_a = np.abs(a) > np.abs(b)
    radius = np.where(use_a, a, b)
    phi = np.where(use_a, (math.pi / 4) * b / np.where(a == 0, 1, a),
                   math.pi / 2 - (math.pi / 4) * a / np.where(b == 0, 1, b))
    x, y = radius * np.cos(phi), radius * np.sin(phi)
    return np.stack([x, y, np.sqrt(np.maximum(0., 1 - x*x - y*y))], -1)


def direction_bin(w, grid):
    """Inverse of the published concentric map; labels outward/source directions."""
    x, y = np.asarray(w)[..., 0], np.asarray(w)[..., 1]
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    phi = np.where(phi < -math.pi / 4, phi + 2 * math.pi, phi)
    q = math.pi / 4
    a = np.select([phi < q, phi < 3*q, phi < 5*q],
                  [r, -(phi - math.pi/2)*r/q, -r], default=(phi-3*math.pi/2)*r/q)
    b = np.select([phi < q, phi < 3*q, phi < 5*q],
                  [phi*r/q, r, -(phi-math.pi)*r/q], default=-r)
    iu = np.clip(np.floor((a+1)*grid/2).astype(np.int64), 0, grid-1)
    iv = np.clip(np.floor((b+1)*grid/2).astype(np.int64), 0, grid-1)
    return iu + grid*iv


def ggx(wi, wo, alpha):
    """GGX with correlated Smith G and unit Fresnel. Density per projected angle.

    f = D(h) G(wi,wo)/(4 mu_i mu_o). A unit incident flux gives
    outgoing flux f * d(omega_projected), not f*domega or pi*f.
    """
    wi, wo = np.asarray(wi, dtype=np.float64), np.asarray(wo, dtype=np.float64)
    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    a2 = alpha * alpha
    D = a2 / (math.pi * (1 + (a2-1)*h[..., 2]**2)**2)
    mi, mo = wi[..., 2], wo[..., 2]
    li = .5 * (np.sqrt(1 + a2*(1-mi*mi)/(mi*mi))-1)
    lo = .5 * (np.sqrt(1 + a2*(1-mo*mo)/(mo*mo))-1)
    return D / ((1 + li + lo) * 4 * mi * mo)


def field_stats(wi, alpha, grid, output_grid):
    """Streaming midpoint quadrature; memory O(64*grid), not O(grid^2*cases)."""
    assert grid % output_grid == 0
    bins = np.zeros(output_grid**2, dtype=np.float64)
    energy_sum = square_sum = 0.
    cols = np.arange(grid)[None, :]
    for row in range(0, grid, 64):
        rows = np.arange(row, min(row+64, grid))[:, None]
        wo = square_direction((cols+.5)/grid, (rows+.5)/grid)
        f = ggx(wi, wo, alpha)
        if not np.isfinite(f).all() or np.min(f) < 0:
            raise ValueError('Invalid independent GGX reference')
        ix = np.broadcast_to(cols*output_grid//grid, f.shape)
        iy = np.broadcast_to(rows*output_grid//grid, f.shape)
        bins += np.bincount((ix+output_grid*iy).ravel(), weights=f.ravel(), minlength=output_grid**2)
        energy_sum += float(f.sum())
        square_sum += float(np.square(f).sum())
    count = (grid//output_grid)**2
    # Projection onto piecewise-constant radiance in equal projected-area bins.
    explained = float(np.square(bins).sum()/count)
    residual = max(0., square_sum-explained)
    area = math.pi/(grid*grid)
    return dict(grid=grid, energy=energy_sum*area, l2_norm=math.sqrt(square_sum*area),
                output_flux=(bins*area).tolist(),
                best_output_relative_l2=math.sqrt(residual/square_sum))


def independent_polar_moments(wi, alpha):
    """Gauss-Legendre(mu) x azimuth midpoint, independent of square-map bins."""
    nodes, weights = np.polynomial.legendre.leggauss(512)
    mu, weights = (nodes+1)/2, weights/2
    phi = (np.arange(2048)+.5)*(2*math.pi/2048)
    e = s = 0.
    for j in range(0, len(mu), 64):
        m = mu[j:j+64, None]
        r = np.sqrt(1-m*m)
        wo = np.stack(np.broadcast_arrays(r*np.cos(phi), r*np.sin(phi), m), -1)
        f = ggx(wi, wo, alpha)
        measure = weights[j:j+64, None]*m*(2*math.pi/2048)
        e += float((f*measure).sum())
        s += float((f*f*measure).sum())
    return dict(energy=e, l2_norm=math.sqrt(s))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def pair_bound(a, b):
    """If a common prediction met error e for BOTH targets, e >= this bound."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b)))


def analytic_checks():
    rng = np.random.default_rng(580101)
    w = square_direction(rng.uniform(.01, .99, 128), rng.uniform(.01, .99, 128))
    v = square_direction(rng.uniform(.01, .99, 128), rng.uniform(.01, .99, 128))
    errors = [rel(ggx(w, v, a), ggx(v, w, a)) for a in (.04, .25)]
    # Normal-incidence closed-form BRDF peak, independent of quadrature.
    n = np.array([0., 0., 1.])
    peak_errors = [abs(float(ggx(n, n, a))*4*math.pi*a*a-1) for a in (.04, .25)]
    return dict(reciprocity_relative_max=max(errors), normal_peak_relative_max=max(peak_errors))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def run(out, protocol_path):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'angular.json').exists():
        raise FileExistsError('Preserve completed gate; do not overwrite evidence')
    t0 = time.perf_counter()
    protocol = json.loads(protocol_path.read_text())
    p = protocol['preflight']
    ng = protocol['candidate']['output_grid']
    ni = protocol['candidate']['input_grid']
    cases, lookup = [], {}

    def evaluate(label, wi, alpha):
        row = dict(label=label, alpha=alpha, incident_direction=np.asarray(wi).tolist())
        row['quadratures'] = [field_stats(wi, alpha, k, ng) for k in p['reference_grids']]
        a, b = row['quadratures'][-2:]
        polar = independent_polar_moments(wi, alpha)
        row['independent_polar'] = polar
        row['reference_discrepancies'] = dict(
            energy=rel(a['energy'], b['energy']), l2_norm=rel(a['l2_norm'], b['l2_norm']),
            binned_output=rel(a['output_flux'], b['output_flux']),
            projection_bound_absolute=abs(a['best_output_relative_l2']-b['best_output_relative_l2']),
            polar_energy=rel(polar['energy'], b['energy']), polar_l2=rel(polar['l2_norm'], b['l2_norm']))
        row['reference_pass'] = bool(max(row['reference_discrepancies'].values()) <= p['reference_relative_convergence_max']
                                     and max(x['energy'] for x in row['quadratures']) <= p['reference_energy_max'])
        row['projection_pass'] = bool(b['best_output_relative_l2'] <= p['output_projection_rel_l2_max'])
        cases.append(row)
        lookup[label] = b['output_flux']
        atomic_json(out/'progress.json', dict(completed=len(cases), latest=label, seconds=time.perf_counter()-t0))
        print(json.dumps(dict(case=label, reference_pass=row['reference_pass'], projection_floor=b['best_output_relative_l2'])), flush=True)
        return label

    for alpha in p['alpha']:
        for theta in p['theta_degrees']:
            for phi in p['phi_degrees']:
                t, f = np.deg2rad([theta, phi])
                wi = np.array([math.sin(t)*math.cos(f), math.sin(t)*math.sin(f), math.cos(t)])
                evaluate(f'a{alpha}_theta{theta}_phi{phi}', wi, alpha)

    pairs = []
    for alpha in p['alpha']:
        for cell in p['pair_input_bins']:
            labels, inputs = [], []
            for key, (u, v) in zip(('X', 'Y'), p['pair_offsets']):
                wi = square_direction((cell%ni+u)/ni, (cell//ni+v)/ni)
                index = int(direction_bin(wi, ni))
                x = np.zeros(ni*ni)
                x[index] = 1.
                inputs.append(x)
                labels.append(evaluate(f'a{alpha}_cell{cell}_{key}', wi, alpha))
            bound = pair_bound(lookup[labels[0]], lookup[labels[1]])
            pairs.append(dict(alpha=alpha, declared_input_bin=cell, labels=labels,
                              input_difference=float(np.max(np.abs(inputs[0]-inputs[1]))),
                              best_common_prediction_error_lower_bound=bound,
                              passed=bool(bound <= p['source_alias_pair_lower_bound_max'])))
    checks = analytic_checks()
    reference_go = all(x['reference_pass'] for x in cases) and checks['reciprocity_relative_max'] <= p['reciprocity_relative_max'] and checks['normal_peak_relative_max'] <= 1e-10
    angular_go = all(x['projection_pass'] for x in cases) and all(x['passed'] for x in pairs)
    status = ('GO_ANGULAR_ONLY' if angular_go else 'STOPPED_ANGULAR_REPRESENTATION') if reference_go else 'STOPPED_REFERENCE'
    source = os.environ.get('REUSE_SOURCE_COMMIT')
    if source is None:
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    result = dict(batch_id=protocol['batch_id'], status=status, source_commit=source,
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_pass=bool(reference_go), angular_pass=bool(angular_go),
                  analytic_checks=checks, cases=cases, source_swap_pairs=pairs,
                  training_executed=False, candidate_F_S_executed=False,
                  seconds=time.perf_counter()-t0, peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                                   hostname=platform.node(), method='deterministic float64 CPU quadrature, no Monte Carlo or GPU model'))
    atomic_json(out/'angular.json', result)
    (out/'STATUS').write_text(status+'\n')
    verdicts = dict(source_independent_response_correctness='INCONCLUSIVE', directional_representation_adequacy='SUPPORTED' if angular_go and reference_go else ('NOT SUPPORTED' if reference_go else 'INCONCLUSIVE'),
                    full_state_learned_fidelity='INCONCLUSIVE', selective_state_fidelity='INCONCLUSIVE', state_invalidation_correctness='INCONCLUSIVE',
                    actual_update_query_advantage='INCONCLUSIVE', K_C_composition_eligibility='NOT SUPPORTED' if not angular_go and reference_go else 'INCONCLUSIVE',
                    rigid_local_deformation='INCONCLUSIVE', global_bend_self_contact='INCONCLUSIVE', abstract_rule_state_hypothesis='INCONCLUSIVE')
    atomic_json(out/'verdict.json', dict(status=status, scopes=verdicts, reason='Preflight only. No learned weights or selective-state costs were measured.'))
    print(json.dumps(dict(status=status, seconds=result['seconds'])), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=HERE/'protocol/reuse_v1.json')
    args = ap.parse_args()
    run(args.out, args.protocol)
