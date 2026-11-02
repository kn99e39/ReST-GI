import json
e=json.load(open('out/eval.json'));R=e['results']
cs=['init','R@500','R@2000','R@6000','R@10000','R@20000','N@500','N@2000','N@6000','N@10000','N@20000','smokeA','C_5902','frozen_P_5902']
print('per-source rest slots 4-7 (abs_coarse/con worst ch); pt0 s4-7')
for c in cs:print(c.ljust(14),' '.join(f"s{s}:{max(R[c]['per_matrix'][f'rest_s{s}']['abs_coarse']):.2f}/{max(R[c]['per_matrix'][f'rest_s{s}']['con_coarse']):.2f}" for s in range(4,8)),' | pt0',' '.join(f"{max(R[c]['per_matrix'][f'pt0_s{s}']['abs_coarse']):.2f}/{max(R[c]['per_matrix'][f'pt0_s{s}']['con_coarse']):.2f}" for s in range(4,8)))
print('flux within 1 cm (rest s0..7); physical:',[round(e['physical_flux'][f'rest_s{s}'][1],3) for s in range(8)])
for c in cs:print(c.ljust(14),[round(R[c]['flux'][f'rest_s{s}'][1],3) for s in range(8)])
print('T change rest s0..7')
for c in cs:print(c.ljust(14),[round(x,2) for x in R[c]['T_change']['rest']])
print('extended coarse mean rest / dyn / passivity')
for c in cs:
    x=R[c]['extended_fields'];print(c.ljust(14),[round(v,3) for v in x['poses']['rest']['metrics']['coarse']['mean']],[(round(d['error'],2),round(d['corr'],2)) for d in x['dynamic']],'pass',all(v['passed'] for v in R[c]['passivity'].values()),round(max(v['max_colsum'] for v in R[c]['passivity'].values()),5))
