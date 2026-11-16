# Track J4: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [Show phase-1 E4 findings] python experiments/audit97/_tools/show.py E4-F1 E4-F2 E4-F3 E4-F4 E4-F5 E4-F6 E4-F7 E4-F8 E4-F9 2>&1 | head -c 30000
RUN [Show remaining findings and list audit directories] python experiments/audit97/_tools/show.py C2-09 B-08 MAP-11 F0-05 A-08 2>&1 | head -c 20000; ls docs/audits/wl97_evidence_reliability/ experiments/audit97/ results/audit97/ results/audit97/E4 experiments/audit97/E4 2>&1 | head -100
  -> ### [E4-F1] (E4) Event accounting of CKF_none is as claimed (e2 = 2 events, e3 = 3..5, _p = adjacent vertex)
category=confirmation material=False verdict=VERIFIED confidence=high p0=['P0-4'] worklogs=['WL96A', 'WL94']
- description: Production field tracer with kmax 2 / 3 / 5 at the stored seed an ...[cut]... ngs 0.00054 / 0.00066 vs same-source 0.00065 / 0.00058. Shares in ROI_T: h 0.32, tail as read 0.178, e3_o 0.0623, so M_img = 0.5 % of ROI radiance = 1.6 % of h = 8.0 % of e3_o. Measured explicit error is 1.0 % of e3_o.
- independent_check: Own statistics, re-rendered component tables (Pc, Fn, Pn, XODn, dOab, dOn).
- numerical_disagreement: None.
- decision_impact: Outcome unchanged for any threshold down to about 0.001 of the ROI total (the detectable effect size; paired camera noise 2e-5, reference tail noise 0.0005-0.0007). No bias in favour of explicit from shared prefix/reference noise.
  -> ### [F0-05] (F0) WL96A answers a narrower question than the smoke WL96 specified to settle its dissent
category=invalid_inference material=True verdict=CONDITIONALLY_VALID confidence=high p0=['P0-4'] worklogs=['WL96', 'WL96A']
- description: WL96 section F and both critics specified: CK2 with the  ...[cut]... 1_learned_sender_error_and_passivity.md
E2_relation_sampling_reproduction.md
E3_deterministic_delivery_oracle.md
E4_resolvent_accuracy_headroom.md
E5_lifecycle_cost_reproduction.md
F0_inference_chain.md
MAP_evidence_dependency_and_provenance.md

experiments/audit97/:
A
B
C1
C2
D
E1
E2
E3
E4
E5
F0
MAP
_tools

experiments/audit97/E4:
e4_extra.py
e4_field.py
e4_fieldan.py
e4_img.py
e4_kmax.py
e4_ref.py
e4_render.py
e4_tail.py

results/audit97/:
A
B
C1
C2
D
E1
E2
E3
E4
E5
F0
MAP
phase1_results.json

results/audit97/E4:
_interrupted_run_log.md
extra.json
fieldan.json
img.json
kmax.json
tails97.json
RUN [Show phase-1 E4 findings] cd /c/Projects/ReST-GI && python experiments/audit97/_tools/show.py E4-F1 E4-F2 E4-F3 E4-F4 E4-F5 E4-F6 E4-F7 E4-F8 E4-F9 2>&1 | head -c 30000
RUN [Show remaining findings and list audit directories] cd /c/Projects/ReST-GI && python experiments/audit97/_tools/show.py C2-09 B-08 MAP-11 F0-05 A-08 2>&1 | head -c 20000; ls docs/audits/wl97_evidence_reliability/ experiments/audit97/ results/audit97/ results/audit97/E4 experiments/audit97/E4 2>&1 | head -100
RUN [Size up WL96A worklog, E4 report and code] wc -l docs/worklogs/96A_*.md docs/audits/wl97_evidence_reliability/E4_*.md experiments/ecro_util96a/*.py experiments/ecro86/pt86.py && ls experiments/ecro_util96a experiments/ecro_util96a/protocol results/ecro_util96a | head -80 && cat results/audit97/E4/_interrupted_run_log.md | head -60
READ c:\Projects\ReST-GI\docs\worklogs\96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md 
  -> 263 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
   374 docs/audits/wl97_evidence_reliability/E4_resolvent_accuracy_headroom.md
    29 experiments/ecro_util96a/figw3_posthoc.py
   819 experiments/ecro_util96a/w96a.py
   498 experiments/ecro86/pt86.py
  1983 total
experiments/ecr ...[cut]... posthoc.py
protocol
run96a.sh
w96a.py

experiments/ecro_util96a/protocol:
ecro_util96a.json

results/ecro_util96a:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz
  263 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
   29 experiments/ecro_util96a/figw3_posthoc.py
  819 experiments/ecro_util96a/w96a.py
 1111 total
  -> Exit code 2
/home/nd/rest_x96a:
chain.out
chain.status
code
data
env.sh
figw3_posthoc.py
results

/home/nd/rest_x96a/results:
cost_s1.0.json
eval96a
fields
final.json
fit
lists
logs
pre
renders
s0.json
s1.json
tables
RUN [Size up WL96A worklog, E4 report and code] cd /c/Projects/ReST-GI && wc -l docs/worklogs/96A_*.md docs/audits/wl97_evidence_reliability/E4_*.md experiments/ecro_util96a/*.py experiments/ecro86/pt86.py && ls experiments/ecro_util96a experiments/ecro_util96a/protocol results/ecro_util96a | head -80 && cat results/audit97/E4/_interrupted_run_log.md | head -60
READ c:\Projects\ReST-GI\docs\worklogs\96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md 
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Dump E4 extra.json structure] python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
def show(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,)):
                print(' '*depth+k+':'); show(v,p,depth+1)
            else:
                s=json.dumps(v)
                print(' '*depth+k+': '+s[:400])
    else: print(o)
show(d)
EOF
  -> <persisted-output>
Output too large (49.7KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bpkum11le.txt

Preview (first 2KB):
per_half_flux:
 all:
  e3_p:
   ckf_halves: [76.24707489522191, 76.35688344323322]
   audi ...[cut]... .9155515560369067
  e3_o:
   ckf_halves: [3229.5673710332335, 3230.1298613739577]
   audit_halves: [3231.703660086945, 3231.268114690288, 3231.789074519265, 3228.252003441894]
   pooled_cv_per_half: 0.0004321978093603942
   ckf_mean_minus_audit_mean_in_pooled_sd_units: -0.7481327927615573
  e2_p:
   ckf_halves: [35.387800298690046, 34.70169100455348]
   audit_halves: [38.85836141016515, 36.70851637495, 36.20676205703577, 35.680873018463444]
   pooled_cv_per_half: 0.03992982885163022
   ckf_mean_minus_audit_mean_in_pooled_sd_units: -1.4507103571708735
  e2_o:
...
</persisted-output>
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Dump E4 extra.json structure] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
def show(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,)):
                print(' '*depth+k+':'); show(v,p,depth+1)
            else:
                s=json.dumps(v)
                print(' '*depth+k+': '+s[:400])
    else: print(o)
show(d)
EOF
  -> <persisted-output>
Output too large (49.7KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bpkum11le.txt

Preview (first 2KB):
per_half_flux:
 all:
  e3_p:
   ckf_halves: [76.24707489522191, 76.35688344323322]
   audi ...[cut]... .9155515560369067
  e3_o:
   ckf_halves: [3229.5673710332335, 3230.1298613739577]
   audit_halves: [3231.703660086945, 3231.268114690288, 3231.789074519265, 3228.252003441894]
   pooled_cv_per_half: 0.0004321978093603942
   ckf_mean_minus_audit_mean_in_pooled_sd_units: -0.7481327927615573
  e2_p:
   ckf_halves: [35.387800298690046, 34.70169100455348]
   audit_halves: [38.85836141016515, 36.70851637495, 36.20676205703577, 35.680873018463444]
   pooled_cv_per_half: 0.03992982885163022
   ckf_mean_minus_audit_mean_in_pooled_sd_units: -1.4507103571708735
  e2_o:
...
</persisted-output>
RUN [Locate table-readout sections in the E4 report] grep -n -i "readout\|table space\|proxy\|ROI_T\b.*mask\|block-16\|block16" docs/audits/wl97_evidence_reliability/E4_resolvent_accuracy_headroom.md | head -60
RUN [Show the table_readout_proxy block and the script that made it] python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
print(list(d.keys()))
t=d['table_readout_proxy']
def show(o,depth=0):
    for k,v in o.items():
        if isinstance(v,dict):
            print(' '*depth+k+':'); show(v,depth+1)
        else:
            print(' '*depth+k+': '+json.dumps(v)[:300])
show(t)
EOF
cat experiments/audit97/E4/e4_extra.py | head -150
  -> 18:noise. Against a new event-split path-traced camera reference, the explicit tail readout is as accurate as the "ceiling" table itself
21:1. the "ceiling" is a 1270 x 64 patch x bin TABLE; it is not the path-traced image. The tail table readout itself deviates from the
154:- Proper estimates: debi ...[cut]...  1270 x 64 table OUTPUT can do. The tail's table-readout error against
223:  (c) `T_READOUT` in WL96A (1.7e-7) tests render reproducibility, not proxy validity; no test of proxy validity was part of the batch.
224:- Evidence: `img.json` (A_F_vs_REF_h), `extra.json` (table_readout_proxy); `pt86.py:228-233` (table lookup at camera vertex 2).
242:  or geometry was not tested. This limits the claim to the tested readout, as its wording ("on the tested CK2 workload") allows.
323:- Cause of the table-readout deviation (E4-F6) and of the sphere-receiver flux deficit (E4-F7): measured, not attributed.
  -> <persisted-output>
Output too large (50.6KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\ble93cgs3.txt

Preview (first 2KB):
['per_half_flux', 'seed_sanity', 'refa_streams_identical (e1 + e2 of k3 vs k6) max abs', 'tabl ...[cut]... 64: 0.004857016326538523
  L1_block64_noise: 0.00021925230530004682
 F_vs_h6new (3..6 events):
  L1: 0.013964337990608077
  L1_null_expectation: 0.00912998038685107
  L1_noise_realization: 0.010247021887019071
  signed: 0.002664388462851576
  y_share_of_tot: 0.32064034823728405
  x_over_y: 1.0083095857321107
  rel_l2_raw_vs_tot: 0.019430548019345923
  rel_l2_noise_vs_tot: 0.01675115136693536
  rel_l2_debiased_vs_tot: 0.009846071511731338
  rel_l2_debiased_vs_y: 0.032101588850798446
  L1_block8: 0.008308009060660893
  L1_block8_noise: 0.0016277410469441747
...
</persisted-output>
RUN [Locate table-readout sections in the E4 report] cd /c/Projects/ReST-GI && grep -n -i "readout\|table space\|proxy\|ROI_T\b.*mask\|block-16\|block16" docs/audits/wl97_evidence_reliability/E4_resolvent_accuracy_headroom.md | head -60
RUN [Show the table_readout_proxy block and the script that made it] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
print(list(d.keys()))
t=d['table_readout_proxy']
def show(o,depth=0):
    for k,v in o.items():
        if isinstance(v,dict):
            print(' '*depth+k+':'); show(v,depth+1)
        else:
            print(' '*depth+k+': '+json.dumps(v)[:300])
show(t)
EOF
cat experiments/audit97/E4/e4_extra.py | head -150
READ c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\E4_resolvent_accuracy_headroom.md 
RUN [Compact view of table_readout_proxy for all masks] python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
t=d['table_readout_proxy']
print(list(t.keys()))
for m,v in t.items():
    if not isinstance(v,dict): print(m, v); continue
    print('==',m)
    for c,w in v.items():
        if isinstance(w,dict):
            keys=['L1','L1_null_expectation','signed','x_over_y','y_share_of_tot','rel_l2_debiased_vs_tot','L1_block16','L1_block16_noise','L1_block64','L1_block64_noise']
            print('  ',c,' | '.join(f"{k.replace(
  -> ['ROI_T', 'DIFF', 'WHOLE', 'PANEL', 'PANEL_NOT_ROI', 'BOXA_PIX', 'FLOOR_PIX', 'WALL_PIX']
== ROI_T
   hstored_vs_h6new (reference consistency) L1=0.01335 | null_expectation=0.01235 | signed=0.00006 | x_over_y=1.00019 | y_share_of_tot=0.32064 | debiased_vs_tot=-0.00273 | block16=0.00093 | block16_n ...[cut]... ll_expectation=0.00310 | signed=-0.00092 | x_over_y=0.99367 | y_share_of_tot=0.14572 | debiased_vs_tot=0.00098 | block16=0.00111 | block16_noise=0.00053 | block64=0.00098 | block64_noise=0.00015
   tail explicit: (XOD - Z) vs (hstored - h3new) L1=0.00863 | null_expectation=0.00744 | signed=-0.00069 | x_over_y=0.99524 | y_share_of_tot=0.14549 | debiased_vs_tot=0.00117 | block16=0.00137 | block16_noise=0.00100 | block64=0.00079 | block64_noise=0.00029
   signed_se F_vs_hstored 5.6305347751401084e-05
   signed_se tail paired 3.097944358684881e-05
   signed_se Z_vs_h3new 0.00010535158601464251
READ c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\E4_resolvent_accuracy_headroom.md 
RUN [Compact view of table_readout_proxy for all masks] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/extra.json'))
t=d['table_readout_proxy']
print(list(t.keys()))
for m,v in t.items():
    if not isinstance(v,dict): print(m, v); continue
    print('==',m)
    for c,w in v.items():
        if isinstance(w,dict):
            keys=['L1','L1_null_expectation','signed','x_over_y','y_share_of_tot','rel_l2_debiased_vs_tot','L1_block16','L1_block16_noise','L1_block64','L1_block64_noise']
            print('  ',
  -> ['ROI_T', 'DIFF', 'WHOLE', 'PANEL', 'PANEL_NOT_ROI', 'BOXA_PIX', 'FLOOR_PIX', 'WALL_PIX']
== ROI_T
   hstored_vs_h6new (reference consistency) L1=0.01335 | null_expectation=0.01235 | signed=0.00006 | x_over_y=1.00019 | y_share_of_tot=0.32064 | debiased_vs_tot=-0.00273 | block16=0.00093 | block16_n ...[cut]... ll_expectation=0.00310 | signed=-0.00092 | x_over_y=0.99367 | y_share_of_tot=0.14572 | debiased_vs_tot=0.00098 | block16=0.00111 | block16_noise=0.00053 | block64=0.00098 | block64_noise=0.00015
   tail explicit: (XOD - Z) vs (hstored - h3new) L1=0.00863 | null_expectation=0.00744 | signed=-0.00069 | x_over_y=0.99524 | y_share_of_tot=0.14549 | debiased_vs_tot=0.00117 | block16=0.00137 | block16_noise=0.00100 | block64=0.00079 | block64_noise=0.00029
   signed_se F_vs_hstored 5.6305347751401084e-05
   signed_se tail paired 3.097944358684881e-05
   signed_se Z_vs_h3new 0.00010535158601464251
RUN [Read E4 reference script and locate patch helpers] cat experiments/audit97/E4/e4_ref.py && sed -n 1,60p experiments/audit97/E4/e4_render.py && grep -n "def patch_bin\|def patch_point\|def shape_id\|def gen_rays\|^D = \|^NB\|^K = \|NZ\|NPHI\|^KAPPA" experiments/ecro86/pt86.py experiments/ecro86/*.py | head -40
  -> """WL97 audit, track E4: independent camera path-traced references split by event count (plain pt86 'cam' mode, new seeds).
For each seed two renders with IDENTICAL random streams: kmax = 3 (h = NEE at camera vertex 3 only = exactly 3 events) and kmax = 6
(h = 3..6 events).  h6 - h3 is therefore exa ...[cut]... 86.py:187:        phi = (g(19) + a) * (TWO_PI / S.SPH_NPHI) - math.pi
experiments/ecro86/pt86.py:196:        z = (F(bn // S.NPHI) + a) * (1.0 / S.NZ)
experiments/ecro86/pt86.py:197:        phi = (F(bn % S.NPHI) + b) * (TWO_PI / S.NPHI) - math.pi
experiments/ecro86/scene86.py:5:Surfaces are split into N patches (constant layout for every geometry); directions into NB = NZ x NPHI equal-solid-angle bins of the
experiments/ecro86/scene86.py:15:NZ, NPHI = REP['bins_z'], REP['bins_phi']
experiments/ecro86/scene86.py:16:NB = NZ * NPHI
experiments/ecro86/scene86.py:17:KAPPA = REP['state_rays_per_bin']
RUN [Read E4 reference script and locate patch helpers] cd /c/Projects/ReST-GI && cat experiments/audit97/E4/e4_ref.py && sed -n 1,60p experiments/audit97/E4/e4_render.py && grep -n "def patch_bin\|def patch_point\|def shape_id\|def gen_rays\|^D = \|^NB\|^K = \|NZ\|NPHI\|^KAPPA" experiments/ecro86/pt86.py experiments/ecro86/*.py | head -40
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Inspect server audit directories and stored renders] ssh -o ConnectTimeout=20 LabServer63 'ls ~/rest_x97/ ~/rest_x97/E4 | head -60; ls ~/rest_x97/E4/* -d | head; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; ls ~/rest_x96a/results/renders | head -30; ls ~/rest_x94/results 2>/dev/null | head -30' 2>&1 | head -150
  -> /home/nd/rest_x97/:
A
B
C1
C2
D
E1
E2
E3
E4
E5
MAP

/home/nd/rest_x97/E4:
Rk
logs
results
scripts
/home/nd/rest_x97/E4/Rk
/home/nd/rest_x97/E4/logs
/home/nd/rest_x97/E4/results
/home/nd/rest_x97/E4/scripts
1 MiB, 0 %
CK_none_F_X_CK2_T1_r0.json
CK_none_F_X_CK2_T1_r0.npz
CK_none_F_X_CK2_T1_r1.json
CK_ ...[cut]... X_CK2_T1_r1.npz
CK_none_Fa_X_CK2_T1_r2.json
CK_none_Fa_X_CK2_T1_r2.npz
CK_none_Fa_X_CK2_T1_r3.json
CK_none_Fa_X_CK2_T1_r3.npz
CK_none_Fb_X_CK2_T1_r0.json
CK_none_Fb_X_CK2_T1_r0.npz
CK_none_Fb_X_CK2_T1_r1.json
CK_none_Fb_X_CK2_T1_r1.npz
CK_none_Fb_X_CK2_T1_r2.json
CK_none_Fb_X_CK2_T1_r2.npz
CK_none_Fb_X_CK2_T1_r3.json
CK_none_Fb_X_CK2_T1_r3.npz
CK_none_G8601_X_CK2_T1_r0.json
CK_none_G8601_X_CK2_T1_r0.npz
CK_none_G8601_X_CK2_T1_r1.json
CK_none_G8601_X_CK2_T1_r1.npz
CK_none_G8601_X_CK2_T1_r2.json
CK_none_G8601_X_CK2_T1_r2.npz
an
eval94
fields
lists
logs
models
pop
pre
renders
select94.json
tables
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Inspect server audit directories and stored renders] ssh -o ConnectTimeout=20 LabServer63 'ls ~/rest_x97/ ~/rest_x97/E4 | head -60; ls ~/rest_x97/E4/* -d | head; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; ls ~/rest_x96a/results/renders | head -30; ls ~/rest_x94/results 2>/dev/null | head -30' 2>&1 | head -150
  -> /home/nd/rest_x97/:
A
B
C1
C2
D
E1
E2
E3
E4
E5
MAP

/home/nd/rest_x97/E4:
Rk
logs
results
scripts
/home/nd/rest_x97/E4/Rk
/home/nd/rest_x97/E4/logs
/home/nd/rest_x97/E4/results
/home/nd/rest_x97/E4/scripts
1 MiB, 0 %
CK_none_F_X_CK2_T1_r0.json
CK_none_F_X_CK2_T1_r0.npz
CK_none_F_X_CK2_T1_r1.json
CK_ ...[cut]... X_CK2_T1_r1.npz
CK_none_Fa_X_CK2_T1_r2.json
CK_none_Fa_X_CK2_T1_r2.npz
CK_none_Fa_X_CK2_T1_r3.json
CK_none_Fa_X_CK2_T1_r3.npz
CK_none_Fb_X_CK2_T1_r0.json
CK_none_Fb_X_CK2_T1_r0.npz
CK_none_Fb_X_CK2_T1_r1.json
CK_none_Fb_X_CK2_T1_r1.npz
CK_none_Fb_X_CK2_T1_r2.json
CK_none_Fb_X_CK2_T1_r2.npz
CK_none_Fb_X_CK2_T1_r3.json
CK_none_Fb_X_CK2_T1_r3.npz
CK_none_G8601_X_CK2_T1_r0.json
CK_none_G8601_X_CK2_T1_r0.npz
CK_none_G8601_X_CK2_T1_r1.json
CK_none_G8601_X_CK2_T1_r1.npz
CK_none_G8601_X_CK2_T1_r2.json
CK_none_G8601_X_CK2_T1_r2.npz
an
eval94
fields
lists
logs
models
pop
pre
renders
select94.json
tables
RUN [Read E4 extra-analysis script header and list server data] sed -n 1,95p experiments/audit97/E4/e4_extra.py; ssh LabServer63 'ls ~/rest_x97/E4/results ~/rest_x97/E4/Rk | head -60; ls ~/rest_x97/E4/logs | head -30; grep -h "REFA" ~/rest_x97/E4/logs/* 2>/dev/null | head -10; ls ~/rest_x96a/code/experiments | head -50; ls ~/rest_x96a/results/tables ~/rest_x96a/results/fields ~/rest_x96a/data 2>/dev/null | head -30'
  -> """WL97 audit, track E4: extra checks (numpy only).
  (a) per-half fluxes of the reference families (is the e3_p flux z-score of fieldan.json real or an SE artefact of 2 halves?)
  (b) new-seed renders really differ from the stored ones (camera noise sanity)
  (c) validity of the table readout as an ...[cut]... stage93c
ecro_util96a
fnto_arch77a
fnto_ato74
fnto_attr60
fnto_attr62
fnto_attrib71b
fnto_c54
fnto_clean65
fnto_closure80d
fnto_cmp72
fnto_comp50
fnto_compiled80c
fnto_compiled81
fnto_compiled82
fnto_compiled83
fnto_compiled84
fnto_components63
fnto_cover71
fnto_delta61
fnto_diag45
fnto_diag48
fnto_dir46
fnto_direction64
fnto_drift71a
fnto_fia76
fnto_fia77
fnto_k55
fnto_lc52
fnto_lego
fnto_lego_k44
/home/nd/rest_x96a/data:
train_A
train_B
val_ck
val_orig

/home/nd/rest_x96a/results/fields:
CKF_none.json
CKF_none.npz

/home/nd/rest_x96a/results/tables:
FZ_own_s8601.npz
FZ_own_s8602.npz
T96A.npz
RUN [Read E4 extra-analysis script header and list server data] cd /c/Projects/ReST-GI && sed -n 1,95p experiments/audit97/E4/e4_extra.py; ssh LabServer63 'ls ~/rest_x97/E4/results ~/rest_x97/E4/Rk | head -60; ls ~/rest_x97/E4/logs | head -30; grep -h "REFA" ~/rest_x97/E4/logs/* 2>/dev/null | head -10; ls ~/rest_x96a/code/experiments | head -50; ls ~/rest_x96a/results/tables ~/rest_x96a/results/fields ~/rest_x96a/data 2>/dev/null | head -30'
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Check pack helper, closure class, env, masks and E4 outputs] grep -n "^def pack" -A 12 experiments/ecro86/pt86.py | head -30; grep -n "def sample\|def eval" -A 6 experiments/fnto_compiled84/*.py 2>/dev/null | grep -n "DrClosure83" | head; grep -rn "class DrClosure83" experiments/ | head -3; ssh LabServer63 'cat ~/rest_x95a/env.sh; python3 - <<EOF
import numpy as np
m=np.load("/home/nd/rest_x94/results/pre/masks94.npz") if __import__("os").path.exists("/home/nd/rest_x94/results/pre/masks94.npz") else None
print(m.files if m else "nomask")
if m:
    for k i
  -> 441:def pack(acc, res, spp):
442-    g = lambda k: (acc[k] / spp).reshape(res, res).astype(np.float32)
443-    o = {}
444-    for name in ('tot', 'e1', 'e2', 'h', 'hl'):
445-        if f'{name}0' in acc:
446-            o[name] = np.stack([g(f'{name}{c}') for c in range(3)], -1)
447-    for name in  ...[cut]... 6_s97701.npz
REFA_k6_s97702.npz
REFA_k6_s97703.npz

/home/nd/rest_x97/E4/results/renders:
A_D_S1_s97600.npz
A_D_S1_s97601.npz
A_D_S1_s97602.npz
A_D_S1_s97603.npz
A_D_floor_s97600.npz
A_D_floor_s97601.npz
A_D_floor_s97602.npz
A_D_floor_s97603.npz
A_D_sphere_s97600.npz
A_D_sphere_s97601.npz
A_D_sphere_s97602.npz
A_D_sphere_s97603.npz
A_F_s86600.npz
A_F_s97600.npz
A_F_s97601.npz
A_F_s97602.npz
A_F_s97603.npz
A_Fa_s97600.npz
A_Fa_s97601.npz
A_Fa_s97602.npz
A_Fa_s97603.npz
A_Fb_s97600.npz
A_Fb_s97601.npz
A_Fb_s97602.npz
A_Fb_s97603.npz
A_Fn_s97600.npz
A_Fn_s97601.npz
A_Fn_s97602.npz
A_Fn_s97603.npz
READ c:\Projects\ReST-GI\experiments\ecro86\pt86.py 
RUN [Check pack helper, closure class, env, masks and E4 outputs] cd /c/Projects/ReST-GI && grep -n "^def pack" -A 12 experiments/ecro86/pt86.py | head -30; grep -n "def sample\|def eval" -A 6 experiments/fnto_compiled84/*.py 2>/dev/null | grep -n "DrClosure83" | head; grep -rn "class DrClosure83" experiments/ | head -3; ssh LabServer63 'cat ~/rest_x95a/env.sh; python3 - <<EOF
import numpy as np
m=np.load("/home/nd/rest_x94/results/pre/masks94.npz") if __import__("os").path.exists("/home/nd/rest_x94/results/pre/masks94.npz") else None
print(m.files if m else "
  -> 441:def pack(acc, res, spp):
442-    g = lambda k: (acc[k] / spp).reshape(res, res).astype(np.float32)
443-    o = {}
444-    for name in ('tot', 'e1', 'e2', 'h', 'hl'):
445-        if f'{name}0' in acc:
446-            o[name] = np.stack([g(f'{name}{c}') for c in range(3)], -1)
447-    for name in  ...[cut]... 6_s97701.npz
REFA_k6_s97702.npz
REFA_k6_s97703.npz

/home/nd/rest_x97/E4/results/renders:
A_D_S1_s97600.npz
A_D_S1_s97601.npz
A_D_S1_s97602.npz
A_D_S1_s97603.npz
A_D_floor_s97600.npz
A_D_floor_s97601.npz
A_D_floor_s97602.npz
A_D_floor_s97603.npz
A_D_sphere_s97600.npz
A_D_sphere_s97601.npz
A_D_sphere_s97602.npz
A_D_sphere_s97603.npz
A_F_s86600.npz
A_F_s97600.npz
A_F_s97601.npz
A_F_s97602.npz
A_F_s97603.npz
A_Fa_s97600.npz
A_Fa_s97601.npz
A_Fa_s97602.npz
A_Fa_s97603.npz
A_Fb_s97600.npz
A_Fb_s97601.npz
A_Fb_s97602.npz
A_Fb_s97603.npz
A_Fn_s97600.npz
A_Fn_s97601.npz
A_Fn_s97602.npz
A_Fn_s97603.npz
RUN [Get E4 chain paths, reference field keys and WL96A table construction] ssh LabServer63 'cat ~/rest_x97/E4/scripts/chain3.sh | head -40; . ~/rest_x95a/env.sh; $PY - <<EOF
import numpy as np
z=np.load("/home/nd/rest_x96a/results/fields/CKF_none.npz"); print({k:z[k].shape for k in z.files})
t=np.load("/home/nd/rest_x96a/results/tables/T96A.npz"); print({k:t[k].shape for k in t.files})
EOF
sed -n 95,125p ~/rest_x96a/code/experiments/ecro_util96a/w96a.py; grep -n "T_READOUT\|def render\|mat94.render\|install" ~/rest_x96a/code/experiments/ecro_util96a/w96a.py | head -20'
  -> #!/bin/bash
cd ~/rest_x97/E4
. ~/rest_x95a/env.sh
$PY scripts/e4_ref.py ~/rest_x96a/code/experiments results/refa 4096 97700 4
touch logs/chain3.done
{'PMAT': (1270,), 'spp': (), 'seeds': (2,), 'e2_p_a': (1270, 64, 3), 'e2_o_a': (1270, 64, 3), 'e3_p_a': (1270, 64, 3), 'e3_o_a': (1270, 64, 3), 'e2_p_ ...[cut]... 87 as X
    L2 = torch.as_tensor(np.asarray(L2, np.float32), device=X.dev)
    L3 = op.apply(L2); L4 = op.apply(L3); L5 = op.apply(L4)
    g = op.gather(L2 + L3 + L4)
178:    # field split (installed field_src, 64 spp)
190:                        'installed_sha': MT._INST['patch_sha']['field_src'], 'pass': bool(c1 and c2 and c3 and hashlib.sha256(s2.encode()).hexdigest() == MT._INST['patch_sha']['field_src'])}
219:    T['T_READOUT'] = {'masks_sha_equal': m_me == m_95, 'render_max_rel': dmax, 'pass': bool(m_me == m_95 and (DEV or dmax <= 1e-4))}
492:def _install_kh():
529:    dr = _install_kh()
RUN [Find mask and reference render paths; create J4 server directory] ssh LabServer63 'grep -h "e4_extra\|e4_img" ~/rest_x97/E4/scripts/*.sh ~/rest_x97/E4/logs/*.log 2>/dev/null | head -5; ls ~/rest_x96a/results/renders | grep -c REF_ER; ls ~/rest_x96a/results/renders | grep REF_ER | head -3; ls ~/rest_x96a/results/pre ~/rest_x94/results/pre 2>/dev/null | head; grep -n "\"lights\"\|\"camera\"" -A 4 ~/rest_x96a/code/experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -30; mkdir -p ~/rest_x97/J4/out ~/rest_x97/J4/logs; nvidia-smi --query-gpu=memory.used
  -> 8
REF_ER_CK_none_CK2_T1_r0.json
REF_ER_CK_none_CK2_T1_r0.npz
REF_ER_CK_none_CK2_T1_r1.json
/home/nd/rest_x94/results/pre:
masks94.npz
pre.json
reused_refs.txt

/home/nd/rest_x96a/results/pre:
masks94.npz
7: "camera": {
8-  "origin": [
9-   0.3,
10-   -1.6,
11-   1.3
--
49: "lights": [
50-  {
51-   "pos": [
52-    -1.45,
53-    -0.6,
1 MiB, 0 %
RUN [Write the attribution script and run a 32-spp smoke on the server] mkdir -p /c/Projects/ReST-GI/experiments/audit97/J4 /c/Projects/ReST-GI/results/audit97/J4 && cd /c/Projects/ReST-GI/experiments/audit97/J4 && cat > j4_attr.py <<'EOF'
"""WL97 audit, track J4: attribution of the table-readout error at camera vertex 2 (decisive check for E4-F6).

The 'ecro' camera mode reads hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)] (pt86.py:228-233).  This script traces the SAME camera
prefix (x1, x2) and replaces the table lookup by fresh probe estimates (unchanged pt86
  -> j4_attr.py: Python script, ASCII text executable
1722 MiB
J4ATTR 97900 32 80.5
J4ATTR DONE

real	1m21.424s
user	1m24.369s
sys	0m4.025s
1 MiB
RUN [Write analysis script and run it on the smoke output] cd /c/Projects/ReST-GI/experiments/audit97/J4 && cat > j4_an.py <<'EOF'
"""WL97 audit, track J4: analysis of j4_attr.py outputs (numpy only).
Relative to the mask total of the stored path-traced reference (e1 + e2 + h, mean of 4 reps), per mask:
  signed (+- SE over reps), L1 of the mean difference, 16x16-block L1 (+ noise realization from the rep half-split),
  debiased relative L2 from the cross product of the two rep halves (unbiased for the structured part).
Usage: python j4_an.py <J4 outdir
  -> ret = um.true_divide(
/home/nd/rest_x97/J4/j4_an.py:55: RuntimeWarning: Mean of empty slice.
  cp = float(((x[:kx // 2].mean(0) - y[:ky // 2].mean(0))[m] * (x[kx // 2:].mean(0) - y[ky // 2:].mean(0))[m]).sum())
/home/nd/rest_x97/J4/j4_an.py:38: RuntimeWarning: Mean of empty slice.
  ha, hb = d[:k // ...[cut]... k16=0.00633 L1_block16_noise=nan rel_l2_debiased_vs_tot=nan
tail (came | C-A angular only (exact poin | signed=0.00004 L1=0.00229 L1_block16=0.00035 L1_block16_noise=nan rel_l2_debiased_vs_tot=nan
tail (came | D-B-C+A interaction | signed=0.00002 L1=0.00237 L1_block16=0.00039 L1_block16_noise=nan rel_l2_debiased_vs_tot=nan
tail (came | T-D stored table minus its e | signed=-0.00057 L1=0.04478 L1_block16=0.00572 L1_block16_noise=nan rel_l2_debiased_vs_tot=nan
tail (came | T-A stored table read minus  | signed=-0.00085 L1=0.04466 L1_block16=0.00580 L1_block16_noise=nan rel_l2_debiased_vs_tot=nan
RUN [Launch the detached attribution run (4 x 2048 spp) and check after 100 s] ssh LabServer63 'cd ~/rest_x97/J4 && . ~/rest_x95a/env.sh && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && setsid nohup $PY j4_attr.py ~/rest_x96a/code/experiments ~/rest_x96a/results/fields/CKF_none.npz out 2048 97900 4 16 > logs/attr.log 2>&1 < /dev/null & sleep 100; cat ~/rest_x97/J4/logs/attr.log | tail -3; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -u nd -f j4_attr.py'
  -> 1 MiB, 0 %
1340 MiB, 31 %
3558033
3558267
3558271
RUN [Find how WL95A defines the oracle-table floor] grep -n -i "floor" docs/worklogs/95A_*.md | head -40
RUN [Read WL96 section F] sed -n 190,240p docs/worklogs/96_WORKLOG_*.md
  -> 27:| T_REPRO94D | WL94D CK2 flight-floor values reproduced to 1.5e-7 under the old convention (W94, deployed state, M64(s16), exact receiver): receivers 1.378 / 0.713 / 0.940; box A 1.211 / 0.716 / 0.528; panel face 1.206 / −0.334 / 0.598 (ratio / corr / rel L1). WL94D CK2 learned-sender rel L1 (W94 ...[cut]... 67 tolerance;
136:  - the deployed M64 is near the floor (0.0152) because its draw happens to under-deliver;
138:  - The higher-order diagnostic G_s<seed> sits 0.013 (s8601) and 0.003 (s8602) above the floor.
155:- **Evidence for an actual angular representation floor.** On this workload, yes for the **binned outgoing sender table**: its expected delivery is biased independently of sampling, and the exact-direction evaluation closes the gap. This is evidence about the patch × 64-bin carrier for a grazing sharp lobe. It is not a statement about other bin layouts, adaptive bases or other scenes.
  -> - Sharp learned C and the resolvent's utility are unmeasured.
- Image-level relevance on CK2 is unresolved.

**What would overturn the verdict.**
- The §F smoke finds accuracy headroom H_acc above its bound in a sharp-reader region on two workloads, which would make it ONE_FALSIFIABLE_CANDIDATE (the ...[cut]... te authorisation) |
| U_max ≥ 1.25 at the largest N, H_acc within the bound | COST_ONLY scaling hypothesis |
| Stage 2 unresolved at the declared spp | the sharp-carrier choice has no image-level target on CK2 (reported) |

**Separable, optional (comparator selection, not architecture).** DQ g4 vs photons 40 k / 160 k vs ESC-R r4 on the same workloads, scored against PT e2_p (not the in-house oracle), with dirty-cache counts and ms per edit for the box edit and the slab rotation.

**Budget [INF].** Under about 1 GPU-hour, plus at most two new PT field references.

## Unresolved owner decisions
RUN [Find how WL95A defines the oracle-table floor] cd /c/Projects/ReST-GI && grep -n -i "floor" docs/worklogs/95A_*.md | head -40
RUN [Read WL96 section F] cd /c/Projects/ReST-GI && sed -n 190,240p docs/worklogs/96_WORKLOG_*.md
RUN [Read WL96A protocol decision rule and pre-freeze exposure] python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
print(list(p.keys()))
for k in ('question','stage_1','pre_freeze_exposure','decision_mapping','claims','scope','stage_2'):
    if k in p: print('==',k); print(json.dumps(p[k],indent=0)[:3500])
EOF
git log --format='%h %ad %s' --date=iso -3 -- experiments/ecro_util96a/protocol/ecro_util96a.json; ls results/wl96_calc | head -20
  -> ['protocol', 'version', 'title', 'brief', 'batch_class', 'question', 'preserved', 'families', 'stage_0_contract', 'stage_1', 'stage_2', 'stage_3', 'verdict', 'stops', 'not_authorised', 'execution', 'pre_freeze_exposure']
== question
"Given a fixed, physically accurate transport prefix, is there me ...[cut]... + Lambda x)",
"attribution": "capacity = in-sample half-a residual; target noise = half-a vs half-b in-sample gap relative to n0; objective mismatch = L2-trained vs L1 metric ordering; source generalization = held-out vs in-sample",
"ACCURACY_HEADROOM_CONFIRMED": "for rho = 24 or 64 on the held-out CK2 light: dI(fit) <= dI(XOD) - M_img AND boxA e3_o field rel L1(fit) < rel L1(XOD) - max(n0 / 2, 0.02)"
}
a1718f7 2026-11-16 02:04:49 +0900 WL96A freeze: learned higher-order transport utility ceiling (protocol ecro_util96a)
ck2_support.py
ck2_support_E.npy
ring_alloc.json
ring_alloc.py
scratch
RUN [Read WL96A protocol decision rule and pre-freeze exposure] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
print(list(p.keys()))
for k in ('question','stage_1','pre_freeze_exposure','decision_mapping','claims','scope','stage_2'):
    if k in p: print('==',k); print(json.dumps(p[k],indent=0)[:3500])
EOF
git log --format='%h %ad %s' --date=iso -3 -- experiments/ecro_util96a/protocol/ecro_util96a.json; ls results/wl96_calc | head -20
RUN [Check attribution run progress] ssh LabServer63 'tail -3 ~/rest_x97/J4/logs/attr.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; ls ~/rest_x97/J4/out 2>/dev/null'
  -> 1340 MiB, 1 %
RUN [Wait and check process CPU/GPU usage] ssh LabServer63 'sleep 150; tail -2 ~/rest_x97/J4/logs/attr.log; ps -o pid,etime,pcpu,rss,cmd -u nd | grep j4_attr | grep -v grep | head -3; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
  -> J4ATTR 97900 2048 167.3
J4ATTR 97901 2048 90.8
3558271       05:04  103 1125428 /home/nd/NeuralShaderDemo/external/8dna26/.venv/bin/python j4_attr.py /home/nd/rest_x96a/code/experiments /home/nd/rest_x96a/results/fields/CKF_none.npz out 2048 97900 4 16
1340 MiB, 0 %
RUN [Read WL96A decision code and the E_H noise-bound definition] sed -n 244,254p experiments/ecro_util96a/w96a.py; sed -n 300,320p experiments/ecro_util96a/w96a.py; sed -n 368,392p experiments/ecro_util96a/w96a.py; grep -n "def eh" -A 22 experiments/ecro_mat94/an94.py | head -45
  -> out = {'ext': EXT, 'R_pool': R_POOL}
    dep, pool = None, []
    cross = {}
    for name, us, sr in realizations(ctx):
        t1 = time.time()
        op, mp, st = explicit_op(ctx, us, sr)
        tp, to, _ = tail(op, mp, e2(rf))
        if name == 'deployed':
            dep = (tp, to)
           ...[cut]... '].shape[0] // 2
195-    da = lum(Rv['h'][:n].mean(0) - R0['h'][:n].mean(0)); db = lum(Rv['h'][n:].mean(0) - R0['h'][n:].mean(0))
196-    a, b = darm[m], dref[m]
197-    return {'slope': slope(a, b), 'corr': corr(a, b), 'rel_err': float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30)),
198-            'ref_half_noise_rel': float(0.5 * np.abs(da[m] - db[m]).sum() / max(np.abs(b).sum(), 1e-30)), 'pixels': int(m.sum())}
199-
200-
201-def image_metrics(R, M, model, arms):
202-    refs = {v: load_ref(R, v) for v in VARIANTS}
203-    regs = {'ROI_T': M['ROI_T'], 'DIFF': M['DIFF']}
204-    out = {}
RUN [Read WL96A decision code and the E_H noise-bound definition] cd /c/Projects/ReST-GI && sed -n 244,254p experiments/ecro_util96a/w96a.py; sed -n 300,320p experiments/ecro_util96a/w96a.py; sed -n 368,392p experiments/ecro_util96a/w96a.py; grep -n "def eh" -A 22 experiments/ecro_mat94/an94.py | head -45
RUN [Await the run and analyse 4 x 2048 spp] ssh LabServer63 'cd ~/rest_x97/J4; while pgrep -u nd -f "j4_attr.py" > /dev/null; do sleep 10; done; tail -3 logs/attr.log; . ~/rest_x95a/env.sh; $PY j4_an.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz ~/rest_x97/E4/results/refa out/j4_attr.json 2>&1 | tail -18'
  -> Command did not complete within its 400s timeout and was moved to the background (ID: b7x65aqjb). Output is being written to: C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\tasks\b7x65aqjb.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. It is running now: do not start a second copy while it runs. If it is no longer needed, stop it with TaskStop. No output had been read from it when it was moved. A program that buffers its output, or a pipe into one such as tail, can keep the output file empty until the command finishes. To check interim output, use Read on that file path.
RUN [Check run completion and find the stuck wait loop] ssh LabServer63 'tail -2 ~/rest_x97/J4/logs/attr.log; ps -u nd -o pid,etime,cmd | grep -v grep | grep "while pgrep" | head -5'
  -> J4ATTR 97903 2048 90.7
J4ATTR DONE
RUN [Read the background task output] cat "C:/Users/dna10/AppData/Local/Temp/claude/462fa2b2dbb71cdd/46d65e15-e083-4057-875f-9282fb68aa5e/tasks/b7x65aqjb.output" 2>/dev/null | tail -20; ssh LabServer63 'ps -u nd -o pid,etime,cmd | grep -v grep | grep -c "j4_"; ls ~/rest_x97/J4/out'
  -> 0
J4_s97900.npz
J4_s97901.npz
J4_s97902.npz
J4_s97903.npz
RUN [Analyse the 4 x 2048 spp attribution run] ssh LabServer63 'cd ~/rest_x97/J4; . ~/rest_x95a/env.sh; $PY j4_an.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz ~/rest_x97/E4/results/refa out/j4_attr.json 2>&1 | tail -18'
  -> sanity_A2_vs_REFA_h3 {"signed": 0.0001610837707183288, "signed_se": 9.443120868250468e-05, "ratio": 1.0011389897908098, "L1": 0.015875383004763457, "L1_block16": 0.0012030554401710367, "L1_block16_noise": 0.0012389409012514723, "rel_l2_debiased_vs_tot": -0.0015913035538456972}
sanity_tailA_vs_REFA_h ...[cut]... signed=0.00000 signed_se=0.00001 L1=0.00042 L1_block16=0.00008 L1_block16_noise=0.00006 rel_l2_debiased_vs_tot=0.00017
tail (came | D-B-C+A interaction | signed=0.00001 signed_se=0.00001 L1=0.00049 L1_block16=0.00009 L1_block16_noise=0.00009 rel_l2_debiased_vs_tot=0.00034
tail (came | T-D stored table minus its e | signed=-0.00088 signed_se=0.00007 L1=0.00549 L1_block16=0.00260 L1_block16_noise=0.00053 rel_l2_debiased_vs_tot=0.00355
tail (came | T-A stored table read minus  | signed=-0.00124 signed_se=0.00004 L1=0.00609 L1_block16=0.00358 L1_block16_noise=0.00050 rel_l2_debiased_vs_tot=0.00465
RUN [Start 4 more reps; read WL95A floor passages] ssh LabServer63 'cd ~/rest_x97/J4 && . ~/rest_x95a/env.sh && setsid nohup $PY j4_attr.py ~/rest_x96a/code/experiments ~/rest_x96a/results/fields/CKF_none.npz out 2048 97904 4 16 > logs/attr2.log 2>&1 < /dev/null & sleep 1; echo started'; cd /c/Projects/ReST-GI && sed -n 40,46p docs/worklogs/95A_*.md && sed -n 88,96p docs/worklogs/95A_*.md && sed -n 128,140p docs/worklogs/95A_*.md
  -> started
  - Deployed relation.
  - Pooled relations: 64 independent redraws of all state-ray offsets (rng 950000 + d) with matched `rw`, paired across arms. The pilot of 16 draws was extended to 64 by the predeclared boundary rule.
- **Receivers.** Every arm delivers only S1-mediated, exactly-2-even ...[cut]... oval of the binned bias and many relation draws; 64 draws were used here, i.e. 32,768 relation rays per receiver patch.
- **ROI.** The image-level effects are mostly below the 2 × 4,096 spp render resolution:
  - pooled X1 = floor (0.0124 vs 0.0118);
  - pooled M64 = 0.0245, +0.0127 over floor, below the 0.0167 tolerance;
  - the deployed M64 is near the floor (0.0152) because its draw happens to under-deliver;
  - learned s8602 (0.047) is the only arm clearly beyond tolerance.
  - The higher-order diagnostic G_s<seed> sits 0.013 (s8601) and 0.003 (s8602) above the floor.

## D. Interpretation
RUN [Inspect stored WL96A tails and decision record locally] python - <<'EOF'
import numpy as np, json, math
t=np.load('results/ecro_util96a/T96A.npz'); tl=np.load('results/ecro_util96a/tails.npz')
print('tails keys', {k:tl[k].shape for k in tl.files})
s1=json.load(open('results/ecro_util96a/s1.json')); print(list(s1.keys())); print(json.dumps(s1.get('decision'))[:600])
im=s1.get('images') or s1.get('img') or {}
for k in ('F|ROI_T','XOD|ROI_T','Z|ROI_T','n_img|ROI_T'):
    if k in im: print(k, json.dumps(im[k])[:400])
EOF
  -> tails keys {'dep_p': (1270, 64, 3), 'dep_o': (1270, 64, 3), 'pool_p': (16, 1270, 64, 3), 'pool_o': (16, 1270, 64, 3)}
['images', 'decision', 'ext']
{"H": -0.0008174718081380707, "U": -0.0007738942659385027, "Lo": -0.0008610493503376388, "n_img": 0.001463755474853132, "nb": 2.178877109978403e-05, " ...[cut]... 613297730652, "signed": 0.0, "H_share": 0.32070131161636745, "pixels": 58035, "noise_bound": 0.008149564761132743, "dI": 0.0, "nb": 0.0}
XOD|ROI_T {"E_H": 0.011972980064968896, "signed": 0.0005652028982229747, "H_share": 0.32070131161636745, "pixels": 58035, "noise_bound": 0.008149227468764422, "dI": 0.0006462836667150613, "nb": 2.178877109978403e-05}
Z|ROI_T {"E_H": 0.17528203757943991, "signed": -0.17788563793273324, "H_share": 0.32070131161636745, "pixels": 58035, "noise_bound": 0.007957399766629989, "dI": 0.17788563793273324, "nb": 0.0002740106301949294}
n_img|ROI_T 0.001463755474853132
RUN [Inspect stored WL96A tails and decision record locally] cd /c/Projects/ReST-GI && python - <<'EOF'
import numpy as np, json, math
t=np.load('results/ecro_util96a/T96A.npz'); tl=np.load('results/ecro_util96a/tails.npz')
print('tails keys', {k:tl[k].shape for k in tl.files})
s1=json.load(open('results/ecro_util96a/s1.json')); print(list(s1.keys())); print(json.dumps(s1.get('decision'))[:600])
im=s1.get('images') or s1.get('img') or {}
for k in ('F|ROI_T','XOD|ROI_T','Z|ROI_T','n_img|ROI_T'):
    if k in im: print(k, json.dumps(im[k])[:400])
EOF
RUN [Recompute e3_o explicit-vs-reference field errors by receiver material class] ssh LabServer63 '. ~/rest_x95a/env.sh; $PY - <<EOF
import numpy as np, math
z=np.load("/home/nd/rest_x96a/results/fields/CKF_none.npz"); t=np.load("/home/nd/rest_x96a/results/tables/T96A.npz")
PM=z["PMAT"]; print("PMAT counts", {int(v):int((PM==v).sum()) for v in np.unique(PM)})
NB,NZ,NPHI=64,4,16
WT=((np.arange(NB)//NPHI+0.5)/NZ)*(2*math.pi/NB)
ro=0.5*(z["e3_o_a"].astype(np.float64)+z["e3_o_b"]); xo=t["XOD"].astype(np.float64)-t["F"]+ro
ra,rb=z["e3_o_a"].astype(np.float64),z["e3_o_b"].astype(np
  -> PMAT counts {0: 256, 1: 42, 2: 32, 3: 832, 4: 54, 5: 54}
mat 0 n 256 flux XOD/ref 0.9981 half a/b 1.0000 exit relL1 0.0175 (n0 0.0059) table relL1 0.0364 (n0 0.0285) share of e3_o flux 0.1339
mat 1 n 42 flux XOD/ref 1.0015 half a/b 0.9987 exit relL1 0.0128 (n0 0.0111) table relL1 0.0903 (n0 0.0471) share of e3_o flux 0.0343
mat 2 n 32 flux XOD/ref 0.9538 half a/b 1.0020 exit relL1 0.0520 (n0 0.0038) table relL1 0.0784 (n0 0.0256) share of e3_o flux 0.0164
mat 3 n 832 flux XOD/ref 0.9958 half a/b 0.9997 exit relL1 0.0163 (n0 0.0051) table relL1 0.0279 (n0 0.0231) share of e3_o flux 0.7518
mat 4 n 54 flux XOD/ref 0.9934 half a/b 1.0010 exit relL1 0.0150 (n0 0.0091) table relL1 0.0357 (n0 0.0315) share of e3_o flux 0.0337
mat 5 n 54 flux XOD/ref 1.0014 half a/b 1.0015 exit relL1 0.0183 (n0 0.0033) table relL1 0.0277 (n0 0.0199) share of e3_o flux 0.0299
RUN [Extend the analysis script printing and metrics] cd /c/Projects/ReST-GI/experiments/audit97/J4 && python - <<'EOF'
s=open('j4_an.py',newline='').read()
s=s.replace("""         'L1_block16': blockL1(dm, m) / den, 'L1_block16_noise': blockL1(0.5 * (ha - hb), m) / den}
    cp = float((ha[m] * hb[m]).sum())""","""         'L1_block16': blockL1(dm, m) / den, 'L1_block16_noise': blockL1(0.5 * (ha - hb), m) / den,
         'L1_block8': blockL1(dm, m, 8) / den, 'L1_block8_noise': blockL1(0.5 * (ha - hb), m, 8) / den,
         'L1_block64': blockL1(dm,
  -> 0
RUN [Show MAP-11 finding] python experiments/audit97/_tools/show.py MAP-11 2>&1 | head -c 4000; python experiments/audit97/_tools/show.py --index 2>&1 | grep -i "MAP" | head -20
  -> F-MAP-01  MAP  M VERIFIED               Every evidence run executed its stated code revision
F-MAP-02  MAP  M VERIFIED               Protocol hashes in chain.status equal the LF content at the freeze commit (7/7 recorded)
F-MAP-03  MAP  - NONE                   Files changed or added in run dirs aft ...[cut]... ts exist only on one server account; no hash manifest
F-MAP-14  MAP  M UNTESTED_INFERENCE     The negative chain WL92-WL94D is dominated by non-decisive labels on one lineage; decisive results are learning-free and single-workload
E1-F7     E1   M CONDITIONALLY_VALID    At box-A delivery level, replacing the learned sender by M64 does not clearly reduce the map error for s8601
E2-F6     E2   - VERIFIED               Original X1 pooled map error 0.111 was a favourable 64-draw pool
B-11      B    - CONDITIONALLY_VALID    Label-mapping and wording gaps resolved by disclosure rather than amendment
RUN [Show MAP-11 finding] cd /c/Projects/ReST-GI && python experiments/audit97/_tools/show.py MAP-11 2>&1 | head -c 4000; python experiments/audit97/_tools/show.py --index 2>&1 | grep -i "MAP" | head -20
RUN [Show F-MAP-11 and check pre-freeze dev directories] python experiments/audit97/_tools/show.py F-MAP-11 2>&1 | head -c 4000; grep -n "0.001\|headroom about" docs/worklogs/96_WORKLOG_*.md | head -12; ssh LabServer63 'ls -la --time-style=long-iso ~/rest_x96adev ~/rest_x96aprobe 2>/dev/null | head -20; ls -la --time-style=long-iso ~/rest_x96adev/results 2>/dev/null | head -20; tail -1 ~/rest_x97/J4/logs/attr2.log'
  -> ### [F-MAP-11] (MAP) WL96A decisive tables existed before the freeze; thresholds were set with the prediction in hand
category=statistical_weakness material=True verdict=CONDITIONALLY_VALID confidence=high p0=['P0-4'] worklogs=['WL95A', 'WL96', 'WL96A']
- description: Dev chain ~/rest_x96adev 16:5 ...[cut]... 1-15 16:58 final.json
drwxr-xr-x  2 nd nd    4096 2026-11-15 16:57 fit
drwxr-xr-x  2 nd nd    4096 2026-11-15 16:57 lists
drwxr-xr-x  2 nd nd    4096 2026-11-15 16:58 logs
drwxr-xr-x  2 nd nd    4096 2026-11-15 16:55 pre
drwxr-xr-x  2 nd nd   12288 2026-11-15 16:57 renders
-rw-r--r--  1 nd nd     914 2026-11-15 16:55 s0.json
-rw-r--r--  1 nd nd    6226 2026-11-15 16:56 s1.json
-rw-r--r--  1 nd nd    6192 2026-11-15 16:57 s1_ext.json
-rw-r--r--  1 nd nd    4461 2026-11-15 16:57 s2fit.json
drwxr-xr-x  2 nd nd    4096 2026-11-15 16:57 tables
-rw-r--r--  1 nd nd   20592 2026-11-15 16:55 tails.json
RUN [Show F-MAP-11 and check pre-freeze dev directories] cd /c/Projects/ReST-GI && python experiments/audit97/_tools/show.py F-MAP-11 2>&1 | head -c 4000; grep -n "0.001\|headroom about" docs/worklogs/96_WORKLOG_*.md | head -12; ssh LabServer63 'ls -la --time-style=long-iso ~/rest_x96adev ~/rest_x96aprobe 2>/dev/null | head -20; ls -la --time-style=long-iso ~/rest_x96adev/results 2>/dev/null | head -20; tail -1 ~/rest_x97/J4/logs/attr2.log'
RUN [Check threshold provenance and what the probe/dev runs produced] grep -o "0\.005[^0-9][^\"]\{0,80\}" experiments/ecro_local93d/protocol/*.json experiments/ecro_attr95a/protocol/*.json experiments/ecro_s94d/protocol/*.json | head -8; ssh LabServer63 'ls --time-style=long-iso -la ~/rest_x96aprobe/results | head -30; ls ~/rest_x96aprobe/results/renders 2>/dev/null | wc -l; python3 -c "
import json;d=json.load(open(\"/home/nd/rest_x96adev/results/s1.json\"));print(json.dumps(d[\"decision\"]))"; tail -1 ~/rest_x97/J4/logs/attr2.log'
  -> experiments/ecro_local93d/protocol/ecro_local93d.json:0.005) on >= 2 of the available T1 conditions (E2, E5k995, E6) for both seeds AND vali
experiments/ecro_local93d/protocol/ecro_local93d.json:0.005) in >= half of the T1 condition x seed cells, or validation single-flight C2_bot
experiments/ecro_a ...[cut]...     4096 2026-11-15 17:00 renders
-rw-r--r-- 1 nd nd     916 2026-11-15 16:59 s0.json
-rw-r--r-- 1 nd nd    5312 2026-11-15 17:03 s2fit.json
drwxr-xr-x 2 nd nd    4096 2026-11-15 17:03 tables
-rw-r--r-- 1 nd nd   21532 2026-11-15 16:59 tails.json
-rw-r--r-- 1 nd nd 6817926 2026-11-15 16:59 tails.npz
12
{"H": -0.0008131618892027268, "U": -0.00018742642906758786, "Lo": -0.0014388973493378657, "n_img": 0.0015543263898880491, "nb": 0.0003128677300675695, "dI_XOD": 0.0007411645006853223, "R_realized": 0.00013196910469097725, "tol_E": 0.03190698273519557, "M_img": 0.005, "outcome": "S1_NO_HEADROOM"}
RUN [Check threshold provenance and what the probe/dev runs produced] cd /c/Projects/ReST-GI && grep -o "0\.005[^0-9][^\"]\{0,80\}" experiments/ecro_local93d/protocol/*.json experiments/ecro_attr95a/protocol/*.json experiments/ecro_s94d/protocol/*.json | head -8; ssh LabServer63 'ls --time-style=long-iso -la ~/rest_x96aprobe/results | head -30; ls ~/rest_x96aprobe/results/renders 2>/dev/null | wc -l; python3 -c "
import json;d=json.load(open(\"/home/nd/rest_x96adev/results/s1.json\"));print(json.dumps(d[\"decision\"]))"; tail -1 ~/rest_x97/J4/logs/attr2.log'
RUN [Analyse the 8 x 2048 spp attribution run] ssh LabServer63 'cd ~/rest_x97/J4; tail -1 logs/attr2.log; ls out | wc -l; . ~/rest_x95a/env.sh; $PY j4_an.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz ~/rest_x97/E4/results/refa out/j4_attr.json 2>/dev/null | cut -c1-420'
  -> <persisted-output>
Output too large (32.6KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bwy1p68ep.txt

Preview (first 2KB):
5
===== ROI_T 58035 rms/mean 1.136 {"A2 (3 camera events)": 0.1416, "A5-A2 (4..6)": 0.1791, "T ...[cut]... .00019 signed_se=+0.00000 L1=+0.00146 L1_block16=+0.00143 L1_block16_noise=+0.00000 L1_block8=+0.00146 L1_block8_noise=+0.00001 L1_block64=+0.00116 L1_block64_noise=+0.00000 l1_equiv_of_debiased_l2=+0.00098 rel_l2_debiased_vs_tot=+0.00188
fail {"A": 0.0, "B": 0.000373, "C": 7e-06, "D": 0.000236}
e2_c | D-A discretisation (pa | signed=+0.00181 signed_se=+0.00007 L1=+0.01864 L1_block16=+0.00612 L1_block16_noise=+0.00144 L1_block8=+0.00677 L1_block8_noise=+0.00279 L1_block64=+0.00470 L1_block64_noise=+0.00043 l1_equiv_of_debiased_l2=+0.00498 rel_l2_debiased_vs_tot=+0.00953
...
</persisted-output>
RUN [Check the second attribution batch status] ssh LabServer63 'cd ~/rest_x97/J4; cat logs/attr2.log | tail -5; ps -u nd -o pid,etime,pcpu,cmd | grep j4_attr | grep -v grep | cut -c1-80; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
  -> 3561980       02:14  0.0 /bin/bash -c cd ~/rest_x97/J4 && . ~/rest_x95a/env.sh &
3561982       02:14  106 /home/nd/NeuralShaderDemo/external/8dna26/.venv/bin/pyt
1340 MiB, 28 %
3561982, 1330 MiB
RUN [Condensed per-mask attribution table (4 reps)] F="C:/Users/dna10/.claude/projects/c--Projects-ReST-GI/46d65e15-e083-4057-875f-9282fb68aa5e/tool-results/bwy1p68ep.txt"; grep -E "^=====|^tail \||^e2_c \||^all  \|" "$F" | sed -E 's/L1_block8=[^ ]+ L1_block8_noise=[^ ]+ //; s/L1_block64=[^ ]+ L1_block64_noise=[^ ]+ //; s/signed_se/se/; s/L1_block16_noise/b16n/; s/L1_block16/b16/; s/l1_equiv_of_debiased_l2/l1eq/; s/rel_l2_debiased_vs_tot/l2d/' | grep -v "interaction" | cut -c1-200 | head -90
  -> ===== ROI_T 58035 rms/mean 1.136 {"A2 (3 camera events)": 0.1416, "A5-A2 (4..6)": 0.1791, "T3o read": 0.0623, "T3p read": 0.1156}
e2_c | D-A discretisation (pa | signed=+0.00181 se=+0.00007 L1=+0.01864 b16=+0.00612 b16n=+0.00144 l1eq=+0.00498 l2d=+0.00953
e2_c | B-A spatial only (rand | signed=+0.00 ...[cut]...  | signed=+0.00056 se=+0.00002 L1=+0.00200 b16=+0.00090 b16n=+0.00021 l1eq=+0.00049 l2d=+0.00084
all  | B-A spatial only (rand | signed=-0.00030 se=+0.00002 L1=+0.00159 b16=+0.00037 b16n=+0.00021 l1eq=+0.00018 l2d=+0.00031
all  | C-A angular only (exac | signed=+0.00086 se=+0.00001 L1=+0.00132 b16=+0.00088 b16n=+0.00010 l1eq=+0.00048 l2d=+0.00083
all  | T-D stored table minus | signed=+0.00023 se=+0.00000 L1=+0.00270 b16=+0.00034 b16n=+0.00022 l1eq=+0.00008 l2d=+0.00014
all  | T-A stored table read  | signed=+0.00079 se=+0.00002 L1=+0.00283 b16=+0.00088 b16n=+0.00022 l1eq=+0.00043 l2d=+0.00074
RUN [Condensed per-mask attribution table (4 reps)] cd /c/Projects/ReST-GI && F="C:/Users/dna10/.claude/projects/c--Projects-ReST-GI/46d65e15-e083-4057-875f-9282fb68aa5e/tool-results/bwy1p68ep.txt"; grep -E "^=====|^tail \||^e2_c \||^all  \|" "$F" | sed -E 's/L1_block8=[^ ]+ L1_block8_noise=[^ ]+ //; s/L1_block64=[^ ]+ L1_block64_noise=[^ ]+ //; s/signed_se/se/; s/L1_block16_noise/b16n/; s/L1_block16/b16/; s/l1_equiv_of_debiased_l2/l1eq/; s/rel_l2_debiased_vs_tot/l2d/' | grep -v "interaction" | cut -c1-200 | head -90
RUN [Find E4 render keys for new-field tables] grep -n "'Fn'\|'Pn'\|'Pc'\|Fn\b" experiments/audit97/E4/e4_tail.py experiments/audit97/E4/e4_img.py | head -20; ssh LabServer63 'ls ~/rest_x97/E4/results/renders | sed -E "s/_s[0-9]+\.npz//" | sort | uniq -c; tail -2 ~/rest_x97/J4/logs/attr2.log'
  -> experiments/audit97/E4/e4_img.py:121:keys = [k for k in ('F', 'Fa', 'Fb', 'XOD', 'dOab', 'XD', 'XOP', 'Z', 'Pc', 'Fn', 'Pn', 'XODn', 'D_S1', 'D_floor', 'D_sphere', 'dOn') if have(k)]
experiments/audit97/E4/e4_img.py:151:if all(k in Y for k in ('F', 'Pc', 'Z', 'XOD')):
experiments/audit97/E4/e4_img.p ...[cut]...  e3o_c = (Y['F'] - Y['Pc']); e3o_n = (Y['Fn'] - Y['Pn']); xo_c = (Y['XOD'] - Y['Pc']); xo_n = (Y['XODn'] - Y['Pn'])
experiments/audit97/E4/e4_img.py:174:             'ref (e2 + e3_p) ckf vs audit': stats(Y['Pc'] - Y['Pn'], m)['L1'],
experiments/audit97/E4/e4_img.py:175:             'F ckf vs F audit (whole table)': stats(Y['F'] - Y['Fn'], m)['L1'],
      4 A_D_S1
      4 A_D_floor
      4 A_D_sphere
      5 A_F
      4 A_Fa
      4 A_Fb
      4 A_Fn
      4 A_Pc
      4 A_Pn
      4 A_XD
      4 A_XOD
      4 A_XODn
      4 A_XOP
      4 A_Z
      4 A_dOab
      4 A_dOn
J4ATTR 97904 2048 166.9
RUN [Find E4 render keys for new-field tables] cd /c/Projects/ReST-GI && grep -n "'Fn'\|'Pn'\|'Pc'\|Fn\b" experiments/audit97/E4/e4_tail.py experiments/audit97/E4/e4_img.py | head -20; ssh LabServer63 'ls ~/rest_x97/E4/results/renders | sed -E "s/_s[0-9]+\.npz//" | sort | uniq -c; tail -2 ~/rest_x97/J4/logs/attr2.log'
RUN [Write table-read script against the independent audit reference tables] cd /c/Projects/ReST-GI/experiments/audit97/J4 && cat > j4_tread.py <<'EOF'
"""WL97 audit, track J4: table reads only, on the SAME camera prefixes as j4_attr.py (same seeds / launch size -> identical lanes).
Reads the CKF_none reference tables (T2c, T3c) and the INDEPENDENT audit reference tables of track E4 (TF_9740x.npz, 4 halves;
T2n, T3n = mean of 4; T2n01 / T2n23, T3n01 / T3n23 = means of two halves) through the unchanged table lookup convention
hl = throughput(x1) * B[patch(x2), bin(x2 -> x
  -> 0
J4ATTR 97904 2048 166.9
{'seed': (), 'spp': (), 'kmax': (), 'PMAT': (1270,), 'PSID': (1270,), 'PAREA': (1270,), 'k2': (1270, 64, 8, 3), 'k3': (1270, 64, 8, 3), 'k4': (1270, 64, 8, 3), 'k5': (1270, 64, 8, 3), 'k6': (1270, 64, 8, 3), 'cnt': (1270, 64)}
RUN [Verify the panel tag code and check run status] sed -n 168,184p experiments/ecro86a/pt86a.py; ssh LabServer63 'tail -3 ~/rest_x97/J4/logs/attr2.log; ps -u nd -o pid,etime,cmd | grep j4_attr | grep -v grep | cut -c1-60'
  -> wo_l = si.to_local(wb)
                    use &= wo_l.z > 1e-6
                    rb = si.spawn_ray(wb); rb.maxt = BEAM_REACH
                    use &= ~scene.ray_test(rb, use)
                    f = cl.eval(wi, dr.select(use, wo_l, mi.Vector3f(0, 0, 1)), use)
                    c += dr.select( ...[cut]...  v2p = dr.eq(mid, U(1))
                    for q in range(4):
                        sel = (v2p if q % 2 == 0 else ~v2p) if ((q < 2) == (k == 2)) else mi.Bool(False)
                        parts[q] += dr.select(sel, c, 0.0)
            if k == kmax:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            wo, pdf, wgt, valid = cl.sample(wi, u1, u2, active)
J4ATTR 97904 2048 166.9
J4ATTR 97905 2048 91.0
3561980       05:26 /bin/bash -c cd ~/rest_x97/J4 && . ~/res
3561982       05:26 /home/nd/NeuralShaderDemo/external/8dna2
RUN [Verify the panel tag code and check run status] cd /c/Projects/ReST-GI && sed -n 168,184p experiments/ecro86a/pt86a.py; ssh LabServer63 'tail -3 ~/rest_x97/J4/logs/attr2.log; ps -u nd -o pid,etime,cmd | grep j4_attr | grep -v grep | cut -c1-60'
WRITE c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\J4_p04_accuracy_headroom_adjudication.md
RUN [Spot-check E4-F5 numbers in fieldan.json] python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/fieldan.json'))
print(list(d.keys()))
c=d.get('e3_o_composition',{})
print(json.dumps(c)[:1500])
f=d.get('field_by_class_and_order',{})
print(list(f.keys())[:10])
print(json.dumps(f)[:1800])
EOF
  -> ['halves_audit', 'weights', 'event_accounting', 'alternatives', 'e3_o_composition', 'field', 'field_by_class_and_order', 'prefix_noise_propagation (explicit e3_o, deployed)', 'cross_independent', 'magnitude']
{"boxA": {"e3_o_flux": 133.3777078384294, "e3_o_share_of_e3": 0.8381690200306298, "last_fl ...[cut]... 0, "table_n0": 0.44233635824489304, "flux_ratio_se": 0.07382351814354188, "table_rel_l2_raw": 1.0, "table_rel_l2_noise": 0.4912590313200703, "table_rel_l2_debiased": 0.8710135269595222, "exit_rel_l2_raw": 1.0, "exit_rel_l2_noise": 0.10068047528767186, "exit_rel_l2_debiased": 0.9949188117107086, "corr_exit": NaN}, "XD(prefix all6)|last floor|sphere": {"flux_ratio": 1.0046976168608035, "exit_rel_l1": 0.056811492917617605, "exit_n0": 0.0041904290324064715, "table_rel_l1": 0.08858473719408354, "table_n0": 0.02716950818199912, "flux_ratio_se": 0.0010639450507448681, "table_rel_l2_raw": 0.1003526067
WRITE c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\J4_p04_accuracy_headroom_adjudication.md
RUN [Spot-check E4-F5 numbers in fieldan.json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/audit97/E4/fieldan.json'))
print(list(d.keys()))
c=d.get('e3_o_composition',{})
print(json.dumps(c)[:1500])
f=d.get('field_by_class_and_order',{})
print(list(f.keys())[:10])
print(json.dumps(f)[:1800])
EOF
RUN [Check whether batch 2 has finished] ssh LabServer63 'cd ~/rest_x97/J4; sleep 45; tail -2 logs/attr2.log; ps -u nd -o pid,etime,cmd | grep "j4_attr.py" | grep -v grep | cut -c1-40'
  -> J4ATTR 97905 2048 91.0
J4ATTR 97906 2048 90.8
3561980       06:26 /bin/bash -c cd ~/re
3561982       06:26 /home/nd/NeuralShade
RUN [Await batch 2, then start the table-read run and check after a minute] ssh LabServer63 'cd ~/rest_x97/J4; n=0; while [ ! -f out/J4_s97907.npz ] && [ $n -lt 16 ]; do sleep 10; n=$((n+1)); done; sleep 5; tail -1 logs/attr2.log; . ~/rest_x95a/env.sh; setsid nohup $PY j4_tread.py ~/rest_x96a/code/experiments ~/rest_x96a/results/fields/CKF_none.npz out 2048 97900 8 16 ~/rest_x97/E4/results/TF_97401.npz ~/rest_x97/E4/results/TF_97402.npz ~/rest_x97/E4/results/TF_97403.npz ~/rest_x97/E4/results/TF_97404.npz > logs/tread.log 2>&1 < /dev/null & sleep 60; tail -3 ~/rest_x97/
  -> J4ATTR DONE
J4TREAD 97902 2048 11.2
J4TREAD 97903 2048 11.2
J4TREAD 97904 2048 11.2
RUN [Analyse table reads against independent audit tables] cd /c/Projects/ReST-GI/experiments/audit97/J4 && cat > j4_an2.py <<'EOF'
"""WL97 audit, track J4: second analysis (numpy only) - uses the INDEPENDENT audit reference tables (track E4, 4 new field halves) read on
the same camera paths (j4_tread.py) to separate, without any noise model,
  * table estimation error of the CKF reference (Tc - Tn; per-table variance from the audit half split),
  * the part of 'table read minus exact continuation' that is common to two independent table references
    
  -> J4TREAD 97903 2048 11.2
J4TREAD 97904 2048 11.2
J4TREAD 97905 2048 11.2
J4TREAD 97906 2048 11.3
Traceback (most recent call last):
  File "/home/nd/rest_x97/J4/j4_an2.py", line 15, in <module>
    ZA = [np.load(f) for f in fa]; ZT = [np.load(f) for f in ft]
  File "/home/nd/rest_x97/J4/j4_an2.py", line 15, in <listcomp>
    ZA = [np.load(f) for f in fa]; ZT = [np.load(f) for f in ft]
  File "/home/nd/NeuralShaderDemo/external/8dna26/.venv/lib/python3.10/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'out/J4T_s97907.npz'
RUN [Re-run the table analysis after the read run finished] ssh LabServer63 'cd ~/rest_x97/J4; sleep 8; tail -2 logs/tread.log | cut -c1-1200; . ~/rest_x95a/env.sh; $PY j4_an2.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz out/j4_tables.json 2>&1 | head -72'
  -> LAYOUT {"r0": {"lanes": 841808, "max_dist_over_patch_diagonal": 0.9772595763206482, "mean": 0.36896294355392456, "min_abs_normal_dot": 1.0, "max_dist_sphere_m": 0.0}, "r1": {"lanes": 340933, "max_dist_over_patch_diagonal": 0.9568451046943665, "mean": 0.36858707666397095, "min_abs_normal_dot": 1.0, " ...[cut]... : 0.00165, "rel_l2": 0.00316}
   audit table noise of the 4-half mean, 0.5 (n01 - n23)                      {"signed": 0.00065, "L1": 0.00143, "L1_block16": 0.00139, "rel_l2": 0.00281}
   implied CKF table noise rel_l2 = sqrt(|Tc-Tn|^2 - noise_n^2)               0.00144
   Tc - A signed                                                              {"signed": 0.00083, "se": 3e-05}
   Tn - A signed                                                              {"signed": -0.0, "se": 3e-05}
   Tc - D signed                                                              {"signed": 0.00048, "se": 2e-05}
RUN [Table-level consistency of CKF vs independent audit reference tables] ssh LabServer63 '. ~/rest_x95a/env.sh; $PY - <<EOF
import numpy as np, math
z=np.load("/home/nd/rest_x96a/results/fields/CKF_none.npz")
tf=[np.load(f"/home/nd/rest_x97/E4/results/TF_9740{i}.npz") for i in (1,2,3,4)]
NB,NZ,NPHI=64,4,16
WT=((np.arange(NB)//NPHI+0.5)/NZ)*(2*math.pi/NB)
PM=z["PMAT"]
reg={"boxA":np.arange(1130,1184),"all":np.arange(1270),"lambert":np.nonzero(PM>=3)[0],"floor":np.arange(256)}
def fam(t,name):
    cls={"p":[1],"o":[0,2,3]}[name[-1]]; tags=[c*2+e for c in cls for e in (
  -> boxA     e2_p relL2: ckf mean noise(half split) 0.5236 | audit mean noise 0.4316/0.4417 | ckf-audit 0.6954 (expected 0.6786) | single-half sd ckf 0.7405 audit 0.8564 | flux ckf/audit 0.9924
boxA     e2_o relL2: ckf mean noise(half split) 0.0148 | audit mean noise 0.0111/0.0146 | ckf-audit 0.0178 (ex ...[cut]... 1.1650 | flux ckf/audit 0.9909
floor    e2_o relL2: ckf mean noise(half split) 0.0261 | audit mean noise 0.0186/0.0209 | ckf-audit 0.0313 (expected 0.0320) | single-half sd ckf 0.0369 audit 0.0401 | flux ckf/audit 1.0003
floor    e3_p relL2: ckf mean noise(half split) 0.1661 | audit mean noise 0.0933/0.0944 | ckf-audit 0.1912 (expected 0.1905) | single-half sd ckf 0.2349 audit 0.1866 | flux ckf/audit 1.0012
floor    e3_o relL2: ckf mean noise(half split) 0.0566 | audit mean noise 0.0515/0.0531 | ckf-audit 0.0765 (expected 0.0765) | single-half sd ckf 0.0800 audit 0.1034 | flux ckf/audit 0.9997
RUN [Final 8-rep analysis output (ROI_T, WHOLE, BOXA_PIX)] ssh LabServer63 'cd ~/rest_x97/J4; . ~/rest_x95a/env.sh; $PY j4_an.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz ~/rest_x97/E4/results/refa out/j4_attr.json 2>/dev/null > out/j4_attr.txt; $PY j4_an2.py out ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz out/j4_tables.json > out/j4_tables.txt 2>/dev/null; grep -E "^=====|^sanity|^T_vs|^table half|^fail|^tail \||^e2_c \||^all  \|" out/j4_attr.txt | sed -E "s/L1_block8=[^ ]+ L1_block8_noise=[^ ]+ //; s/L1_bl
