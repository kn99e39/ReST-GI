# Track MAP: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [Survey repo state, worklogs and experiment dirs] git status --short | head; git log --oneline | head -60; ls docs/worklogs | sed -n '80,200p'; ls docs/audits 2>/dev/null; ls experiments | tr '\n' ' '; echo; ls results | tr '\n' ' '
RUN [Check server reachability and run dirs] ssh -o BatchMode=yes -o ConnectTimeout=15 LabServer63 'hostname; ls -d ~/rest_* | tr "\n" " "; echo; ls ~/rest_x97 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
  -> <persisted-output>
Output too large (38.9KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\b5nzvemfe.txt

Preview (first 2KB):
efcf86a WL96A complete: LEARNED_RESOLVENT_UTILITY NO_MEANINGFUL_HEADROOM (CK2)
a1718f7 WL96A f ...[cut]... reeze: sharp-panel transport exposure smoke (protocol ecro_exp94b)
b34bdf9 WL94A (brief label WL94-R) complete: INCONCLUSIVE by the frozen mapping (seed disagreement)
5b7c0c8 WL94A (brief label WL94-R) freeze: material fidelity retention smoke (protocol ecro_mat94r)
f3b431f WL94 complete: H-M INCONCLUSIVE, H-A INCONCLUSIVE (INCONCLUSIVE_SMOKE_FIT) by the frozen mapping
e347904 WL94 freeze: ECRO material generalization vs sharp angular representation (protocol ecro_mat94)
16a8aa6 WL93D checkerboard addendum rev. 2: manifest hashes for sg JSONs and scene-iteration renders
...
</persisted-output>
  -> 63
/home/nd/rest_8dna_baseline /home/nd/rest_arch77a /home/nd/rest_ato74 /home/nd/rest_ato74_design /home/nd/rest_ato74_dev /home/nd/rest_attrib71b /home/nd/rest_b4rep /home/nd/rest_closure80d /home/nd/rest_cmp72 /home/nd/rest_cmp72_dev /home/nd/rest_compiled80c /home/nd/rest_compiled81 /home/nd/res ...[cut]... est_x93bdev /home/nd/rest_x93c /home/nd/rest_x93cdev /home/nd/rest_x93d /home/nd/rest_x93d_ck /home/nd/rest_x93d_ck2 /home/nd/rest_x93d_ck2dev /home/nd/rest_x93d_ckdev /home/nd/rest_x93ddev /home/nd/rest_x93dev /home/nd/rest_x93probe /home/nd/rest_x94 /home/nd/rest_x94b /home/nd/rest_x94bdev /home/nd/rest_x94c /home/nd/rest_x94cdev /home/nd/rest_x94cfeas /home/nd/rest_x94d /home/nd/rest_x94ddev /home/nd/rest_x94dev /home/nd/rest_x94dfeas /home/nd/rest_x94r /home/nd/rest_x94rdev /home/nd/rest_x95a /home/nd/rest_x95adev /home/nd/rest_x96a /home/nd/rest_x96adev /home/nd/rest_x96aprobe 
1 MiB, 0 %
RUN [Survey repo state, worklogs and experiment dirs] cd /c/Projects/ReST-GI && git status --short | head; git log --oneline | head -60; ls docs/worklogs | sed -n '80,200p'; ls docs/audits 2>/dev/null; ls experiments | tr '\n' ' '; echo; ls results | tr '\n' ' '
RUN [Check server reachability and run dirs] ssh -o BatchMode=yes -o ConnectTimeout=15 LabServer63 'hostname; ls -d ~/rest_* | tr "\n" " "; echo; ls ~/rest_x97 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
RUN [List worklogs, experiment code, protocols and local results per batch] ls docs/worklogs | grep -E '^(52|8[6-9]|9[0-6])' ; echo ---; for d in ecro86 ecro86a gm1_explicit87 gm1_relation88 gm1_relation89 gm1_gap90 gm1_c3urx1_91 ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d: $(ls experiments/$d 2>/dev/null | tr '\n' ' ')"; echo "   proto: $(ls experiments/$d/protocol 2>/dev/null | tr '\n' ' ')"; echo "   results: $(l
RUN [List server run dirs, COMMIT files and chain status] ssh -o BatchMode=yes LabServer63 'for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do echo "== $d"; ls ~/$d | tr "\n" " "; echo; for f in code/COMMIT COMMIT chain.status; do [ -f ~/$d/$f ] && { echo "-- $f:"; head -c 1500 ~/$d/$f; echo; }; done; done'
  -> 52_WORKLOG_LEARNED_INTERDOMAIN_TRANSPORT.md
86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md
86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md
89_WORKLOG_GM ...[cut]... son models pop q_s8601.json q_s8602.json repro.json 
== ecro_s94d: protocol run94d.sh s94.py 
   proto: ecro_s94d.json 
   results: audit.json chain.status final_s.json gpu.txt lam.json models pop repro.json s_s8601.json s_s8602.json sender 
== ecro_attr95a: a95.py protocol run95a.sh wl95calc 
   proto: ecro_attr95a.json 
   results: an.json chain.status deliv.npz gpu.txt main.json ops95a.npz s0.json 
== ecro_util96a: figw3_posthoc.py protocol run96a.sh w96a.py 
   proto: ecro_util96a.json 
   results: T96A.npz chain.status cost_s1.0.json final.json gpu.txt s0.json s1.json tails.json tails.npz
  -> <persisted-output>
Output too large (36.3KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bl73qkpvi.txt

Preview (first 2KB):
== rest_ecro86
chain.out chain.status chain.status.run1 chain2.out chain_frozen_run1.status ch ...[cut]... T02:30:57Z END ORA_E1_T0_r1 rc=0
2026-11-11T02:30:57Z END ORA_E1_T1_r1 rc=0
2026-11-11T02:30:57Z END ORA_E2_T0_r1 rc=0
2026-11-11T02:30:57Z END ORA_E2_T1_r1 rc=0
2026-11-11T02:30:57Z START S1_stage1
2026-11-11T02:31:03Z END S1_stage1 rc=0
2026-11-11T02:31:03Z GATE STAGE1 PASS
2026-11-11T02:31:03Z START GEN_train_0
2026-11-11T02:31:03Z START GEN_train_1
2
== rest_ecro86a
chain.out chain.status chain2.out chain2.status code env.sh gpu_util_trace.txt logs post86a.py results stageA stageB watch.out watch.sh 
-- code/COMMIT:
46593f0 + amendment1 1bd7df1 + amendment2 f6483ab

...
</persisted-output>
RUN [List worklogs, experiment code, protocols and local results per batch] cd /c/Projects/ReST-GI && ls docs/worklogs | grep -E '^(52|8[6-9]|9[0-6])' ; echo ---; for d in ecro86 ecro86a gm1_explicit87 gm1_relation88 gm1_relation89 gm1_gap90 gm1_c3urx1_91 ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d: $(ls experiments/$d 2>/dev/null | tr '\n' ' ')"; echo "   proto: $(ls experiments/$d/protocol 2>/dev/null | tr '\n' ' 
RUN [List server run dirs, COMMIT files and chain status] ssh -o BatchMode=yes LabServer63 'for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do echo "== $d"; ls ~/$d | tr "\n" " "; echo; for f in code/COMMIT COMMIT chain.status; do [ -f ~/$d/$f ] && { echo "-- $f:"; head -c 1500 ~/$d/$f; echo; }; done; done'
RUN [Extract commit and protocol hash lines from server chain status files] ssh -o BatchMode=yes LabServer63 'for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do echo "== $d"; ls ~/$d | tr "\n" " "; echo; for f in code/COMMIT COMMIT; do [ -f ~/$d/$f ] && { echo "-- $f: $(head -c 300 ~/$d/$f | tr "\n" " ")"; }; done; for s in ~/$d/chain*.status; do [ -f $s ] && { echo "-- $
  -> == rest_ecro86
chain.out chain.status chain.status.run1 chain2.out chain_frozen_run1.status chain_resume.out code data diag_v3.py env.sh gpu_util_trace.txt gpu_util_trace_run1.txt results watch.out watch_run1.out 
-- code/COMMIT: 7786d09 + amendment1 63972c8 
-- /home/nd/rest_ecro86/chain.status (18 ...[cut]... 8Z CHAIN_DONE
== rest_x95a
chain.out chain.status code data env.sh results 
-- code/COMMIT: c152120 
-- /home/nd/rest_x95a/chain.status (38 lines)
2026-11-14T21:02:21Z CHAIN_START dev= commit=c152120 protocol=fb136ad2da79fdca
2026-11-14T21:08:08Z CHAIN_DONE
   last: 2026-11-14T21:08:08Z CHAIN_DONE
== rest_x96a
chain.out chain.status code data env.sh figw3_posthoc.py results 
-- code/COMMIT: a1718f7 
-- /home/nd/rest_x96a/chain.status (28 lines)
2026-11-15T17:05:02Z CHAIN_START dev= commit=a1718f7 protocol=0a5936504fb8c345
2026-11-15T17:09:01Z CHAIN_DONE
   last: 2026-11-15T17:09:01Z CHAIN_DONE
RUN [Inspect code directory layout in each server run dir] ssh -o BatchMode=yes LabServer63 'mkdir -p ~/rest_x97/MAP; cd ~; for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do find $d/code -type f ! -path "*__pycache__*" ! -name "*.pyc" -printf "%p\t%TY-%Tm-%TdT%TH:%TM\t%s\n" | sort > /tmp/_x97_$d.lst; n=$(wc -l < /tmp/_x97_$d.lst); echo "== $d files=$n"; 
  -> == rest_ecro86 files=86
rest_ecro86/code rest_ecro86/code/experiments rest_ecro86/code/experiments/ecro86 rest_ecro86/code/experiments/ecro86/data rest_ecro86/code/experiments/ecro86/protocol rest_ecro86/code/experiments/fnto_closure80d rest_ecro86/code/experiments/fnto_closure80d/viz rest_ecro86/co ...[cut]... s/wl80b rest_x95a/code/docs/plans/wl77a rest_x95a/code/docs/plans/wl81 
== rest_x96a files=1283
rest_x96a/code rest_x96a/code/experiments rest_x96a/code/experiments/fnto_compiled84 rest_x96a/code/experiments/fnto_compiled84/protocol rest_x96a/code/experiments/fnto_persist59 rest_x96a/code/experiments/fnto_persist59/protocol rest_x96a/code/experiments/ecro_local93d rest_x96a/code/experiments/ecro_local93d/addendum_checker rest_x96a/code/experiments/ecro_local93d/protocol rest_x96a/code/experiments/ecro_exp94b rest_x96a/code/experiments/ecro_exp94b/protocol rest_x96a/code/experiments/fnto_rank66
RUN [Compute git blob hashes of deployed code files on the server] ssh -o BatchMode=yes LabServer63 'rm -f /tmp/_x97_*.lst; which git; git --version; cd ~; for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do ( cd $d/code && find . -type f ! -path "*__pycache__*" ! -name "*.pyc" ! -path "./docs/*" ! -path "./results/*" -print0 | sort -z | xargs -0 git hash-object 2
  -> /usr/bin/git
git version 2.43.0
rest_ecro86 86
rest_ecro86a 99
rest_gm187 92
rest_rel88 108
rest_rel89 120
rest_gap90 129
rest_x191 138
rest_x92 100
rest_x92a 106
rest_x92b 112
rest_x93 122
rest_x93a 131
rest_x93b 137
rest_x93c 143
rest_x93d 149
rest_x93d_ck2 1253
rest_x94 1259
rest_x94r 1263
rest_x94b 1271
rest_x94c 1274
rest_x94d 1277
rest_x95a 1282
rest_x96a 1283
AGENTS.md
COMMIT
docs
experiments
results
RUN [Recompute aligned blob hash manifests per run dir] ssh -o BatchMode=yes LabServer63 'export LC_ALL=C; cd ~; M=~/rest_x97/MAP; for d in rest_ecro86 rest_ecro86a rest_gm187 rest_rel88 rest_rel89 rest_gap90 rest_x191 rest_x92 rest_x92a rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do ( cd $d/code && find . -type f ! -path "*__pycache__*" ! -name "*.pyc" ! -path "./docs/*" ! -path "./results/*" | sort > $M/_l.tmp; git hash-object --stdin-paths < $M/_l.tm
  -> rest_ecro86 86 86
rest_ecro86a 99 99
rest_gm187 92 92
rest_rel88 108 108
rest_rel89 120 120
rest_gap90 129 129
rest_x191 138 138
rest_x92 100 100
rest_x92a 106 106
rest_x92b 112 112
rest_x93 122 122
rest_x93a 131 131
rest_x93b 137 137
rest_x93c 143 143
rest_x93d 149 149
rest_x93d_ck2 1253 1253
rest_x94 1259 1259
rest_x94r 1263 1263
rest_x94b 1271 1271
rest_x94c 1274 1274
rest_x94d 1277 1277
rest_x95a 1282 1282
rest_x96a 1283 1283
0a03f6a4c369c0ecd2f41f2eec4a6a5052f49319	./AGENTS.md	2026-11-15 17:04:49	67479
069e4baaab1b5fd3493195eaad435bf08200712a	./COMMIT	2026-11-15 17:04:56	8
3d97bf8859dd283de2d0ed9730e5ea27381ef952	./experiments/8dna_deformation_replication/8DNA_REPRESENTATION_AUDIT.md	2026-11-15 17:04:49	5890
-rw-r--r-- 1 nd nd 403951 Nov 15 17:46 blobs.tgz
RUN [Copy blob manifests locally] mkdir -p results/audit97/MAP/server_blobs experiments/audit97/MAP docs/audits/wl97_evidence_reliability && scp -q LabServer63:rest_x97/MAP/blobs.tgz results/audit97/MAP/server_blobs/ && cd results/audit97/MAP/server_blobs && tar xzf blobs.tgz && rm blobs.tgz && ls | wc -l; cd /c/Projects/ReST-GI; git config core.autocrlf; python --version; ssh LabServer63 date -u; date -u
  -> 23
true
Python 3.11.9
Sun Nov 15 17:46:15 UTC 2026
Sun Nov 15 17:46:16 UTC 2026
RUN [Copy blob manifests locally] cd /c/Projects/ReST-GI && mkdir -p results/audit97/MAP/server_blobs experiments/audit97/MAP docs/audits/wl97_evidence_reliability && scp -q LabServer63:rest_x97/MAP/blobs.tgz results/audit97/MAP/server_blobs/ && cd results/audit97/MAP/server_blobs && tar xzf blobs.tgz && rm blobs.tgz && ls | wc -l; cd /c/Projects/ReST-GI; git config core.autocrlf; python --version; ssh LabServer63 date -u; date -u
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\compare_blobs.py
RUN [Compare server code blobs with repo trees at freeze commits] python experiments/audit97/MAP/compare_blobs.py 2>&1 | cut -c1-330
  -> rest_ecro86    WL86            stated=63972c8 files=86 same=7 diff=78 extra=0 own=3/19 mtime[2026-11-11 01:27:39 .. 2026-11-11 02:23:51]
    DIFF  experiments/ecro86/analyze86.py  server=7453c6d9e4 (2026-11-11 02:23:36) repo=6399d2436f  server blob in: NO COMMIT
    DIFF  experiments/ecro86/data/m ...[cut]... ted=6d8b254 files=1274 same=1273 diff=0 extra=0 own=3/3 mtime[2026-11-14 14:54:01 .. 2026-11-14 14:54:11]
rest_x94d      WL94D           stated=a08c9bf files=1277 same=1276 diff=0 extra=0 own=3/3 mtime[2026-11-14 16:30:23 .. 2026-11-14 16:30:23]
rest_x95a      WL95A           stated=c152120 files=1282 same=1281 diff=0 extra=0 own=5/5 mtime[2026-11-14 21:01:38 .. 2026-11-14 21:01:38]
rest_x96a      WL96A           stated=a1718f7 files=1283 same=1282 diff=0 extra=0 own=3/3 mtime[2026-11-15 17:04:49 .. 2026-11-15 17:04:56]
wrote c:\Projects\ReST-GI\results\audit97\MAP\code_revision_check.json
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\compare_blobs.py
RUN [Compare server code blobs with repo trees at freeze commits] cd /c/Projects/ReST-GI && python experiments/audit97/MAP/compare_blobs.py 2>&1 | cut -c1-330
RUN [Check whether WL86/86A code differences are CRLF-only] ssh -o BatchMode=yes LabServer63 'export LC_ALL=C; M=~/rest_x97/MAP; for d in rest_ecro86 rest_ecro86a; do cd ~/$d/code; : > $M/norm_$d.tsv; find . -type f ! -path "*__pycache__*" ! -name "*.pyc" | sort | while read f; do h=$(tr -d "\r" < "$f" | git hash-object --stdin); c=$(grep -c $(printf "\r") "$f"); n=$(wc -l < "$f"); echo "$h	$f	$c	$n" >> $M/norm_$d.tsv; done; wc -l $M/norm_$d.tsv; done; head -3 $M/norm_rest_ecro86.tsv; ls -la ~/rest_ecro86a/ ~/rest_ecro86a/code | head -40; cat ~/rest_ecro
  -> 86 /home/nd/rest_x97/MAP/norm_rest_ecro86.tsv
99 /home/nd/rest_x97/MAP/norm_rest_ecro86a.tsv
5e45428e1f0e1b94a5f16f6abfa8128fb08dc355	./COMMIT	0	1
6399d2436f3ca936b62954b74f9fbebd85502d3f	./experiments/ecro86/analyze86.py	248	248
f29e6d5c4c50115632524fdb4e649e1dbefbb482	./experiments/ecro86/data/mat ...[cut]...  nd nd   968 Nov 11 05:17 watch.sh

/home/nd/rest_ecro86a/code:
total 16
drwxr-xr-x 3 nd nd 4096 Nov 11 05:17 .
drwxr-xr-x 7 nd nd 4096 Nov 11 05:30 ..
-rw-r--r-- 1 nd nd   50 Nov 11 05:53 COMMIT
drwxr-xr-x 9 nd nd 4096 Nov 11 05:53 experiments
2026-11-11T05:33:35Z END FIT_ecro_s8601 rc=0
2026-11-11T05:33:35Z END FIT_ecro_s8602 rc=0
2026-11-11T05:33:35Z START E2_analyze
2026-11-11T05:33:37Z END E2_analyze rc=0
2026-11-11T05:33:37Z CHAIN_DONE
A_baseline.log
A_baseline.rc
B_localize.log
B_localize.rc
C_beams.log
C_beams.rc
C_fields_E1_T1.log
C_fields_E1_T1.rc
C_fields_E2_T0.log
C_fields_E2_T0.rc
RUN [Test CRLF-only hypothesis for WL86/86A deployments] cat > experiments/audit97/MAP/compare_crlf.py <<'EOF'
#!/usr/bin/env python3
"""WL97 MAP: for the two run dirs whose deployed code is not byte-identical to
the repository (rest_ecro86, rest_ecro86a) test whether the difference is only
CRLF line endings. Input norm_<rundir>.tsv = git blob id of each server file
after `tr -d '\r'`, number of CR-containing lines, number of lines."""
import json, os, subprocess
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
B = os.
  -> rest_ecro86 {"n": 85, "crlf_files": 78, "per_commit_identical_after_cr_strip": {"7786d09": 80, "63972c8": 85}}
rest_ecro86a {"n": 98, "crlf_files": 88, "per_commit_identical_after_cr_strip": {"46593f0": 91, "1bd7df1": 95, "f6483ab": 97, "bd81442": 98}}
bd81442 2026-11-11T14:56:46+09:00 WL86A compl ...[cut]... 027d 2026-11-11T13:35:27+09:00 WL86 complete: ECRO v0 evidence run - workload ADMITTED (E1, held-out E2), representation ceiling PASS (oracle E_H 0
63972c8 2026-11-11T11:23:36+09:00 WL86 amendment 1 reconciliation: one amendment rule (amendment1.json: frame L1, outlier-pixel fraction, vertex agree
e914e3a 2026-11-11T11:16:48+09:00 WL86 amendment 1 (post hoc, labelled): frozen chain stopped at the Stage 0 gate on validation test V3 only (in-place
7786d09 2026-11-11T10:27:39+09:00 WL86 protocol v1 freeze + code: ECRO v0 (ER-conditioned learned global transport response) - room fixture family wit
RUN [Test CRLF-only hypothesis for WL86/86A deployments] cd /c/Projects/ReST-GI && cat > experiments/audit97/MAP/compare_crlf.py <<'EOF'
#!/usr/bin/env python3
"""WL97 MAP: for the two run dirs whose deployed code is not byte-identical to
the repository (rest_ecro86, rest_ecro86a) test whether the difference is only
CRLF line endings. Input norm_<rundir>.tsv = git blob id of each server file
after `tr -d '\r'`, number of CR-containing lines, number of lines."""
import json, os, subprocess
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), 
RUN [Compute protocol sha256 at freeze commits vs chain status] while read c p st; do h=$(git -c core.autocrlf=false show $c:$p | sha256sum | cut -c1-16); hh=$(git -c core.autocrlf=false show HEAD:$p | sha256sum | cut -c1-16); hc=$(git -c core.autocrlf=false show $c:$p | sed 's/$/\r/' | sha256sum | cut -c1-16); hw=$(sha256sum $p | cut -c1-16); echo "$c $p LF@freeze=$h LF@HEAD=$hh CRLF@freeze=$hc worktree=$hw status=$st"; done <<'EOF'
7786d09 experiments/ecro86/protocol/ecro86.json -
63972c8 experiments/ecro86/protocol/amendment1.json -
46593f0 experiments/ec
  -> 7786d09 experiments/ecro86/protocol/ecro86.json LF@freeze=ecca67246ef013d7 LF@HEAD=ecca67246ef013d7 CRLF@freeze=5ec57d62e9e82cca worktree=ecca67246ef013d7 status=-
63972c8 experiments/ecro86/protocol/amendment1.json LF@freeze=032a624e61ef10fd LF@HEAD=032a624e61ef10fd CRLF@freeze=88af3533ccf60ab7 wor ...[cut]... bf9fd6571 status=006941ac6cce4741
a08c9bf experiments/ecro_s94d/protocol/ecro_s94d.json LF@freeze=e0a389463e172bc7 LF@HEAD=e0a389463e172bc7 CRLF@freeze=99866314cf9d0bca worktree=1159baaa9449e7aa status=e0a389463e172bc7
c152120 experiments/ecro_attr95a/protocol/ecro_attr95a.json LF@freeze=fb136ad2da79fdca LF@HEAD=fb136ad2da79fdca CRLF@freeze=011479ed09461998 worktree=e1ab9b386f7701ee status=fb136ad2da79fdca
a1718f7 experiments/ecro_util96a/protocol/ecro_util96a.json LF@freeze=0a5936504fb8c345 LF@HEAD=0a5936504fb8c345 CRLF@freeze=1334aab361554d0c worktree=0a5936504fb8c345 status=0a5936504fb8c345
RUN [Compute protocol sha256 at freeze commits vs chain status] cd /c/Projects/ReST-GI && while read c p st; do h=$(git -c core.autocrlf=false show $c:$p | sha256sum | cut -c1-16); hh=$(git -c core.autocrlf=false show HEAD:$p | sha256sum | cut -c1-16); hc=$(git -c core.autocrlf=false show $c:$p | sed 's/$/\r/' | sha256sum | cut -c1-16); hw=$(sha256sum $p | cut -c1-16); echo "$c $p LF@freeze=$h LF@HEAD=$hh CRLF@freeze=$hc worktree=$hw status=$st"; done <<'EOF'
7786d09 experiments/ecro86/protocol/ecro86.json -
63972c8 experiments/ecro86/protocol/amendment1.jso
RUN [List commit history per experiment dir to detect post-freeze edits] for d in ecro86 ecro86a gm1_explicit87 gm1_relation88 gm1_relation89 gm1_gap90 gm1_c3urx1_91 ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d"; git log --format='  %h %cI %s' --name-status -- experiments/$d | grep -vE '^$' | awk '/^  [0-9a-f]{7} /{print substr($0,1,110); next} {printf "      %s\n",$0}' | head -60; done 2>&1 | head -400
  -> == ecro86
  63972c8 2026-11-11T11:23:36+09:00 WL86 amendment 1 reconciliation: one amendment rule (amendment1.json: fram
      D	experiments/ecro86/amend1_86.py
      M	experiments/ecro86/analyze86.py
      M	experiments/ecro86/run86.sh
  e914e3a 2026-11-11T11:16:48+09:00 WL86 amendment 1 (post hoc, ...[cut]... /ecro_attr95a.json
      A	experiments/ecro_attr95a/run95a.sh
      A	experiments/ecro_attr95a/wl95calc/wl95_calc.py
      A	experiments/ecro_attr95a/wl95calc/wl95_calc2.py
== ecro_util96a
  efcf86a 2026-11-16T02:13:06+09:00 WL96A complete: LEARNED_RESOLVENT_UTILITY NO_MEANINGFUL_HEADROOM (CK2)
      A	experiments/ecro_util96a/figw3_posthoc.py
  a1718f7 2026-11-16T02:04:49+09:00 WL96A freeze: learned higher-order transport utility ceiling (protocol ecr
      A	experiments/ecro_util96a/protocol/ecro_util96a.json
      A	experiments/ecro_util96a/run96a.sh
      A	experiments/ecro_util96a/w96a.py
RUN [List commit history per experiment dir to detect post-freeze edits] cd /c/Projects/ReST-GI && for d in ecro86 ecro86a gm1_explicit87 gm1_relation88 gm1_relation89 gm1_gap90 gm1_c3urx1_91 ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d"; git log --format='  %h %cI %s' --name-status -- experiments/$d | grep -vE '^$' | awk '/^  [0-9a-f]{7} /{print substr($0,1,110); next} {printf "      %s\n",$0}' | head -60; done 2
RUN [Size up living documents and worklogs, locate ECRO entries in roadmap] wc -l docs/RESEARCH_ROADMAP.md docs/Architecture.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md docs/RESEARCH_CENTRIC_TOPIC.md docs/worklogs/{86,86A,86B,87,88,89,90,91,92,92A,92B,93,93A,93B,93C,93D,94,94A,94B,94C,94D,95,95A,96,96A}_WORKLOG_*.md; grep -nE 'WL(8[6-9]|9[0-6])|ECRO|M64|CK2' docs/RESEARCH_ROADMAP.md | cut -c1-220 | head -80
  -> 171 docs/RESEARCH_ROADMAP.md
   116 docs/Architecture.md
    78 docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md
    58 docs/RESEARCH_CENTRIC_TOPIC.md
   291 docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
   262 docs/worklogs/86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
   267 docs/worklogs/86B_WORKL ...[cut]... TY_AND_FLIGHT_ATTRIBUTION.md) (2026-11-15; suffix of WL95; learning-free, CK2 'none', one grazing light): RANK16_DELIVERY **INADEQUATE_ON_TESTED_WORKLOAD** (a
155:- [WL96](worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md) (2026-11-15; design-only; 4 independent designers + 2 adversarial critics; CPU scratch only; material in docs/plans/wl96_spatio_angular_carrier
156:- [WL96A](worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md) (2026-11-16; suffix of WL96; learning-free, CK2 'none', one grazing light; protocol ecro_util96a, freeze a1718f7): LEARNED_RESOLVENT_UTILITY **N
RUN [Size up living documents and worklogs, locate ECRO entries in roadmap] cd /c/Projects/ReST-GI && wc -l docs/RESEARCH_ROADMAP.md docs/Architecture.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md docs/RESEARCH_CENTRIC_TOPIC.md docs/worklogs/{86,86A,86B,87,88,89,90,91,92,92A,92B,93,93A,93B,93C,93D,94,94A,94B,94C,94D,95,95A,96,96A}_WORKLOG_*.md; grep -nE 'WL(8[6-9]|9[0-6])|ECRO|M64|CK2' docs/RESEARCH_ROADMAP.md | cut -c1-220 | head -80
RUN [Read roadmap ECRO section in full] sed -n '120,171p' docs/RESEARCH_ROADMAP.md
  -> <persisted-output>
Output too large (37.1KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\b4ekk05kh.txt

Preview (first 2KB):
- **WL77A** selected no architecture candidate. It also found that the WL76/WL77 angular-failu ...[cut]... /ownership hypothesis** that addresses the measured learned-flight sharp deficit and compensation without sacrificing source-independent reuse, edited-geometry response or quality-matched economics. Fix the physical/angular interface, T0/T1 and held-out controls, independent references, matched explicit comparator and prospective gates before an authorised evidence run. Do not tune WL86 retrospectively, promote an explicit candidate as the project goal, or launch an automatic successor batch.

**GM1 evidence record (measured; nothing adopted; decisions remain human).**

...
</persisted-output>
RUN [Read roadmap ECRO section in full] cd /c/Projects/ReST-GI && sed -n '120,171p' docs/RESEARCH_ROADMAP.md
RUN [Read roadmap entries WL94D through WL96A and trailing section] sed -n '152,171p' docs/RESEARCH_ROADMAP.md | fold -w 4000 | head -c 22000
  -> - [WL94D](worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md) (2026-11-15; suffix of WL94; earlier labels unchanged): sharp sender oracle-supervised smoke. The learned S1 once-scattered sender (internal Bq = a c^T s) is the same quantity as the C2_srcS1 teacher s M64 (identity 8e-8; n ...[cut]... VIDENCE_STRATEGY.md#mandatory-baseline-terminology-and-comparison-gate).

## Updating this roadmap

Replace the current frontier when a user direction decision or completed gate changes it. Link the owning plan, protocol or Worklog; do not append a result diary. A newly proposed mechanism needs its own explicit decision. Preserve failed conditions, amendments and provenance at their original evidence locations.

The former batch program, resolved baseline prerequisites and chronological decision record are preserved in the [source archive](history/living_document_cleanup_2026-10-29/README.md).
RUN [Read roadmap entries WL94D through WL96A and trailing section] cd /c/Projects/ReST-GI && sed -n '152,171p' docs/RESEARCH_ROADMAP.md | fold -w 4000 | head -c 22000
RUN [Read WL95A and WL96A chain scripts] for f in experiments/ecro_util96a/run96a.sh experiments/ecro_attr95a/run95a.sh; do echo "=== $f"; cat $f; done
  -> === experiments/ecro_util96a/run96a.sh
#!/bin/bash
# WL96A chain (ONE detached chain, end to end): links (read-only) -> Stage 0 -> explicit tails -> renders -> Stage 1 analysis
# [-> predeclared extension: 6 field halves, tails, 12-rep renders, analysis] [-> Stage 2 only on S1_HEADROOM]
# -> Stage 3 ...[cut]... ocal J=()
  split -n l/6 -d "$O/lists/r95a.txt" "$O/lists/r95a_${tag}_"
  for f in "$O"/lists/r95a_${tag}_*; do J+=(R_${tag}_$(basename $f) "$MT render $R $f $DEV"); done
  pool 3 "${J[@]}"
}
run M1 "$A main $R $R92A $DEV"
renders p
run N1 "$A an $R $O/eval95a"
if grep -q '^AN EXTEND' $O/logs/N1.log; then
  log "EXTEND to 64 draws (rule: decision boundary within 2 SE)"
  run M2 "A95_EXTEND=1 $A main $R $R92A $DEV"
  rm -f "$O"/renders/CK_none_pool_*                    # pooled tables changed; deployed / floor / G tables are identical
  renders e
  run N2 "$A an $R $O/eval95a"
fi
log CHAIN_DONE
RUN [Read WL95A and WL96A chain scripts] cd /c/Projects/ReST-GI && for f in experiments/ecro_util96a/run96a.sh experiments/ecro_attr95a/run95a.sh; do echo "=== $f"; cat $f; done
RUN [Inspect WL95A/96A result dirs and symlink targets on server] ssh -o BatchMode=yes LabServer63 'cd ~; for r in rest_x95a rest_x96a; do echo "=== $r"; ls -la $r | sed -n "1,30p"; echo "-- results:"; ls -la $r/results | cut -c1-200; for s in fields pre renders tables wl94d wl95_calc; do echo "-- $s:"; ls -la $r/results/$s 2>/dev/null | cut -c30-260 | head -40; done; done' 2>&1 | head -230
  -> === rest_x95a
total 36
drwxr-xr-x   4 nd nd  4096 Nov 14 21:02 .
drwx------ 172 nd nd 12288 Nov 15 17:45 ..
-rw-r--r--   1 nd nd    96 Nov 14 21:02 chain.out
-rw-r--r--   1 nd nd  1557 Nov 14 21:08 chain.status
drwxr-xr-x   5 nd nd  4096 Nov 14 21:01 code
lrwxrwxrwx   1 nd nd    22 Nov 14 21:02 data ...[cut]... .json
ov 15 17:07 CK_none_G8601_X_CK2_T1_r2.npz
ov 15 17:07 CK_none_G8601_X_CK2_T1_r3.json
ov 15 17:07 CK_none_G8601_X_CK2_T1_r3.npz
ov 15 17:07 CK_none_G8602_X_CK2_T1_r0.json
ov 15 17:07 CK_none_G8602_X_CK2_T1_r0.npz
ov 15 17:08 CK_none_G8602_X_CK2_T1_r1.json
ov 15 17:08 CK_none_G8602_X_CK2_T1_r1.npz
ov 15 17:08 CK_none_G8602_X_CK2_T1_r2.json
-- tables:

Nov 15 17:05 .
Nov 15 17:08 ..
Nov 15 17:05 FZ_own_s8601.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8601.npz
Nov 15 17:05 FZ_own_s8602.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8602.npz
Nov 15 17:05 T96A.npz
-- wl94d:
-- wl95_calc:
RUN [Inspect dev/probe run dirs preceding WL95A/96A freezes] ssh -o BatchMode=yes LabServer63 'cd ~; for r in rest_x96adev rest_x96aprobe rest_x95adev wl95_calc; do echo "=== $r"; ls -la --time-style=+%m-%dT%H:%M:%S $r | cut -c1-160; [ -f $r/chain.status ] && { echo "-- chain.status"; cat $r/chain.status | cut -c1-200; }; [ -f $r/code/COMMIT ] && echo "COMMIT: $(cat $r/code/COMMIT)"; echo "-- results"; ls -la --time-style=+%m-%dT%H:%M:%S $r/results 2>/dev/null | cut -c1-160; done'
  -> === rest_x96adev
total 36
drwxr-xr-x   4 nd nd  4096 11-15T16:55:25 .
drwx------ 172 nd nd 12288 11-15T17:45:38 ..
-rw-r--r--   1 nd nd    96 11-15T16:55:25 chain.out
-rw-r--r--   1 nd nd  2926 11-15T16:58:24 chain.status
drwxr-xr-x   3 nd nd  4096 11-15T16:53:55 code
lrwxrwxrwx   1 nd nd    22 11-1 ...[cut]... -14T18:04:38 S1_ops.npz
drwxr-xr-x   2 nd nd    4096 11-14T18:05:30 __pycache__
-rw-r--r--   1 nd nd      89 11-14T18:18:14 run.log
-rw-r--r--   1 nd nd    2347 11-14T18:06:27 run2.log
-rw-r--r--   1 nd nd     374 11-14T18:27:52 run3.log
-rw-r--r--   1 nd nd   61130 11-14T18:18:14 wl95_calc.json
-rw-r--r--   1 nd nd    9156 11-14T18:02:21 wl95_calc.py
-rw-r--r--   1 nd nd    2592 11-14T18:06:27 wl95_calc2.json
-rw-r--r--   1 nd nd    3836 11-14T18:05:29 wl95_calc2.py
-rw-r--r--   1 nd nd     331 11-14T18:27:52 wl95_calc3.json
-rw-r--r--   1 nd nd    2174 11-14T18:18:47 wl95_calc3.py
-- results
RUN [Check whether worklogs disclose pre-freeze probes and dev runs] git log --format='%h %cI %s' -8 | cut -c1-120; echo; grep -n -iE 'probe|dev smoke|dev run|--dev|pre-freeze|before (the )?freeze|full-scale' docs/worklogs/96A_WORKLOG_*.md | cut -c1-400; echo ----95A; grep -n -iE 'probe|dev smoke|dev run|--dev|pre-freeze|before (the )?freeze|full-scale' docs/worklogs/95A_WORKLOG_*.md | cut -c1-400
  -> efcf86a 2026-11-16T02:13:06+09:00 WL96A complete: LEARNED_RESOLVENT_UTILITY NO_MEANINGFUL_HEADROOM (CK2)
a1718f7 2026-11-16T02:04:49+09:00 WL96A freeze: learned higher-order transport utility ceiling (protocol ecro_util96a)
dbdc0e7 2026-11-15T22:39:31+09:00 WL96 design-only: NO_VIABLE_LEARNED_MECHAN ...[cut]... xes before the freeze:** a module import order, numpy-bool JSON serialisation, and pool sizes.
----95A
114:- [FACT] The main stage is CPU-bound: the flights and receivers run in numpy. It was not probed for GPU utilisation at full scale before the freeze; only the dev path ran. This is a disclosed shortfall against the standing utilisation rule, though small in absolute time.
193:- **Environment.** LabServer63, RTX 3080 Ti 12 GB, `~/rest_x95a`, Python env `~/NeuralShaderDemo/external/8dna26/.venv`, `cuda_ad_rgb` with the OptiX path from `env.sh`. The dev run is `~/rest_x95adev` (not evidence).
RUN [Check whether worklogs disclose pre-freeze probes and dev runs] cd /c/Projects/ReST-GI && git log --format='%h %cI %s' -8 | cut -c1-120; echo; grep -n -iE 'probe|dev smoke|dev run|--dev|pre-freeze|before (the )?freeze|full-scale' docs/worklogs/96A_WORKLOG_*.md | cut -c1-400; echo ----95A; grep -n -iE 'probe|dev smoke|dev run|--dev|pre-freeze|before (the )?freeze|full-scale' docs/worklogs/95A_WORKLOG_*.md | cut -c1-400
RUN [Read WL96A provenance and pre-freeze exposure section] sed -n '225,263p' docs/worklogs/96A_WORKLOG_*.md | cut -c1-900
  -> - `masks94.npz` b9a7d46b1a23d8c4;
  - `REF_ER_CK_none_CK2_T1_r0..r3` (4 × 8,192 spp; WL94).
- **Seeds:**
  - explicit realizations 86001 (deployed) and 960000 + d (pooled), quadrature seed 87001 (+ 1000(d + 1));
  - renders 86600 + r;
  - Stage 0 field test 96001;
  - lifecycle sources rng 96500; ra ...[cut]... ; only the render spp was reduced. The figure therefore showed qualitatively that |XOD − F| and |XOP − F| were the smallest tail differences, and that G and Z were larger. No number was read.
  - All criteria were written before any WL96A code ran and were not changed.
- **Post-dev fixes before the freeze:** a module import order, numpy-bool JSON serialisation, and pool sizes.
- **No amendment.**
- **Post-hoc, figure only:** figW3 colours (`figw3_posthoc.py`).
- **The full-run `T_READOUT` record** stores the masks check under a key containing "sha". The pass flag includes it (masks identical).
RUN [Read WL96A provenance and pre-freeze exposure section] cd /c/Projects/ReST-GI && sed -n '225,263p' docs/worklogs/96A_WORKLOG_*.md | cut -c1-900
RUN [Diff WL96A dev/probe protocol and code against the frozen versions] ssh -o BatchMode=yes LabServer63 'cd ~; for p in rest_x96adev rest_x96aprobe; do sha256sum $p/code/experiments/ecro_util96a/protocol/ecro_util96a.json | cut -c1-16; done; echo "--- diff dev vs frozen (96A protocol)"; diff <(python3 -m json.tool rest_x96adev/code/experiments/ecro_util96a/protocol/ecro_util96a.json) <(python3 -m json.tool rest_x96a/code/experiments/ecro_util96a/protocol/ecro_util96a.json) | cut -c1-700; echo "--- diff probe vs frozen"; diff <(python3 -m json.tool rest_x96aprobe/co
  -> 1f5cd55e705592e6
1f5cd55e705592e6
--- diff dev vs frozen (96A protocol)
106c106,107
<     "pre_freeze_exposure": "All WL86-WL96 results are known, including WL95A's G_s8601 / G_s8602 ROI E_H (0.0251 / 0.0150 vs floor 0.0118) and WL96's cross-workload Amdahl estimate (1.09x / 1.16x from WL91 GM1 timi ...[cut]... ool 3 "${J[@]}"
---
>   pool 4 "${J[@]}"                                  # probe: one 4,096-spp render uses ~23 % GPU
54c54
<   for i in 0 1 2 3 4 5; do run FX$i "$A fext $R $i $DEV"; done
---
>   pool 3 FX0 "$A fext $R 0 $DEV" FX1 "$A fext $R 1 $DEV" FX2 "$A fext $R 2 $DEV" FX3 "$A fext $R 3 $DEV" FX4 "$A fext $R 4 $DEV" FX5 "$A fext $R 5 $DEV"   # CPU-bound accumulation (probe ~3 % GPU)
62c62
<   pool 2 F2a "$A s2ref $R 0 80 $DEV" F2b "$A s2ref $R 80 160 $DEV"
---
>   pool 4 F2a "$A s2ref $R 0 40 $DEV" F2b "$A s2ref $R 40 80 $DEV" F2c "$A s2ref $R 80 120 $DEV" F2d "$A s2ref $R 120 160 $DEV"
RUN [Locate OWN checkpoints in WL92A run dir] ssh -o BatchMode=yes LabServer63 'cd ~; echo "## OWN ckpts"; find rest_x92a -name "*own_s860*" | head -20; echo; find rest_x92a/results -maxdepth 2 -type d | head -30'
  -> ## OWN ckpts
rest_x92a/results/renders/I92A_own_s8602_E1_T1_LMIX_r1.npz
rest_x92a/results/renders/I92A_own_s8602_E4_T1_r0.npz
rest_x92a/results/renders/I92A_own_s8602_E2_T0_r0.npz
rest_x92a/results/renders/I92A_own_s8601_E2_T0_r1.npz
rest_x92a/results/renders/I92A_own_s8602_E2_T1_r0.npz
rest_x92a/re ...[cut]... 02_E2_T1_r1.npz
rest_x92a/results/renders/I92A_own_s8602_E4_T1_r1.npz
rest_x92a/results/renders/I92A_own_s8601_E4_T1_LSPOT4_r0.npz
rest_x92a/results/renders/I92A_own_s8601_E1_T1_LSPOT_r1.npz
rest_x92a/results/renders/I92A_own_s8602_E1_T1_LSPOT_r1.npz
rest_x92a/results/renders/I92A_own_s8601_E2_T1_r1.npz
rest_x92a/results/renders/I92A_STALE_own_s8601_E2_T1_r1.npz

rest_x92a/results
rest_x92a/results/tables
rest_x92a/results/logs
rest_x92a/results/lists
rest_x92a/results/renders
rest_x92a/results/fits
rest_x92a/results/viz
rest_x92a/results/models
rest_x92a/results/fields
rest_x92a/results/masks
RUN [List and hash checkpoints and WL94 shared artifacts] ssh -o BatchMode=yes LabServer63 'cd ~; ls -la --time-style=+%m-%dT%H:%M:%S rest_x92a/results/models | cut -c24-200; echo "## sha256"; sha256sum rest_x92a/results/models/*own* rest_ecro86/results/models/* 2>/dev/null | cut -c1-16,65-; ls rest_ecro86/results | tr "\n" " "; echo; echo "## W94 artifacts"; ls -la --time-style=+%m-%dT%H:%M:%S rest_x94/results/fields rest_x94/results/pre rest_x94/results/tables | cut -c24-220; ls -la --time-style=+%m-%dT%H:%M:%S rest_x94/results/renders | grep REF_ER 
  -> 4096 11-13T02:15:54 .
4096 11-13T02:26:03 ..
4590 11-13T02:15:43 own_s8601.json
5670 11-13T02:15:43 own_s8601.pt
4591 11-13T02:15:44 own_s8602.json
5670 11-13T02:15:44 own_s8602.pt
4584 11-13T02:15:47 sum_s8601.json
5670 11-13T02:15:47 sum_s8601.pt
4602 11-13T02:15:54 sum_s8602.json
5670 11-13T02:15 ...[cut]... R_CK_bx75_CK2_T1_r3.json
99373 11-14T10:46:06 REF_ER_CK_bx75_CK2_T1_r3.npz
  957 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r0.json
56790 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r0.npz
  959 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r1.json
56448 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r1.npz
  957 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r2.json
57152 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r2.npz
  959 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r3.json
56678 11-14T10:46:06 REF_ER_CK_none_CK2_T1_r3.npz

 Nov 14 11:02 .
 Nov 14 11:12 ..
 Nov 14 11:02 train_A
 Nov 14 11:04 train_B
 Nov 14 11:05 val_ck
 Nov 14 11:05 val_orig
RUN [Hash P0-critical artifacts and trace reused reference renders] ssh -o BatchMode=yes LabServer63 'cd ~; cat rest_x94/results/pre/reused_refs.txt; echo; echo "## REF json"; cat rest_x94/results/renders/REF_ER_CK_none_CK2_T1_r0.json; echo; cat rest_x94/results/renders/REF_ER_CK_U03_CK2_T1_r0.json; echo; echo "## CKF json"; cat rest_x94/results/fields/CKF_none.json; echo; echo "## sha"; sha256sum rest_x94/results/fields/CKF_none.npz rest_x94/results/pre/masks94.npz rest_x94/results/tables/FZ_own_s860?.npz rest_x94/results/renders/REF_ER_CK_none_CK2_T1_r?.npz re
  -> REF_ER_CK_none_CK2_T1_r0 89839c42be19acc94448042a54db9fb60da8ec9b2c284f3d24a162c1ea27a25e
REF_ER_CK_none_CK2_T1_r1 85e8176b0a500b9eda5965cba0da76a2d3966d442c6aa4c5e0b6eb424bd64633
REF_ER_CK_none_CK2_T1_r2 a701f32b714b7c2fa743c1186bab57c4468c0ca8d9093f91986f7576cf8f6ad8
REF_ER_CK_none_CK2_T1_r3 84fe5 ...[cut]... r1.npz
 958 11-14T08:40:19 REF_ER_CK_bx75_CK2_T1_r2.json
8412 11-14T08:40:19 REF_ER_CK_bx75_CK2_T1_r2.npz
 958 11-14T08:41:28 REF_ER_CK_bx75_CK2_T1_r3.json
9373 11-14T08:41:28 REF_ER_CK_bx75_CK2_T1_r3.npz
 957 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r0.json
6790 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r0.npz
 959 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r1.json
6448 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r1.npz
 957 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r2.json
7152 11-14T08:34:38 REF_ER_CK_none_CK2_T1_r2.npz
 959 11-14T08:36:00 REF_ER_CK_none_CK2_T1_r3.json
6678 11-14T08:36:00 REF_ER_CK_none_CK2_T1_r3.npz
RUN [Size key modules; check reference seeds and WL94 pre-check] wc -l experiments/ecro86/*.py experiments/ecro86a/pt86a.py experiments/gm1_explicit87/{quad87,xport87,pt87}.py experiments/ecro_mat94/*.py experiments/ecro_attr95a/a95.py experiments/ecro_util96a/w96a.py experiments/ecro_s94d/s94.py experiments/ecro_local93d/addendum_checker/*.py experiments/ecro_own92a/own92a.py experiments/ecro_local93d/ops93d.py | tail -50; ssh LabServer63 'cd ~; for r in 0 1 2 3; do python3 -c "import json;d=json.load(open(\"rest_x94/results/renders/REF_ER_CK_none_CK2_T1_r$r
  -> 248 experiments/ecro86/analyze86.py
    37 experiments/ecro86/diag_v3.py
   179 experiments/ecro86/econ86.py
   206 experiments/ecro86/ecro86.py
    62 experiments/ecro86/gen86.py
   151 experiments/ecro86/masks86.py
    34 experiments/ecro86/mats86.py
   137 experiments/ecro86/model86.py
   498 exp ...[cut]... f60a5ec17018e64d970381e5e68388c50a2e413"
  }
 },
 "geometry": {
  "roi_pixels": 58035,
  "diff_pixels": 9039,
  "diff_rule": "view factor >= 0.10",
  "panel_facing_bins_per_target_patch_mean": 10.907407407407407,
  "panel_face_patches": 9
 },
 "patch_sha256": {
  "trace": "57b30a3268542c6ff5db7ed705b8e3f97088e57f6a25ab1f8e56061186848c44",
  "field_src": "14f7367ceebdd0133f13e33632c14f5d000203ddc89e170a45b77abac11b3210"
 },
 "scene_sha256": "977ba122ab3232e7d428155c0963cff2f7764ce56704274832ee12d37b09de75",
 "protocol_sha256": "99e199f3847939e8d1178004ffa9cdb958e86787fa1850810bc31cba59bda16b"
}
RUN [Size key modules; check reference seeds and WL94 pre-check] cd /c/Projects/ReST-GI && wc -l experiments/ecro86/*.py experiments/ecro86a/pt86a.py experiments/gm1_explicit87/{quad87,xport87,pt87}.py experiments/ecro_mat94/*.py experiments/ecro_attr95a/a95.py experiments/ecro_util96a/w96a.py experiments/ecro_s94d/s94.py experiments/ecro_local93d/addendum_checker/*.py experiments/ecro_own92a/own92a.py experiments/ecro_local93d/ops93d.py | tail -50; ssh LabServer63 'cd ~; for r in 0 1 2 3; do python3 -c "import json;d=json.load(open(\"rest_x94/results/renders
READ c:\Projects\ReST-GI\experiments\ecro_mat94\mat94.py 
RUN [Extract imports, loads, seeds and function index of metric and measurement modules] for f in experiments/ecro_mat94/an94.py experiments/ecro_attr95a/a95.py experiments/ecro_util96a/w96a.py; do echo "=========== $f"; grep -nE "^import |^from |import [a-z0-9_]+ as|np\.load|jl\(|json\.load|torch\.load|os\.path\.join\(R[A-Z0-9_]*,|seed|86001|default_rng|def [a-zA-Z_0-9]+\(|sys\.path" $f | cut -c1-230; done
  -> =========== experiments/ecro_mat94/an94.py
1:"""WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.
4:  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints FIT / run-next only)
8:import glob, hashlib,  ...[cut]... d os.path.exists(os.path.join(R, 'results', 's2fit.json')):
697:        s2 = jl(os.path.join(R, 'results', 's2fit.json'))
698:        tl = jl(os.path.join(R, 'results', 'tails_ext.json' if ext else 'tails.json'))
714:    for fn in sorted(glob.glob(os.path.join(R, 'results', 'cost_s*.json'))):
715:        c = jl(fn); costs[c['s']] = c
732:    jd(res, os.path.join(R, 'results', 'final.json'))
740:def figures(R, outdir, ext):
773:    T = np.load(os.path.join(R, 'results', f'tails{sfx}.npz'))
783:    costs = sorted(glob.glob(os.path.join(R, 'results', 'cost_s*.json')))
785:        c = jl(costs[0])
RUN [Extract imports, loads, seeds and function index of metric and measurement modules] cd /c/Projects/ReST-GI && for f in experiments/ecro_mat94/an94.py experiments/ecro_attr95a/a95.py experiments/ecro_util96a/w96a.py; do echo "=========== $f"; grep -nE "^import |^from |import [a-z0-9_]+ as|np\.load|jl\(|json\.load|torch\.load|os\.path\.join\(R[A-Z0-9_]*,|seed|86001|default_rng|def [a-zA-Z_0-9]+\(|sys\.path" $f | cut -c1-230; done
RUN [Trace WL95 design-phase calc scripts and S1_ops provenance] git show --stat --format='%h %s' 255cc4f | grep -iE 'calc|\.py|npz|results' | head -30; ls -la results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>/dev/null | head -60; sha256sum results/wl95_calc/* 2>/dev/null | cut -c1-16,65-; for f in wl95_calc.py wl95_calc2.py; do git -c core.autocrlf=false show c152120:experiments/ecro_attr95a/wl95calc/$f | git hash-object --stdin; done; ssh LabServer63 'cd ~/wl95_calc; git hash-object wl95_calc.py wl95_calc2.py wl95_calc3.py; cat run.log run3.l
  -> results/wl95_calc/S1_ops.npz                       |  Bin 0 -> 2077680 bytes
 results/wl95_calc/wl95_calc.json                   | 2377 ++++++++++++++++++++
 results/wl95_calc/wl95_calc.py                     |  168 ++
 results/wl95_calc/wl95_calc2.json                  |  112 +
 results/wl95_calc/w ...[cut]... l95_calc/wl95_calc3.py
3010fe80bfda35ebfbe8fde2c4f87d6219fbe800
641d20ee20990932b7a12e7dfde3a024bdb70794
b5f8082f7fb36a838817f897fb2f63275fcce842
641d20ee20990932b7a12e7dfde3a024bdb70794
9e2bc97279fb9c9f5e7971f918633853fcf661f1
done S1_panel
done B_floor
done S2_sphere

real	15m51.951s
user	18m10.161s
sys	2m21.541s
{
 "train": {
  "field_rel_l1": 0.2631036732230663,
  "energy": 0.7868592119476995,
  "peak": 0.44309231927843473
 },
 "val": {
  "field_rel_l1": 0.4508283315620381,
  "energy": 0.5922260362489609,
  "peak": 0.3169238083307622
 },
 "op_l1_mean": 0.5046978417255796,
 "op_l1_range": [
RUN [Trace WL95 design-phase calc scripts and S1_ops provenance] cd /c/Projects/ReST-GI && git show --stat --format='%h %s' 255cc4f | grep -iE 'calc|\.py|npz|results' | head -30; ls -la results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>/dev/null | head -60; sha256sum results/wl95_calc/* 2>/dev/null | cut -c1-16,65-; for f in wl95_calc.py wl95_calc2.py; do git -c core.autocrlf=false show c152120:experiments/ecro_attr95a/wl95calc/$f | git hash-object --stdin; done; ssh LabServer63 'cd ~/wl95_calc; git hash-object wl95_calc.py wl95_calc2.py wl95_ca
RUN [Compare WL95 calc script versions across commits] for f in wl95_calc.py wl95_calc2.py wl95_calc3.py; do echo "$f results@255cc4f: $(git ls-tree 255cc4f results/wl95_calc/$f | awk '{print $3}')  HEAD: $(git ls-tree HEAD results/wl95_calc/$f | awk '{print $3}')"; done; git ls-tree c152120 experiments/ecro_attr95a/wl95calc/; git diff --stat 255cc4f:results/wl95_calc/wl95_calc.py c152120:experiments/ecro_attr95a/wl95calc/wl95_calc.py; git diff 255cc4f:results/wl95_calc/wl95_calc.py c152120:experiments/ecro_attr95a/wl95calc/wl95_calc.py | head -60
  -> wl95_calc.py results@255cc4f: 3010fe80bfda35ebfbe8fde2c4f87d6219fbe800  HEAD: 3010fe80bfda35ebfbe8fde2c4f87d6219fbe800
wl95_calc2.py results@255cc4f: 641d20ee20990932b7a12e7dfde3a024bdb70794  HEAD: 641d20ee20990932b7a12e7dfde3a024bdb70794
wl95_calc3.py results@255cc4f: 9e2bc97279fb9c9f5e7971f918633853fcf661f1  HEAD: 9e2bc97279fb9c9f5e7971f918633853fcf661f1
100644 blob 3010fe80bfda35ebfbe8fde2c4f87d6219fbe800	experiments/ecro_attr95a/wl95calc/wl95_calc.py
100644 blob 641d20ee20990932b7a12e7dfde3a024bdb70794	experiments/ecro_attr95a/wl95calc/wl95_calc2.py
RUN [Compare WL95 calc script versions across commits] cd /c/Projects/ReST-GI && for f in wl95_calc.py wl95_calc2.py wl95_calc3.py; do echo "$f results@255cc4f: $(git ls-tree 255cc4f results/wl95_calc/$f | awk '{print $3}')  HEAD: $(git ls-tree HEAD results/wl95_calc/$f | awk '{print $3}')"; done; git ls-tree c152120 experiments/ecro_attr95a/wl95calc/; git diff --stat 255cc4f:results/wl95_calc/wl95_calc.py c152120:experiments/ecro_attr95a/wl95calc/wl95_calc.py; git diff 255cc4f:results/wl95_calc/wl95_calc.py c152120:experiments/ecro_attr95a/wl95calc
RUN [Check CRLF in server WL95 calc script] ssh LabServer63 'cd ~/wl95_calc; tr -d "\r" < wl95_calc.py | git hash-object --stdin; grep -c $(printf "\r") wl95_calc.py wl95_calc2.py wl95_calc3.py; ls -la --time-style=full-iso . | cut -c24-120'
  -> 3010fe80bfda35ebfbe8fde2c4f87d6219fbe800
wl95_calc.py:168
wl95_calc2.py:0
wl95_calc3.py:0

 4096 2026-11-14 18:27:52.086601630 +0000 .
12288 2026-11-15 17:45:38.743621245 +0000 ..
77680 2026-11-14 18:04:38.789398312 +0000 S1_ops.npz
 4096 2026-11-14 18:05:30.151941994 +0000 __pycache__
   89 2026-11-14 18:18:14.192080977 +0000 run.log
 2347 2026-11-14 18:06:27.504431369 +0000 run2.log
  374 2026-11-14 18:27:52.115601559 +0000 run3.log
61130 2026-11-14 18:18:14.159081275 +0000 wl95_calc.json
 9156 2026-11-14 18:02:21.669611201 +0000 wl95_calc.py
 2592 2026-11-14 18:06:27.474431636 +0000 wl95_calc2.json
 3836 2026-11-14 18:05:29.488947890 +0000 wl95_calc2.py
  331 2026-11-14 18:27:52.086601630 +0000 wl95_calc3.json
 2174 2026-11-14 18:18:47.167783070 +0000 wl95_calc3.py
RUN [Read WL95 calc script inputs] sed -n '1,60p' results/wl95_calc/wl95_calc.py | cut -c1-220
  -> """WL95 Stage B offline CPU calculation (read-only inputs; writes only ~/wl95_calc/).
Matched-norm representability bounds for the M64 local operator (S1 panel mat 1; contrast B mat 0, S2 mat 2).
Metrics copied from experiments/ecro_s94d/s94.py:sender (operator, rel_l1_flux, energy_ratio, peak_ratio ...[cut]... nius NMF towards min L1 of the flux matrix, initialised from WL92 NMF16 (feasible nonneg rank-16)."""
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c] * ZZ
        Wf = Cm[c] * zb[:, None]; Hf = (Am[c] * zb[:, None]).T
        d = 1e-3 * V.mean(); eps = 1e-30
        for _ in range(outer):
            Om = 1.0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(inner):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + eps)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + eps)
        A[..., c] = (Wf @ Hf) / ZZ
    return A
RUN [Read WL95 calc script inputs] cd /c/Projects/ReST-GI && sed -n '1,60p' results/wl95_calc/wl95_calc.py | cut -c1-220
RUN [Read fixture injection module and the M64 local matrix constructor] echo "=== fx92.py"; cat experiments/ecro_attr92/fx92.py | cut -c1-240; echo "=== diag92.local_matrices"; grep -n "def local_matrices" -A 40 experiments/ecro_attr92/diag92.py | cut -c1-240
  -> === fx92.py
"""WL92 configuration: imports the frozen WL86 / WL86A / WL87 modules unchanged and adds the prospectively defined condition E4.

E4 is chosen from the never-rendered WL87 candidates E3b / E3c by the geometry / visibility-only rule of the protocol and recorded in
<root>/results/e4_choice ...[cut]... _energy': {str(q): int(np.searchsorted(np.cumsum(e2), q) + 1) for q in (0.9, 0.99, 0.999)},
170-                    'participation_rank': float((sv.sum() ** 2) / (sv ** 2).sum()),
171-                    'svd16': mat_metrics(X16 / zb[:, None] / zb[None, :], V, zb), 'nmf16': mat_metrics(Cn[m, c] @ An[m, c].T, V, zb),
172-                    'directional_albedo_mean': float((X.sum(1) * (2 * math.pi / NB) / zb).mean()),
173-                    'unweighted': {'eckart_young_rank16_fro': float(math.sqrt(eu[r:].sum())), 'note': 'plain Frobenius norm of M64 (dominated by the grazing ring); not used'}}
RUN [Read fixture injection module and the M64 local matrix constructor] cd /c/Projects/ReST-GI && echo "=== fx92.py"; cat experiments/ecro_attr92/fx92.py | cut -c1-240; echo "=== diag92.local_matrices"; grep -n "def local_matrices" -A 40 experiments/ecro_attr92/diag92.py | cut -c1-240
RUN [Read shared metric helper and scene constants] sed -n '1,70p' experiments/ecro_mat94/an94.py | cut -c1-260; sed -n '155,200p' experiments/ecro_mat94/an94.py | cut -c1-260; echo ==== scene86 consts; grep -nE "^(NB|NZ|NPHI|K|KAPPA|N_?[A-Z]*|[A-Z_]+) *=" experiments/ecro86/scene86.py | cut -c1-200 | head -50
  -> """WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.

  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints F ...[cut]... db[m]).sum() / max(np.abs(b).sum(), 1e-30)), 'pixels': int(m.sum())}


==== scene86 consts
11:HERE = os.path.dirname(os.path.abspath(__file__))
12:P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86.json')))
13:FX = P['fixture']
14:REP = P['representation']
16:NB = NZ * NPHI
17:KAPPA = REP['state_rays_per_bin']
18:K = NB * KAPPA
19:PSRC = REP['source_points_per_patch']
20:DOMEGA = 2 * math.pi / NB
23:MAT_CLASS = [2, 0, 1, 2, 2, 2]                 # object-level class 0 sharp, 1 moderate, 2 broad
24:N_RECT = 18                                    # 6 stationary rectangles + 2 x 6 box faces
RUN [Read shared metric helper and scene constants] cd /c/Projects/ReST-GI && sed -n '1,70p' experiments/ecro_mat94/an94.py | cut -c1-260; sed -n '155,200p' experiments/ecro_mat94/an94.py | cut -c1-260; echo ==== scene86 consts; grep -nE "^(NB|NZ|NPHI|K|KAPPA|N_?[A-Z]*|[A-Z_]+) *=" experiments/ecro86/scene86.py | cut -c1-200 | head -50
RUN [Confirm bin layout and trace users of the an94 weight vector] sed -n '14,22p' experiments/ecro86/scene86.py; python - <<'EOF'
import json
P=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print({k:P['representation'][k] for k in P['representation'] if not isinstance(P['representation'][k],(dict,list))})
EOF
grep -rnE "an94|AN\.W|AN\.ex\(|AN\.eh\(|\bWT\b *=|W94 *=|NPHI *= *8|NPHI, NZ" experiments/ecro_mat94 experiments/ecro_mat94r experiments/ecro_exp94b experiments/ecro_q94c experiments/ecro_s94d experiments/ecro_attr95a experiments/ecro_util96a
  -> REP = P['representation']
NZ, NPHI = REP['bins_z'], REP['bins_phi']
NB = NZ * NPHI
KAPPA = REP['state_rays_per_bin']
K = NB * KAPPA
PSRC = REP['source_points_per_patch']
DOMEGA = 2 * math.pi / NB
SPH_NPHI, SPH_NZ = 8, 4
# material ids: 0 B (floor), 1 S1 (panel), 2 S2 (sphere), 3 Lambert walls / ceil ...[cut]... 8:    res = {'note': 'W94 = an94.W as coded (NPHI 8, NZ 8); WT = true-layout cos dOmega; zb = true ring cosine', 'materials': {}}
experiments/ecro_util96a/w96a.py:2:Production code (pt86 / pt86a / ecro86 / model86 / quad87 / xport87 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.
experiments/ecro_util96a/w96a.py:19:import an94 as AN  # noqa: E402
experiments/ecro_util96a/w96a.py:33:WT = ZB * DOM                                          # corrected cos dOmega (4 rings x 16 sectors), sums to pi
experiments/ecro_util96a/w96a.py:368:            e = AN.eh(ref['h'], X[k], tot, m)
RUN [Confirm bin layout and trace users of the an94 weight vector] cd /c/Projects/ReST-GI && sed -n '14,22p' experiments/ecro86/scene86.py; python - <<'EOF'
import json
P=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print({k:P['representation'][k] for k in P['representation'] if not isinstance(P['representation'][k],(dict,list))})
EOF
grep -rnE "an94|AN\.W|AN\.ex\(|AN\.eh\(|\bWT\b *=|W94 *=|NPHI *= *8|NPHI, NZ" experiments/ecro_mat94 experiments/ecro_mat94r experiments/ecro_exp94b experiments/ecro_q94c experiments/ecro_s94d experiments/ecro_attr95
RUN [Read WL94B exposure audit use of the weight vector] sed -n '36,110p' experiments/ecro_exp94b/b94.py | cut -c1-250
  -> def jd(o, fn):
    AN.jd(o, fn)


# ------------------------------------------------------------------------------------------------ Stage A: split re-render
def agen(R, R86, which, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = C94.sel_files(R, 'train'); a ...[cut]...  1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ Stage A: audit
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def c_counts(seed, n=96, steps=None):
    steps = steps or C94.STEPS
    rng = np.random.default_rng(int(seed) + 94); order = []; c = np.zeros(n, np.int64)
    for _ in range(steps):
        if not order:
            order = rng.permutation(n).tolist()
        c[order.pop()] += 1
    return c


def allocate(pm_half, steps_half):
RUN [Read WL94B exposure audit use of the weight vector] cd /c/Projects/ReST-GI && sed -n '36,110p' experiments/ecro_exp94b/b94.py | cut -c1-250
RUN [Extract upstream artifact links from chain scripts WL92 to WL94D] for d in ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d; do for f in experiments/$d/run*.sh; do echo "=== $f"; grep -nE '^#|^R[0-9A-Z_]*=|^W[0-9A-Z]*=|\$R[0-9A-Z]+|\$W[0-9A-Z]*[/"]|ln -s|cp -n|cp ' $f | grep -vE '^\s*$' | cut -c1-260 | head -40; done; done
  -> <persisted-output>
Output too large (32.2KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\b21y86q1b.txt

Preview (first 2KB):
=== experiments/ecro_attr92/run92.sh
1:#!/bin/bash
2:# WL92 chain: ONE detached chain, end to  ...[cut]... .py gen $R $R86 train ${s%%:*} ${s##*:} $DEV"); done
72:  if [ -n "$DEV" ]; then j+=(I1_gen_val "$PY interv92.py gen $R $R86 val 0 8 $DEV" I1_gen_trainhq "$PY interv92.py gen $R $R86 trainhq 0 4 $DEV")
73:  else j+=(I1_gen_val "$PY interv92.py gen $R $R86 val 0 64" I1_gen_trainhq "$PY interv92.py gen $R $R86 trainhq 0 32"); fi
86:      for t in T0 T1; do for r in $REPS; do
87:        j+=(I2_refER_${t}_$r "$PY interv92.py ref $R $R86 E4_$t ER $((92610+r)) $RSPP REF_ER_E4_${t}_r$r" I2_refP_${t}_$r "$PY interv92.py ref $R $R86 E4_$t P $((92630+r)) $RSPP REF_P_E4_${t}_r$r")
...
</persisted-output>
RUN [Extract upstream artifact links from chain scripts WL92 to WL94D] cd /c/Projects/ReST-GI && for d in ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d; do for f in experiments/$d/run*.sh; do echo "=== $f"; grep -nE '^#|^R[0-9A-Z_]*=|^W[0-9A-Z]*=|\$R[0-9A-Z]+|\$W[0-9A-Z]*[/"]|ln -s|cp -n|cp ' $f | grep -vE '^\s*$' | cut -c1-260 | head -40; done; done
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\extract_deps.py
RUN [Run static dependency extraction and show module-level edges] python experiments/audit97/MAP/extract_deps.py 2>&1 | cut -c1-250 > results/audit97/MAP/static_deps.txt; wc -l results/audit97/MAP/static_deps.txt; grep -vE "^   artifact" results/audit97/MAP/static_deps.txt | cut -c1-600
  -> 279 results/audit97/MAP/static_deps.txt
== WL86 (ecro86)
   modules: fnto_compiled81/layered81.py[1], fnto_compiled83/closure83_dr.py[1], fnto_compiled83/render83.py[1]
   seed/state refs: 10 ; metric-helper refs: 0
== WL86A (ecro86a)
   modules: ecro86/ecro86.py[6], ecro86/model86.py[5], ecro86/pt8 ...[cut]... ric-helper refs: 3
== WL95A (ecro_attr95a)
   modules: ecro86/ecro86.py[2], ecro86/model86.py[3], ecro_mat94/an94.py[1], ecro_mat94/mat94.py[1], ecro_mat94/viz94.py[1], gm1_explicit87/quad87.py[1]
   seed/state refs: 2 ; metric-helper refs: 6
== WL96A (ecro_util96a)
   modules: ecro86/ecro86.py[3], ecro86/model86.py[1], ecro86/pt86.py[3], ecro86/scene86.py[2], ecro_attr92/fx92.py[5], ecro_mat94/an94.py[1], ecro_mat94/mat94.py[1], gm1_explicit87/quad87.py[2], gm1_explicit87/xport87.py[6]
   seed/state refs: 7 ; metric-helper refs: 4
wrote c:\Projects\ReST-GI\results\audit97\MAP\static_deps.json
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\extract_deps.py
RUN [Run static dependency extraction and show module-level edges] cd /c/Projects/ReST-GI && python experiments/audit97/MAP/extract_deps.py 2>&1 | cut -c1-250 > results/audit97/MAP/static_deps.txt; wc -l results/audit97/MAP/static_deps.txt; grep -vE "^   artifact" results/audit97/MAP/static_deps.txt | cut -c1-600
RUN [Show extracted upstream artifact edges] awk '/^== /{h=$0} /^   artifact/{print h" |"$0}' results/audit97/MAP/static_deps.txt | sed 's/^== //; s/   artifact //' | cut -c1-200
  -> WL87 (gm1_explicit87) |WL86:results/fields/F_E1_T1.npz                                        <- tests87.py:325
WL87 (gm1_explicit87) |WL86:results/fields/f'F_{cfg}.npz                                      <- analyze87.py:111
WL87 (gm1_explicit87) |WL86:results/final.json                             ...[cut]...                                            <- run96a.sh:33
WL96A (ecro_util96a) |WL94:results/renders/REF_ER_CK_none_*                                  <- run96a.sh:34
WL96A (ecro_util96a) |WL94:results/tables/FZ_own_s860?.npz                                   <- run96a.sh:35
WL96A (ecro_util96a) |WL95A:results/pre/masks94.npz                                          <- w96a.py:212
WL96A (ecro_util96a) |WL95A:results/renders/CK_none_floor_X_CK2_T1_r0.npz                    <- w96a.py:217
WL96A (ecro_util96a) |WL95A:results/tables/T95A.npz                                          <- w96a.py:192
RUN [Show extracted upstream artifact edges] cd /c/Projects/ReST-GI && awk '/^== /{h=$0} /^   artifact/{print h" |"$0}' results/audit97/MAP/static_deps.txt | sed 's/^== //; s/   artifact //' | cut -c1-200
RUN [Read the WL89 post-freeze amendment commit and protocol diff] git show 5d6bb5c --stat --format='%h %cI%n%B' | cut -c1-1500 | head -60; echo; git diff 7515a29 5d6bb5c -- experiments/gm1_relation89/protocol/gm1_relation89.json | cut -c1-1800 | head -60
  -> 5d6bb5c 2026-11-12T15:01:21+09:00
WL89 POST-FREEZE AMENDMENT 1: R1 selective = full gate judged on the operational state (valid samples, operator tensors, u state, segments, MIS; bitwise) instead of a digest that also hashed non-semantic stored bins of invalid samples - code-level read-path audit an ...[cut]... Execution evidence (results/amend1/audit89.json): operational digest, every operator tensor, u state, segments and validity equal bitwise; deterministic CPU transport outputs (random field, q, 4-order recurrence) equal bitwise; null edit and move-and-return equal; perturbing all 157,538 invalid entries (dbin random / 255, cell random / max, w NaN) leaves operator tensors, outputs (finite) and a later selective update bitwise unchanged. The earlier session diagnosis (inactive-lane garbage) was wrong and is corrected by this evidence.", "old_predicate": "tests88.T_EDIT: rel88.SAM.digest() equ
 }
RUN [Read the WL89 post-freeze amendment commit and protocol diff] cd /c/Projects/ReST-GI && git show 5d6bb5c --stat --format='%h %cI%n%B' | cut -c1-1500 | head -60; echo; git diff 7515a29 5d6bb5c -- experiments/gm1_relation89/protocol/gm1_relation89.json | cut -c1-1800 | head -60
RUN [Timeline of WL89 chain stages around the stop and resume] ssh LabServer63 'cd ~/rest_rel89; awk "{print \$1, \$2, \$3, \$4, \$5}" chain.status | grep -E "START|END|GATE|STOP|RESUME|CHAIN|W1|VERDICT" | awk "{t=substr(\$1,12,8); printf \"%s %s %s %s|\", t, \$2, \$3, \$4} END{print \"\"}" | tr "|" "\n" | grep -vE "^[0-9:]+ START" | awk "NR<=400" | cut -c1-70 | paste - - - - | head -90; echo; ls -la --time-style=+%m-%dT%H:%M:%S results | cut -c24-140; ls results/orig_stop 2>/dev/null | head; ls -la --time-style=+%m-%dT%H:%M:%S results/fields | cut -c24-140
  -> 03:55:44 CHAIN_START dev= commit=7515a29	03:56:35 END S0_wscreen rc=0	04:06:29 END QER_V3_T0_r0 rc=0	04:06:29 END QER_V3_T1_r0 rc=0
04:06:29 END QER_V3_T0_r1 rc=0	04:06:29 END QER_V3_T1_r1 rc=0	04:06:29 END QER_V3_T0_r2 rc=0	04:06:29 END QER_V3_T1_r2 rc=0
04:06:29 END QER_V3_T0_r3 rc=0	04:06:29 END  ...[cut]... bles
262 11-12T06:02:37 tests88.json
526 11-12T06:02:59 tests89.json
096 11-12T06:36:46 viz
135 11-12T04:06:42 w_choice.json
225 11-12T03:56:34 w_screen.json
083 11-12T04:06:41 wq_V3.json
A_verdict.log
COMMIT_freeze
S0_tests.log
chain.status
tests88.json
tests89.json

4096 11-12T04:35:32 .
4096 11-12T06:36:30 ..
0645 11-12T04:34:52 F89_W1_T0.npz
1793 11-12T04:34:57 F89_W1_T1.npz
5751 11-12T04:35:20 F89_W1_T1_LMIX89.npz
2976 11-12T04:35:33 F89_W1_T1_LSPOT89.npz
1232 11-12T04:35:11 F89_W2_T0.npz
8334 11-12T04:35:16 F89_W2_T1.npz
5760 11-12T04:35:28 ORD_W1_T0.npz
3939 11-12T04:35:32 ORD_W1_T1.npz
RUN [Locate WL96 Amdahl estimate sources] ls -R results/wl96_calc/scratch | head -60; grep -n -E "1\.09|1\.16" docs/worklogs/96_WORKLOG_*.md | cut -c1-500 | head -12
  -> results/wl96_calc/scratch:
critic_cost
critic_phys
d1
d2
d3
d4

results/wl96_calc/scratch/critic_cost:
dirty.py

results/wl96_calc/scratch/critic_phys:
rankw.json
rankw.log
rankw.py

results/wl96_calc/scratch/d1:
deliv96.py
deliv96_hp100_nr2.json
deliv96_hp40_nr4.json
er_np.py
photon2.json
photon2.l ...[cut]... g
photon96.py
rank96.json
rank96.py
rep.py
val2.py
val_m64.py
var96.json
var96.py

results/wl96_calc/scratch/d2:
ck2_surrogate.py
detquad.py
er_np.py
estimators.py
rank_transfer.json
rank_transfer.py
truth_area.py
variance.py

results/wl96_calc/scratch/d3:
fbc.py
photon.py
portal2.py
rankM.py
res_exact_L0.json
res_exact_L1.json
res_fbc_L0.json
res_fbc_L1.json
res_ladder_L0_0.json
res_ladder_L0_1.json
res_ladder_L0_2.json
res_ladder_L0_3.json
res_ladder_L0_4.json
res_ladder_L0_5.json
res_ladder_L0_6.json
145:  - edit→frame 124.7 / 114.1 = **1.09×**;
146:  - source→frame 75.8 / 65.2 = **1.16×**;
RUN [Locate WL96 Amdahl estimate sources] cd /c/Projects/ReST-GI && ls -R results/wl96_calc/scratch | head -60; grep -n -E "1\.09|1\.16" docs/worklogs/96_WORKLOG_*.md | cut -c1-500 | head -12
RUN [Read WL96 cost-ceiling section and list WL91 result files] sed -n '128,160p' docs/worklogs/96_WORKLOG_*.md | cut -c1-700; ls results/gm1_c3urx1_91/results | head -30
  -> ## C. Adversarial critique (strongest objections)

**Physics / representation.**
- **Ownership gaps** (physics critic §4):
  - d2: relayed e3_p sits inside b2 and also in Λ's e3 target, so it is double counted.
  - d1: incidence on the panel from broad senders is never relayed, so part of e3_p is un ...[cut]... 4 ms wall time against 2 ms GPU time.
- **Explicit carriers carry real costs that the designers underestimated:**
  - DQ / CIS per-edit dirty pairs are 60 % on the CK2 box shift (≈ 136–140 ms per edit, about 1.8× ECRO);
  - α⁻⁴ pairs;
  - per-light photon cost ∝ L·F_s;
  - ESC-R memory 18.6–74 MB per facet;
  - per-frame camera continuation;
  - a moving sharp object invalidates every cache of that object.
- **No designer smoke can discriminate learned utility with power.**
  - The cost rules are component-level, not lifecycle.
econ
final91.json
lists
logs
tables
tests91.json
viz
w_choice.json
RUN [Read WL96 cost-ceiling section and list WL91 result files] cd /c/Projects/ReST-GI && sed -n '128,160p' docs/worklogs/96_WORKLOG_*.md | cut -c1-700; ls results/gm1_c3urx1_91/results | head -30
RUN [Search WL91 economics JSON for the anchors quoted in WL96] ls results/gm1_c3urx1_91/results/econ; python - <<'EOF'
import json,glob,re
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,p+'/'+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,p+'/'+str(i))
    else: yield p,o
targets=[124.7,114.1,75.8,65.2,10.6,0.1247,0.0758,0.0106]
for f in glob.glob('results/gm1_c3urx1_91/results/**/*.json',recursive=True)+glob.glob('results/gm1_relation89/results/econ/*.json')+glob.glob('r
  -> <persisted-output>
Output too large (259.1KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bic0sbo6x.txt

Preview (first 2KB):
cold91_C3UR.json
cold91_C3URX1.json
cold91_PT.json
cold91_R0.json
ladder91.json
life91.json
s ...[cut]... nal91.json /conditions/W1_T0/paired/C3URX1_minus_C3UR/D/ci95/1 3.534900456794187e-05
gm1_c3urx1_91/results\final91.json /conditions/W1_T0/tables_meta/min 0.0
gm1_c3urx1_91/results\final91.json /conditions/W1_T1/deployed/R0/C_ss/E_H 0.010931950480527204
gm1_c3urx1_91/results\final91.json /conditions/W1_T1/deployed/R0/R_mirror/ref_noise 0.01017089997548321
gm1_c3urx1_91/results\final91.json /conditions/W1_T1/deployed/C3UR/R_mirror/ref_noise 0.01017089997548321
gm1_c3urx1_91/results\final91.json /conditions/W1_T1/deployed/C3URX1/R_mirror/ref_noise 0.01017089997548321
...
</persisted-output>
RUN [Search WL91 economics JSON for the anchors quoted in WL96] cd /c/Projects/ReST-GI && ls results/gm1_c3urx1_91/results/econ; python - <<'EOF'
import json,glob,re
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,p+'/'+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,p+'/'+str(i))
    else: yield p,o
targets=[124.7,114.1,75.8,65.2,10.6,0.1247,0.0758,0.0106]
for f in glob.glob('results/gm1_c3urx1_91/results/**/*.json',recursive=True)+glob.glob('results/gm1_relation89/results/
RUN [Inspect WL91 lifecycle economics JSON] python - <<'EOF'
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and depth<3: show(v,p+'/'+str(k),depth+1)
            elif not isinstance(v,(dict,list)): print(p+'/'+str(k), v if not isinstance(v,float) else round(v,5))
            else: print(p+'/'+str(k), type(v).__name__, (list(v.keys())[:8] if isinstance(v,dict) else len(v)))
    elif is
  -> /foreign_before 
/gpu NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB
/frame_spp 64
/arms/C3URX1/edit/wall_s 0.12466
/arms/C3URX1/edit/wall_raw_s 0.12466
/arms/C3URX1/edit/events_with_compilation 0
/arms/C3URX1/edit/compile_s_total 0.0
/arms/C3URX1/edit/cpu_s 0.12898
/arms/C3URX1/edit/gpu_k ...[cut]... 
/arms/PT/amortisation_sequence_raw_s 48.96619
/arms/PT/amortisation_sequence_compile_s 45.2125
/arms/PT/amortisation_events list 17 [['edit', 0.2517061401158571, 0.0], ['source', 0.21445122174918652, 0.0]]
/arms/PT/relation_digest_constant_across_sources True
/arms/PT/relation_rays_retraced_per_edit 0
/arms/PT/torch_peak_mib 69.98535
/arms/PT/process_gpu_mem 2536691, 3160 MiB
/arms/PT/amortisation_sequence_predicted_s 3.69122
/arms/PT/render_rays_nominal_per_frame_one_light/intersections 100663296
/arms/PT/render_rays_nominal_per_frame_one_light/shadow_rays 100663296
/foreign_after
RUN [Inspect WL91 lifecycle economics JSON] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and depth<3: show(v,p+'/'+str(k),depth+1)
            elif not isinstance(v,(dict,list)): print(p+'/'+str(k), v if not isinstance(v,float) else round(v,5))
            else: print(p+'/'+str(k), type(v).__name__, (list(v.keys())[:8] if isinstance(v,dict)
RUN [Compare local and server result hashes for WL95A and WL96A] (cd results/ecro_attr95a && sha256sum an.json main.json s0.json ops95a.npz deliv.npz chain.status | cut -c1-16,65-) ; (cd results/ecro_util96a && sha256sum s0.json s1.json tails.json tails.npz T96A.npz cost_s1.0.json final.json chain.status | cut -c1-16,65-); ssh LabServer63 'cd ~/rest_x95a/results && sha256sum an.json main.json s0.json ops95a.npz deliv.npz ../chain.status | cut -c1-16,65-; cd ~/rest_x96a/results && sha256sum s0.json s1.json tails.json tails.npz tables/T96A.npz cost_s1.0.json fi
  -> d8f4ebdc72d78472 *an.json
28d104991bcad6f7 *main.json
ea95017a511dcb0a *s0.json
d4d02e5a2f423ce9 *ops95a.npz
f2324a32effc80ca *deliv.npz
47e9ff9a3cce1643 *chain.status
aabaddad35eee08b *s0.json
92f5e763126d83b6 *s1.json
395f0166deedeb5b *tails.json
04da33a8b57eeb32 *tails.npz
8019cb39d03421dd *T96A. ...[cut]... ro_attr95a/protocol/ecro_attr95a.json). Freeze commit `c152120`; protocol sha256 prefix fb136ad2da79fdca (committed file).
28:| T_OPS | A1 = WL95 `nmf16_wl92` and A2 = `nmf16_l1irls` (from `S1_ops.npz`, sha256 8654ae79…). A3 = rank-32 non-negative L1 operator regenerated deterministically with the WL95 ladder code; it reproduces WL95 op L1 0.23306 and val field 0.14582 to 1e-6. All non-negative. The `nmf.npz` M64 equa
192:- **Commits.** Freeze `c152120` (protocol sha256 prefix fb136ad2da79fdca). Results are committed with this worklog.
209:  - `results/evaluation/95A/`, with a sha256 manifest:
RUN [Compare local and server result hashes for WL95A and WL96A] cd /c/Projects/ReST-GI && (cd results/ecro_attr95a && sha256sum an.json main.json s0.json ops95a.npz deliv.npz chain.status | cut -c1-16,65-) ; (cd results/ecro_util96a && sha256sum s0.json s1.json tails.json tails.npz T96A.npz cost_s1.0.json final.json chain.status | cut -c1-16,65-); ssh LabServer63 'cd ~/rest_x95a/results && sha256sum an.json main.json s0.json ops95a.npz deliv.npz ../chain.status | cut -c1-16,65-; cd ~/rest_x96a/results && sha256sum s0.json s1.json tails.json tails.npz tables/
RUN [Compare pre-freeze probe and frozen WL96A outputs numerically] cat > /tmp/cmp96.py <<'EOF'
import numpy as np, json, os, sys
H=os.path.expanduser('~')
a=np.load(H+'/rest_x96aprobe/results/tables/T96A.npz'); b=np.load(H+'/rest_x96a/results/tables/T96A.npz')
print('keys equal', sorted(a.files)==sorted(b.files), len(a.files))
worst=[]
for k in a.files:
    x=a[k].astype(np.float64); y=b[k].astype(np.float64)
    if x.shape!=y.shape: print('shape', k, x.shape, y.shape); continue
    d=np.abs(x-y); worst.append((float(d.sum()/max(np.abs(y).sum(),1e-30)), float(d
  -> cmp96.py: Python script, ASCII text executable
Traceback (most recent call last):
  File "/home/nd/rest_x97/MAP/cmp96.py", line 1, in <module>
    import numpy as np, json, os, sys
ModuleNotFoundError: No module named 'numpy'
RUN [Run probe-vs-frozen comparison with the project Python (CPU only)] cp /tmp/cmp96.py experiments/audit97/MAP/cmp96_probe_vs_frozen.py && ssh LabServer63 'cd ~/rest_x97/MAP && . ~/rest_x95a/env.sh && CUDA_VISIBLE_DEVICES= $PY cmp96.py 2>&1 | head -80'
  -> keys equal True 12
relL1 3.396e-09 relmax 7.700e-08 XD bitwise=False
relL1 3.322e-09 relmax 7.512e-08 XOD bitwise=False
relL1 1.642e-09 relmax 7.666e-08 XP bitwise=False
relL1 1.637e-09 relmax 7.503e-08 XOP bitwise=False
relL1 1.366e-10 relmax 3.859e-08 XSD bitwise=False
relL1 6.024e-11 relmax 3.859 ...[cut]... iming_s/pool12 probe=1.0494 final=1.07212
  rel 1.98e-02 /timing_s/pool1 probe=1.05923 final=1.08067
  rel 1.78e-02 /timing_s/pool4 probe=1.04916 final=1.0682
  rel 1.65e-02 /timing_s/pool11 probe=1.05373 final=1.07142
  rel 1.48e-02 /timing_s/pool15 probe=1.06936 final=1.08542
  rel 1.41e-02 /total_s probe=24.1226 final=24.4679
  rel 1.38e-02 /timing_s/pool2 probe=1.05321 final=1.06793
  cost /arms/R0/edit/parts/tail_orders probe=0.0034209 final=0.0033112
  cost /arms/R0/source/parts/tail_orders probe=0.0034109 final=0.0033108
  cost /components/tail_3_orders_s probe=0.0044324 final=0.0032933
RUN [Run probe-vs-frozen comparison with the project Python (CPU only)] cd /c/Projects/ReST-GI && cp /tmp/cmp96.py experiments/audit97/MAP/cmp96_probe_vs_frozen.py && ssh LabServer63 'cd ~/rest_x97/MAP && . ~/rest_x95a/env.sh && CUDA_VISIBLE_DEVICES= $PY cmp96.py 2>&1 | head -80'
RUN [Inspect the post-freeze edit of the WL93 analyzer and the run timeline] git diff c30c797 56be753 -- experiments/ecro_stage93/analyze93.py | cut -c1-260 | head -50; ssh LabServer63 'cd ~/rest_x93; ls -la --time-style=+%H:%M:%S results | cut -c24-120; cat chain.status | cut -c1-160'
  -> diff --git a/experiments/ecro_stage93/analyze93.py b/experiments/ecro_stage93/analyze93.py
index d2933f1..006b71e 100644
--- a/experiments/ecro_stage93/analyze93.py
+++ b/experiments/ecro_stage93/analyze93.py
@@ -116,7 +116,7 @@ def main():
         for s in SEEDS:
             for cn in ('E6', 'E2' ...[cut]... 53bbee1e
2026-11-13T06:42:56Z START S0_recon92b
2026-11-13T06:42:57Z END S0_recon92b rc=0
2026-11-13T06:42:57Z RECON92B {"H_interact_official": "SUPPORTED", "H_interact_written_rule": "INCONCLUSIVE", "outcome_official": "C", "outcome_written_rule": "
2026-11-13T06:42:57Z START S1_qgen_val
2026-11-13T06:43:15Z END S1_qgen_val rc=0
2026-11-13T06:43:15Z START S1_ident
2026-11-13T06:44:01Z END S1_ident rc=0
2026-11-13T06:44:01Z IDENT FAIL
2026-11-13T06:44:01Z STOP STOP_TARGET_NOT_IDENTIFIABLE
2026-11-13T06:44:01Z START A_final
2026-11-13T06:44:08Z END A_final rc=1
2026-11-13T06:44:08Z CHAIN_FAILED
RUN [Inspect the post-freeze edit of the WL93 analyzer and the run timeline] cd /c/Projects/ReST-GI && git diff c30c797 56be753 -- experiments/ecro_stage93/analyze93.py | cut -c1-260 | head -50; ssh LabServer63 'cd ~/rest_x93; ls -la --time-style=+%H:%M:%S results | cut -c24-120; cat chain.status | cut -c1-160'
RUN [Read WL93 reconciliation of the WL92B outcome] cat results/ecro_stage93/recon92b.json | head -c 2500; echo; grep -n -iE "recon|written rule|written_rule|analyze93|crash" docs/worklogs/93_WORKLOG_*.md | cut -c1-420 | head -14
  -> {
 "written_rule": "SUPPORTED if no single counterfactual is DOMINANT in both E2 cases but at least two have f >= 0.25 for both OWN seeds in a case",
 "implemented_rule": "no_dom = not (H_local or H_global SUPPORTED)  (only X_M12 and X_L3 checked)",
 "dominant_counterfactuals_both_E2_cases": {
  "X_ ...[cut]... TOP. See Amendment 1.
185:- **Change.** In `analyze93.py`, the STAGE value extraction now returns None when diagnostics are absent: `(((m or {}).get('diag') or {}).get('T1') or {}).get(...)`. That was the intended behaviour; the original expression could return `{}`.
187:- **Rerun.** `analyze93.py` (log `A_final_amend1.log`) and `viz93.py` were rerun. Label: **STOPPED-INCONCLUSIVE**, primary cause STOP_TARGET_NOT_IDENTIFIABLE.
235:- **Server.** LabServer63 (RTX 3080 Ti 12 GB); root `~/rest_x93` (`chain.status`, `results/ident.json`, `recon92b.json`, `posthoc_ident.json`, `final93.json`, logs).
RUN [Read WL93 reconciliation of the WL92B outcome] cd /c/Projects/ReST-GI && cat results/ecro_stage93/recon92b.json | head -c 2500; echo; grep -n -iE "recon|written rule|written_rule|analyze93|crash" docs/worklogs/93_WORKLOG_*.md | cut -c1-420 | head -14
RUN [Read roadmap entries WL92 to WL93D] sed -n '140,147p' docs/RESEARCH_ROADMAP.md | cut -c1-2600
  -> - [WL92](worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md) (2026-11-13): attribution of ECRO v0's sharp failure on the frozen WL86 family (nothing retrained; operator recovered from code: tied local factors a c^T, single-flight term = exactly 2 events, global term = 3-5 ...[cut]...  26-32 %); swapping in the other geometry's operator barely changes transport; the receiver-use error lies mostly inside the training incident span. Physical source use with learned receiver/global (C2_src) gives the best rendered sharp readers on every inspected condition (E2 T1 region C 0.018-0.024 vs 0.040-0.077) while worsening pooled validation single flight (+0.5-0.8 points); physical receiver use does the opposite (co-adaptation). Held-out E6 (frozen WL93 rule) not admitted (3/3 candidates fail the sharp-family clause): conclusions limited to inspected E2 / E5. No adoption or successor.
RUN [Read roadmap entries WL92 to WL93D] cd /c/Projects/ReST-GI && sed -n '140,147p' docs/RESEARCH_ROADMAP.md | cut -c1-2600
RUN [Read WL95A stage 0 contracts and result sections] sed -n '12,112p' docs/worklogs/95A_WORKLOG_*.md | cut -c1-900
  -> - LabServer63 `~/rest_x95a`, one detached chain, 21:02–21:08 UTC (6 min).
  - Local results: `results/ecro_attr95a/`.
  - Figures: `results/evaluation/95A/`.
- **Workload.** CK2 "none" only: one point light at [−1.45, −0.6, 1.7], rgb 30, grazing on the S1 panel. The E2 secondary workload was not use ...[cut]... oled A1 / A3 / A2 | 0.0209 / 0.0204 / 0.0395 | −0.001 / +0.010 / −0.039 |
| G_s8601 / G_s8602 (reference e2 + own learned G) | 0.0251 / 0.0150 | −0.016 / −0.006 |

### B4. Compute (RTX 3080 Ti 12 GB, LabServer63)

- The main stage took 74 s in total:
  - learned factors 2.1 s;
  - 64 source draws 0.7 s;
  - q256: 2.3 s for 2 seeds × 11.0 M MIS lanes;
  - deployed relation 1.6 s;
  - 16 pilot draws 17 s;
  - extension to 64 draws 49 s.
- Peak torch memory was 169 MiB.
- The deployed relation sends 9,995 rays to S1 patches.
- The final set has 38 renders at 8.6–11.4 s each (374 s total, pool 3).
RUN [Read WL95A stage 0 contracts and result sections] cd /c/Projects/ReST-GI && sed -n '12,112p' docs/worklogs/95A_WORKLOG_*.md | cut -c1-900
RUN [Locate the passivity (directional albedo) computation in WL95A code] sed -n '139,166p' experiments/ecro_attr95a/a95.py | cut -c1-240; grep -n "albedo" experiments/ecro_attr95a/a95.py | cut -c1-240
  -> def field_metrics(X, Y, W=WT):
    """S1 field [42, NB, 3]: flux rel L1 / energy / peak of X against reference Y."""
    Wd = W[None, :, None]
    pk = Y.argmax(1)[:, None, :]
    return {'rel_l1': float((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum()), 'energy': float((X * Wd).sum() / (Y * Wd).sum()),
 ...[cut]... ntry': float(A.min()), 'passive': bool(alb.max() <= 1 + 1e-6), 'nonneg': bool(A.min() >= 0)}
189:    # ER cosine convention on a Lambert material (id 3, albedo 0.7)
196:    alb = float(np.asarray(ctx.mats.albedo)[3])
317:        om[f'A0_s{sd}'] = {'op_flux_l1_mean': float(np.mean([p['op_flux_l1'] for p in per])), 'albedo_max': float(np.max([p['albedo_max'] for p in per])),
559:    ax[3].bar(ok_, [om[k]['albedo_max'] for k in ok_]); ax[3].axhline(1, color='r', ls='--', label='passivity limit')
560:    ax[3].set_title('operator max directional albedo (W_T)', fontsize=8); ax[3].legend(fontsize=7)
RUN [Locate the passivity (directional albedo) computation in WL95A code] cd /c/Projects/ReST-GI && sed -n '139,166p' experiments/ecro_attr95a/a95.py | cut -c1-240; grep -n "albedo" experiments/ecro_attr95a/a95.py | cut -c1-240
RUN [Read WL95A stored operator metrics and labels] python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(list(m.keys()))
for k,v in m.get('operators',m.get('ops',{})).items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
s0=json.load(open('results/ecro_attr95a/s0.json'))
print(s0['status'], list(s0['tests'].keys()))
an=json.load(open('results/ecro_attr95a/an.json'))
print(list(an.keys()))
for k in ('labels','verdict','decision','rules'):
    if k in an: print(k, json.dumps(an[k])[
  -> ['timing', 'counts', 'sender', 'operator', 'D_pilot', 'D_used', 'T_DEPLOYED_ID', 'gpu_mem_peak_mib_torch', 'tables']
PASS ['T_BINS', 'T_MASKS', 'T_EVENT', 'T_COS', 'T_REPRO94D', 'T_OPS']
['delivery', 'flight', 'rank16', 'images', 'floor_E_H', 'components', 'verdicts', 'D_used']
    qs['er_pts'] = ...[cut]... ], M1) for i in (1, 2, 3, 4)}
    for sd in SEEDS:
        per = [op_metrics(lop[sd][j], M1) for j in range(len(S1))]
        om[f'A0_s{sd}'] = {'op_flux_l1_mean': float(np.mean([p['op_flux_l1'] for p in per])), 'albedo_max': float(np.max([p['albedo_max'] for p in per])),
                           'passive_all': bool(all(p['passive'] for p in per)), 'nonneg': bool(min(p['min_entry'] for p in per) >= 0),
                           'mass_mean': float(np.mean([p['mass'] for p in per]))}
    out['sender'] = sm; out['operator'] = om
    # propagation: deployed + pooled draws
    rw0 = rw_of(u0, S)
RUN [Read WL95A stored operator metrics and labels] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(list(m.keys()))
for k,v in m.get('operators',m.get('ops',{})).items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
s0=json.load(open('results/ecro_attr95a/s0.json'))
print(s0['status'], list(s0['tests'].keys()))
an=json.load(open('results/ecro_attr95a/an.json'))
print(list(an.keys()))
for k in ('labels','verdict','decision','rules'):
    if k in an: p
RUN [Read stored verdict structures and Architecture.md dispositions] python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
for k,v in m['operator'].items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
an=json.load(open('results/ecro_attr95a/an.json'))
print(json.dumps(an['verdicts'])[:2500])
print('floor', an['floor_E_H'], 'D_used', an['D_used'])
f=json.load(open('results/ecro_util96a/final.json')); print(json.dumps(f)[:900])
s1=json.load(open('results/ecro_util96a/s1.json')); print(json.dumps(s1['decis
  -> A1 {'op_flux_l1': 0.6025, 'mass': 1.038, 'albedo_max': 0.9601, 'albedo_mean': 0.5685, 'min_entry': 0.0, 'passive': True, 'nonneg': True}
A2 {'op_flux_l1': 0.3798, 'mass': 0.6891, 'albedo_max': 0.8057, 'albedo_mean': 0.3286, 'min_entry': 0.0, 'passive': True, 'nonneg': True}
A3 {'op_flux_l1': 0.233 ...[cut]... global spatial/angular transport representation and edit-update mechanism can preserve sharp glossy/specular and multi-bounce response under a rigid edit using frozen ER, with qualified T0/T1 response and measured end-to-end economics? One mechanism must be explicitly 
114:- **Contract version boundary:** current reference semantics, explicit ballistic readout, near-contact spatial requirements and geometry-state invalidation are later requirements, not retroactive contents or validation of v0.1. A materially different detailed learned implementation needs an explicitly versioned successor des
RUN [Read stored verdict structures and Architecture.md dispositions] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
for k,v in m['operator'].items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
an=json.load(open('results/ecro_attr95a/an.json'))
print(json.dumps(an['verdicts'])[:2500])
print('floor', an['floor_E_H'], 'D_used', an['D_used'])
f=json.load(open('results/ecro_util96a/final.json')); print(json.dumps(f)[:900])
s1=json.load(open('results/ecro_util96a/s1.json')); 
RUN [Trace fixed seeds of the state realization, source points and M64 quadrature] grep -nE "86001|table_seed|u_src_np|u_state_np|def build_state|def realization_us|default_rng" experiments/ecro86/pt86.py experiments/ecro86/ecro86.py experiments/ecro86/scene86.py experiments/gm1_explicit87/xport87.py experiments/gm1_explicit87/quad87.py experiments/ecro86/gen86.py experiments/ecro86/train86.py | cut -c1-230; python - <<'EOF'
import json
P=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'))
E=P['explicit']; print({k:(v if not isinstance(v,(dict,list)) els
  -> experiments/ecro86/pt86.py:114:        rs = np.random.default_rng(P['representation'].get('table_seed', 86001))
experiments/ecro86/pt86.py:116:        self.u_state_np = us
experiments/ecro86/pt86.py:117:        self.u_src_np = rs.random((2, 1, PSRC), np.float32)
experiments/ecro86/pt86.py:126:       ...[cut]... riments/gm1_explicit87/xport87.py:45:    ctx.u_state_np = us
experiments/gm1_explicit87/xport87.py:139:    u0 = np.random.default_rng(seed).random()
experiments/ecro86/gen86.py:30:        rng = np.random.default_rng([seed, idx])
experiments/ecro86/train86.py:81:    rng = np.random.default_rng(seed)
{'M_ray': 65536, 'seed_ray': 87001, 'M_dense': 16384, 'seed_dense': 87002, 'dense_cells': '8 x 8 strata per incident bin (4,096 cells, cell centres)', 'P': 16, 'Mq': 1024, 'seed_q': 87003, 'seed_c3': 87004, 'orders': 4, 'not_imposed': 'no renormalisation, clamping or energy correction of any table'}
RUN [Trace fixed seeds of the state realization, source points and M64 quadrature] cd /c/Projects/ReST-GI && grep -nE "86001|table_seed|u_src_np|u_state_np|def build_state|def realization_us|default_rng" experiments/ecro86/pt86.py experiments/ecro86/ecro86.py experiments/ecro86/scene86.py experiments/gm1_explicit87/xport87.py experiments/gm1_explicit87/quad87.py experiments/ecro86/gen86.py experiments/ecro86/train86.py | cut -c1-230; python - <<'EOF'
import json
P=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'))
E=P['explicit']; print({k:(v if not isi
RUN [Inspect the amendment change to the event-split field reference module] git diff --stat 46593f0 1bd7df1 -- experiments/ecro86a/pt86a.py; git diff 46593f0 1bd7df1 -- experiments/ecro86a/pt86a.py | grep -E '^[+-]' | cut -c1-200 | head -60
  -> experiments/ecro86a/pt86a.py | 19 +++++++++++++++++--
 1 file changed, 17 insertions(+), 2 deletions(-)
--- a/experiments/ecro86a/pt86a.py
+++ b/experiments/ecro86a/pt86a.py
-def field_src(ctx, src, spp, seed, kmax=PT.D - 1, b=128):
-    """As pt86.Ctx.field, with the incident field of `src`. Return ...[cut]...     for q in range(4):
+                        sel = (v2p if q % 2 == 0 else ~v2p) if ((q < 2) == (k == 2)) else mi.Bool(False)
+                        parts[q] += dr.select(sel, c, 0.0)
+        if split:
+            for q, nm in enumerate(('e2_p', 'e2_o', 'e3_p', 'e3_o')):
+                out[f'{nm}0'] = parts[q].x; out[f'{nm}1'] = parts[q].y; out[f'{nm}2'] = parts[q].z
+    if split:
+        return {nm: (np.stack([acc[f'{nm}{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]).astype(np.float32)
+                for nm in ('e2_p', 'e2_o', 'e3_p', 'e3_o')}
RUN [Inspect the amendment change to the event-split field reference module] cd /c/Projects/ReST-GI && git diff --stat 46593f0 1bd7df1 -- experiments/ecro86a/pt86a.py; git diff 46593f0 1bd7df1 -- experiments/ecro86a/pt86a.py | grep -E '^[+-]' | cut -c1-200 | head -60
RUN [Check whether the mis-specified weight vector enters the WL94D training loss] sed -n '122,135p;280,300p' experiments/ecro_s94d/s94.py | cut -c1-260; grep -n -iE "an94\.W|8 ?[x×] ?8|W94|weight" docs/worklogs/94D_WORKLOG_*.md | cut -c1-300 | head -12; grep -n -iE "an94\.W|8 x 8|8 × 8" docs/worklogs/95_WORKLOG_*.md | cut -c1-500 | head -8
  -> tab = np.load(os.path.join(R, 'results', 'tables', f'CD_D_s{sd}.npz'))
        sem[sd] = {'C2_srcS1_vs_stored': float(np.abs(Smine - tab['none|S_C2_srcS1']).max() / np.abs(tab['none|S_C2_srcS1']).max()),
                   'Bl_path_vs_model_apply': float(np.abs(Sl - Sdep).max() / np.abs(Sdep).max()) ...[cut]... ransfer to CK2. Optimisation versus representation is not separated at this budget.
25:  - Confirmed the `an94.W` layout defect: `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, whereas the protocol is 4 rings × 16 sectors (`experiments/ecro86/protocol/ecro86.json:47`; `experiments/ecro86/pt86.py:153-156`).
214:1. **`an94.W` ring layout** [FACT]
636:  - Weight: true W_T[d] = ((d//16) + ½)/4·ΔΩ. `an94.W` may appear only as a bridge to the old numbers.
799:- the `an94.W` (8 × 8) defect;
873:  - Each is reported with W94 (= `an94.W`, as coded) and with WT = zb·ΔΩ (true layout).
RUN [Check whether the mis-specified weight vector enters the WL94D training loss] cd /c/Projects/ReST-GI && sed -n '122,135p;280,300p' experiments/ecro_s94d/s94.py | cut -c1-260; grep -n -iE "an94\.W|8 ?[x×] ?8|W94|weight" docs/worklogs/94D_WORKLOG_*.md | cut -c1-300 | head -12; grep -n -iE "an94\.W|8 x 8|8 × 8" docs/worklogs/95_WORKLOG_*.md | cut -c1-500 | head -8
RUN [Locate teacher loss weighting in WL94D code] grep -nE "loss|teach|lam\b|\* W\b|zb\b" experiments/ecro_s94d/s94.py | cut -c1-220 | sed -n '1,45p'
  -> 8:  T = + lam_T x teacher(Bl vs Bm on S1 panel patches);  R = + lam_R x teacher(Bl vs Bm on all non-S1 patches);  P = plain continuation.
10:  python s94.py audit <R> <R86> <R92A>          Stage A contract / oracle audit, teacher weights (results/audit94d/)
57:def teacher_loss(Bl, Bm):
127:    # (li ...[cut]... 3)), (Dm * W).sum((2, 3))
317:        X = (A - M1.unsqueeze(0)).abs() * zb.view(1, -1, 1, 1) * zb.view(1, 1, -1, 1)
318:        T_ = M1.unsqueeze(0).expand_as(A) * zb.view(1, -1, 1, 1) * zb.view(1, 1, -1, 1)
366:            m['fit'] = {k: j[k] for k in ('val_orig_rel_l1_D', 'val_orig_rel_l1_final', 'nonfinite_steps', 'init_sha256', 'final_sha256', 'train_s', 'lam', 'steps')}
367:            m['fit']['teacher_S1_first50'] = j['log'][0]['teacher_S1']; m['fit']['teacher_S1_last50'] = j['log'][-1]['teacher_S1']
466:            for k, q in enumerate(('L_sum', 'L_single', 'L_global', 'teacher_S1')):
RUN [Locate teacher loss weighting in WL94D code] cd /c/Projects/ReST-GI && grep -nE "loss|teach|lam\b|\* W\b|zb\b" experiments/ecro_s94d/s94.py | cut -c1-220 | sed -n '1,45p'
RUN [Read WL94D teacher loss and the frozen mapping inputs] sed -n '55,64p' experiments/ecro_s94d/s94.py | cut -c1-240; grep -n "mapping\|NO_RELIABLE\|field_rel\|rel_l1_flux\|delivery" experiments/ecro_s94d/protocol/ecro_s94d.json | cut -c1-600 | head -8
  -> def teacher_loss(Bl, Bm):
    """per-entry normalised squared error, den = per-(lighting, patch, channel) bin mean of the target + 0.05 x its mean (OWN form)."""
    den = Bm.mean(2, keepdim=True) + 0.05 * Bm.mean(dim=(1, 2, 3), keepdim=True) + 1e-12
    return (((Bl - Bm) / den) ** 2).mean()


def  ...[cut]... er (all receivers / box); S1 local operator matrix vs M64 per val geometry and CK2, flux-transfer weighted, split into incident bins excited in training vs never excited",
49:   "NO_RELIABLE_IMPROVEMENT": "SENDER_IMP on neither seed",
53: "answers_mapping": "supervisable = Stage A; fit = T training-field sender error and operator error on trained bins vs D / P; generalization = val / CK2 field errors and never-excited-bin operator error vs D / P; both seeds = SENDER_IMP / RENDER_IMP per seed; global / F2 / MR = panel-face G and KEEP; gap to exact sender = E_H(T) - E_H(D with exact S1 sender)",
RUN [Read WL94D teacher loss and the frozen mapping inputs] cd /c/Projects/ReST-GI && sed -n '55,64p' experiments/ecro_s94d/s94.py | cut -c1-240; grep -n "mapping\|NO_RELIABLE\|field_rel\|rel_l1_flux\|delivery" experiments/ecro_s94d/protocol/ecro_s94d.json | cut -c1-600 | head -8
RUN [Check whether WL94D decision metrics were re-evaluated under corrected weights] grep -n -iE "SENDER_IMP|re-evaluat|recomput.*W_?T|label.*unchanged|WL94D.*(label|outcome|verdict)" docs/worklogs/95_WORKLOG_*.md | cut -c1-420 | head -14; sed -n 214,224p docs/worklogs/95_WORKLOG_*.md | cut -c1-900
  -> 1. **`an94.W` ring layout** [FACT]
   - `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, and `:32` computes W = ((d//8) + ½)/8·ΔΩ. The true layout is 4 rings × 16 sectors.
   - Effect: inside a ring, sectors 0–7 and 8–15 get cosine weights (iz + ¼)/4 and (iz + ¾)/4. In the grazing ring that is a 1:3 ratio. ΣW = π is preserved.
   - Where W is used:
     - the WL94D field and delivery metrics (`s94.py:287, 297-312`);
     - the CK2 flight floor (`s94.py:157`);
     - G_PHYS (`s94.py:129`);
     - WL94 exitance `an94.ex` (`an94.py:37, 104`);
     - WL94B (`experiments/ecro_exp94b/b94.py:47`).
   - It is neutral for Lambert outgoing radiance [INF], so the box-A and panel-face CK2 floor values are unaffected.
   - It biases non-Lambert numbers: the all-receiver 1.378, and S1 energy_ratio 1.71–1.75.
RUN [Check whether WL94D decision metrics were re-evaluated under corrected weights] cd /c/Projects/ReST-GI && grep -n -iE "SENDER_IMP|re-evaluat|recomput.*W_?T|label.*unchanged|WL94D.*(label|outcome|verdict)" docs/worklogs/95_WORKLOG_*.md | cut -c1-420 | head -14; sed -n 214,224p docs/worklogs/95_WORKLOG_*.md | cut -c1-900
RUN [Cross-check recorded artifact hashes in worklogs against computed values] python - <<'EOF'
import json
r=json.load(open('results/ecro_cf92b/repro.json'))
print('92B repro checkpoints:', {k:v[:16] for k,v in r.get('checkpoints',{}).items()})
EOF
grep -n -oE "(CKF_none[^;|]{0,60}|FZ_own[^;|]{0,60}|masks94[^;|]{0,40}|T95A[^;|]{0,50}|REF_ER_CK_none[^;|]{0,80}|own_s860[12][^;|]{0,60})" docs/worklogs/96A_WORKLOG_*.md docs/worklogs/95A_WORKLOG_*.md docs/worklogs/94_WORKLOG_*.md | grep -E "[0-9a-f]{8}" | cut -c1-230 | head -20; ls results/ecro_mat94 results/evaluation/96A 2>/
  -> 92B repro checkpoints: {'ecro_s8601': '9d332cd61a7118bd', 'ecro_s8602': '4d993121e41dd0bb', 'sum_s8601': '3ec1d5910a3e3a3e', 'sum_s8602': 'd911c6b22cfd9054', 'own_s8601': '739e7eb50b25d329', 'own_s8602': 'cee9a09a289fca8c'}
docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md:222:CKF_n ...[cut]... IAL_GENERALIZATION_VS_SHARP_ANGULAR.md:299:own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…` (file sha256, = WL92
results/ecro_mat94:
chain.status
final.json
gpu.txt
models
pop
posthoc94.json
pre.json
reused_refs.txt
select94.json
stageB.json
stageB_rho.json
stageC_s8601.json

results/evaluation/96A:
figW1_tail_arms.png
figW2_boxA_tail_field.png
figW3_lifecycle.png
figW3_lifecycle_posthoc.png
manifest.json
f2324a32effc80ca *results/ecro_attr95a/deliv.npz
d4d02e5a2f423ce9 *results/ecro_attr95a/ops95a.npz
8019cb39d03421dd *results/ecro_util96a/T96A.npz
04da33a8b57eeb32 *results/ecro_util96a/tails.npz
RUN [Cross-check recorded artifact hashes in worklogs against computed values] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
r=json.load(open('results/ecro_cf92b/repro.json'))
print('92B repro checkpoints:', {k:v[:16] for k,v in r.get('checkpoints',{}).items()})
EOF
grep -n -oE "(CKF_none[^;|]{0,60}|FZ_own[^;|]{0,60}|masks94[^;|]{0,40}|T95A[^;|]{0,50}|REF_ER_CK_none[^;|]{0,80}|own_s860[12][^;|]{0,60})" docs/worklogs/96A_WORKLOG_*.md docs/worklogs/95A_WORKLOG_*.md docs/worklogs/94_WORKLOG_*.md | grep -E "[0-9a-f]{8}" | cut -c1-230 | head -20; ls results/ecro_mat94 
RUN [Check local availability of P0 artifacts, CK2 scene hash and dataset sizes] find results/ecro_own92a results/ecro86 results/ecro_mat94 -name "*.pt" -o -name "CKF_none*" -o -name "masks94*" -o -name "FZ_own*" | head; git -c core.autocrlf=false show HEAD:experiments/ecro_local93d/addendum_checker/scene_ck2.json | sha256sum | cut -c1-16; git -c core.autocrlf=false show d8d092f:experiments/ecro_local93d/addendum_checker/scene_ck2.json | sha256sum | cut -c1-16; git log --format='%h %cI' -- experiments/ecro_local93d/addendum_checker/scene_ck2.json; ssh LabServer63 'cd ~; du -
  -> results/ecro86/results/models/ecro_s8601.pt
results/ecro86/results/models/ecro_s8602.pt
977ba122ab3232e7
977ba122ab3232e7
d8d092f 2026-11-14T17:30:55+09:00
1.1G	rest_x94/results 12M	rest_x92a/results/models 2.6G	rest_ecro86/data train
trainhq
val
1024
ls: cannot access 'rest_ecro86/data/sp': No such file or directory
0
64
RUN [Check local availability of P0 artifacts, CK2 scene hash and dataset sizes] cd /c/Projects/ReST-GI && find results/ecro_own92a results/ecro86 results/ecro_mat94 -name "*.pt" -o -name "CKF_none*" -o -name "masks94*" -o -name "FZ_own*" | head; git -c core.autocrlf=false show HEAD:experiments/ecro_local93d/addendum_checker/scene_ck2.json | sha256sum | cut -c1-16; git -c core.autocrlf=false show d8d092f:experiments/ecro_local93d/addendum_checker/scene_ck2.json | sha256sum | cut -c1-16; git log --format='%h %cI' -- experiments/ecro_local93d/addendum_checker/scene_ck2.json; s
RUN [Inspect WL89 amendment audit evidence and uncommitted debug scripts] git log --all --oneline -- '*dbg89*' '*dbgp.py' | head -3; ls results/gm1_relation89/results | tr '\n' ' '; ls results/gm1_relation89/results/amend1 results/gm1_relation89/results/orig_stop 2>/dev/null | tr '\n' ' '; python - <<'EOF'
import json,os
p='results/gm1_relation89/results/amend1/audit89.json'
if os.path.exists(p):
    d=json.load(open(p)); print(list(d.keys()))
    for k,v in d.items():
        s=json.dumps(v); print(' ',k, s[:330])
EOF
ssh LabServer63 'ls -la --time-style=+%m-%dT%H:%M
  -> amend1 amend1_audit.log amend1_audit_run1_testbug.log econ final89.json lists logs masks orig_stop renders stage1_89.json stage2_89.json stage3_89.json tables tests88.json tests89.json viz w_choice.json w_screen.json wq_V3.json results/gm1_relation89/results/amend1: audit89.json audit89_run1_testbug ...[cut]... op_equal": true, "move_and_return_strict_equal": true}
  perturbation {"dbin_random": {"perturbed_invalid_entries": 157538, "operator_tensors_unchanged": true, "cpu_outputs_bitwise_unchanged_and_finite": true, "later_selective_update_uncontaminated": true}, "dbin_255_cell_max": {"perturbed_invalid_entries": 157538, "operator_tensors_unchanged": true, "cpu_outputs_bitwise_unchanged_and_finite": tru
  AUDIT_PASS true
 11-12T04:47:23 /home/nd/rest_rel89/dbg89.py
 11-12T04:47:57 /home/nd/rest_rel89/dbg89b.py
 11-12T05:58:23 /home/nd/rest_rel89/dbgp.py
 11-12T05:59:50 /home/nd/rest_rel89/dbgs.py
RUN [Inspect WL89 amendment audit evidence and uncommitted debug scripts] cd /c/Projects/ReST-GI && git log --all --oneline -- '*dbg89*' '*dbgp.py' | head -3; ls results/gm1_relation89/results | tr '\n' ' '; ls results/gm1_relation89/results/amend1 results/gm1_relation89/results/orig_stop 2>/dev/null | tr '\n' ' '; python - <<'EOF'
import json,os
p='results/gm1_relation89/results/amend1/audit89.json'
if os.path.exists(p):
    d=json.load(open(p)); print(list(d.keys()))
    for k,v in d.items():
        s=json.dumps(v); print(' ',k, s[:330])
EOF
ssh LabServer63 'ls -la
RUN [Verify cross-batch quoted numbers against stored JSON] python - <<'EOF'
import json,glob,os
an=json.load(open('results/ecro_attr95a/an.json'))
im=an['images']
for k in im:
    if k.startswith('G_') or 'floor' in k: print('95A image',k, round(im[k]['E_H'],4), round(im[k].get('noise_bound',0),4))
s0=json.load(open('results/ecro_attr95a/s0.json'))
print('T_REPRO94D', json.dumps(s0['tests']['T_REPRO94D'])[:600])
print('T_BINS', json.dumps(s0['tests']['T_BINS'])[:400])
for f in glob.glob('results/ecro_attr92/results/*.json'):
    print(os.path.basename(f
  -> 95A image G_s8601 0.0251 0.0084
95A image G_s8602 0.015 0.0083
95A image floor 0.0118 0.0084
T_REPRO94D {"pass": true, "max_abs_dev": 1.4813043613948196e-07, "floor_W94": {"receivers": {"ratio": 1.3782469041792993, "corr": 0.7125124849079415, "rel_l1": 0.9397396152226791}, "boxA": {"ratio": 1.211 ...[cut]... 003313}}, "wl94d_floor": {"receivers": {"ratio": 1.3782468983831326, "corr": 0.712512482334751, "corr_halves": 0.9297572792340049, "rel_l1": 0.9397396120024692, "half_noise_rel_l1": 0.12721187196988423}, "boxA": {"ratio": 1.211470856649812
T_BINS {"pass": 1.0, "bin_formula_agreement": 1.0, "sum_WT": 3.141592653589793, "sum_W94": 3.141592653589793, "W94_vs_WT_ring0_sector_ratio": [0.5, 1.5], "an94_W_equals_W94": true}
core92.json e4_choice.json final92.json gate.json local.json prov.json tests92.json 
B_floor [0.03, 0.028, 0.04]
S1_panel [0.52, 0.458, 0.461]
S2_sphere [0.204, 0.335, 0.234]
RUN [Verify cross-batch quoted numbers against stored JSON] cd /c/Projects/ReST-GI && python - <<'EOF'
import json,glob,os
an=json.load(open('results/ecro_attr95a/an.json'))
im=an['images']
for k in im:
    if k.startswith('G_') or 'floor' in k: print('95A image',k, round(im[k]['E_H'],4), round(im[k].get('noise_bound',0),4))
s0=json.load(open('results/ecro_attr95a/s0.json'))
print('T_REPRO94D', json.dumps(s0['tests']['T_REPRO94D'])[:600])
print('T_BINS', json.dumps(s0['tests']['T_BINS'])[:400])
for f in glob.glob('results/ecro_attr92/results/*.json'):
  
RUN [Read WL94D chain script upstream links] sed -n '3,8p;27,38p' experiments/ecro_s94d/run94d.sh | cut -c1-230
  -> set -u
R=$1; W=$2; WA=$3; WB=$4; WC=$5; R86=$6; R92A=$7; MODE=${8:-}
DEV=; [ "$MODE" = --dev ] && DEV=--dev
. "$R/env.sh"
CODE=$R/code/experiments/ecro_s94d; MT94=$R/code/experiments/ecro_mat94; B94=$R/code/experiments/ecro_exp94b/b94.py
O=$R/results; ST=$R/chain.status
( while true; do echo "$(date ...[cut]... for f in "$WC"/results/renders/*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz "$WA"/results/tables/CD_C_s860?.npz "$WC"/results/tables/CD_D_s860?.npz; do ln -sf "$f" "$O/tables/"; done
for f in "$WC"/results/models/*; do ln -sf "$f" "$O/models/"; done
for f in "$WC"/results/pop/*; do ln -sf "$f" "$O/pop/"; done
for f in "$WC"/results/manifest/W_s860?.npz; do ln -sf "$f" "$O/manifest/"; done
ln -sf "$W/results/an/stageB.json" "$O/an/wl94_stageB.json"
for s in 8601 8602; do ln -sf "$WA/results/an/r_s$s.json" "$O/an/wl94a_r_s$s.json"; done
cd "$CODE" || exit 1
RUN [Read WL94D chain script upstream links] cd /c/Projects/ReST-GI && sed -n '3,8p;27,38p' experiments/ecro_s94d/run94d.sh | cut -c1-230
RUN [Diff WL95A dev versus frozen protocol and measurement code] ssh LabServer63 'cd ~; diff <(python3 -m json.tool rest_x95adev/code/experiments/ecro_attr95a/protocol/ecro_attr95a.json) <(python3 -m json.tool rest_x95a/code/experiments/ecro_attr95a/protocol/ecro_attr95a.json) | cut -c1-900; echo "--- a95.py:"; diff rest_x95adev/code/experiments/ecro_attr95a/a95.py rest_x95a/code/experiments/ecro_attr95a/a95.py | head -30; echo "--- run95a.sh:"; diff rest_x95adev/code/experiments/ecro_attr95a/run95a.sh rest_x95a/code/experiments/ecro_attr95a/run95a.sh | head;
  -> 21c21
<         "T_OPS": "A1 = S1_ops nmf16_wl92, A2 = S1_ops nmf16_l1irls (WL95, file sha recorded); A3 = rank-32 field-adapted non-negative L1 operator regenerated deterministically with the WL95 wl95_calc2 ladder code (seed 9500 + 320 + c, W94 fit weights as in WL95) and must reproduce wl95_calc2 ...[cut]... es are not opened beyond the Stage 0 test flags."
---
>     "pre_freeze_exposure": "All WL92-WL95 results are known. Dev run ~/rest_x95adev (D = 2 / 4 draws, q Mq = 64, 32-spp renders): Stage 0 test flags were read (all PASS; first attempt crashed on a JSON print of numpy bools after writing s0.json, fixed); the chain exercised main, renders, analysis and the rule-based extension path ('AN EXTEND' printed). No dev metric value, table or figure was opened."
--- a95.py:
--- run95a.sh:

96 20:59:55 .
96 20:59:26 ..
00 20:59:55 a95.py
96 20:54:39 protocol
56 20:58:42 run95a.sh
96 20:53:59 wl95calc
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\build_map.py
RUN [Build the machine-readable dependency map and check files] python experiments/audit97/MAP/build_map.py && python -c "import json;d=json.load(open('results/audit97/MAP/dependency_map.json'));print(list(d.keys()));print([ (r['worklog'],r['n_identical_to_stated'],r['n_different'],r['n_not_in_stated_tree']) for r in d['runs_code_revision']][:6])" && rm -f results/audit97/MAP/static_deps.txt.bak && file experiments/audit97/MAP/*.py | cut -c1-120 && git status --short | head
  -> wrote c:\Projects\ReST-GI\results\audit97\MAP\dependency_map.json 76769 bytes
['audit', 'repo_head', 'generated_by', 'runs_code_revision', 'protocol_hashes', 'artifact_identities', 'batch_edges', 'static_code_dependencies', 'shared_dependencies', 'wl89_episode', 'p0_claim_chains', 'pre_freeze_runs' ...[cut]...  78, 0), ('WL86A', 9, 88, 1), ('WL87', 91, 0, 0), ('WL88', 105, 2, 0), ('WL89', 119, 0, 0), ('WL90', 128, 0, 0)]
experiments/audit97/MAP/build_map.py:             Python script, ASCII text executable, with very long lines (749)
experiments/audit97/MAP/cmp96_probe_vs_frozen.py: Python script, ASCII text executable
experiments/audit97/MAP/compare_blobs.py:         Python script, ASCII text executable
experiments/audit97/MAP/compare_crlf.py:          Python script, ASCII text executable
experiments/audit97/MAP/extract_deps.py:          Python script, ASCII text executable
?? experiments/audit97/
WRITE c:\Projects\ReST-GI\experiments\audit97\MAP\build_map.py
RUN [Build the machine-readable dependency map and check files] cd /c/Projects/ReST-GI && python experiments/audit97/MAP/build_map.py && python -c "import json;d=json.load(open('results/audit97/MAP/dependency_map.json'));print(list(d.keys()));print([ (r['worklog'],r['n_identical_to_stated'],r['n_different'],r['n_not_in_stated_tree']) for r in d['runs_code_revision']][:6])" && rm -f results/audit97/MAP/static_deps.txt.bak && file experiments/audit97/MAP/*.py | cut -c1-120 && git status --short | head
WRITE c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\MAP_evidence_dependency_and_provenance.md
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
