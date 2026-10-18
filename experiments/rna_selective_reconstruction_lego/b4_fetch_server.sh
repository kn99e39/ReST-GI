#!/usr/bin/env bash
# Worklog 37 (Windows Git Bash, detached): copy each LabServer63 held-out set when its .done marker appears, verify the
# size, then write results/.../launcher/fetched_<set> for the WSL chain.
L=/c/Projects/NeuralShaderDemo/results/rna_selective_reconstruction_lego; S=nsd_b4
for v in T0 T1; do for s in 1 2; do n=heldout_${v}_s$s
  until ssh -o BatchMode=yes -o ConnectTimeout=20 LabServer63 "test -f $S/launcher/$n.done" 2>/dev/null; do sleep 120; done
  for f in ${n}_val.h5 ${n}_val.json; do
    for try in 1 2 3; do
      scp -q LabServer63:$S/data/$f $L/data/$f.part && \
      [ "$(stat -c %s $L/data/$f.part)" = "$(ssh -o BatchMode=yes LabServer63 "stat -c %s $S/data/$f")" ] && mv $L/data/$f.part $L/data/$f && break
      sleep 60
    done
  done
  [ -f $L/data/${n}_val.h5 ] && echo "fetched $(date -Is)" > $L/launcher/fetched_$n
  echo "$n $(date -Is)" >> $L/launcher/fetch.log
done; done
echo "fetch done $(date -Is)" >> $L/launcher/fetch.log
