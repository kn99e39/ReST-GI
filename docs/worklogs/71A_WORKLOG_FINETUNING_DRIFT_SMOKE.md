# Worklog — Fine-Tuning Drift Smoke, WL71A (2026-11-03)

**Numbering.** This worklog is numbered 71A under the user's rule of 2026-11-03: small work that does not receive a formal number is labelled with the last worklog number plus a letter, in alphabetical order. It is a diagnostic-only smoke on development data, not a research batch. WL71 is unchanged.

**Authorisation.** On 2026-11-02 the user issued a pasted brief: "Investigate whether physical fine-tuning of the unchanged WL59 K family destroys cross-source generalization as optimization progresses" — a bounded diagnostic, with no architecture development and no hyperparameter optimisation. The user also gave a separate instruction for the run: "이거 63번 서버에서 돌려라" (run this on LabServer63).

**Repository copies.** The run used the local root `C:/Users/dna10/rest_drift_smoke` and the server root `~/rest_drift_smoke`. The repository holds copies:
- code and protocol: `experiments/fnto_drift71a/`;
- results: `results/fnto_drift71a/`, including the original report, `eval.json`, `label.json`, `tests.json` and checkpoint hashes;
- reviewer figure: `results/evaluation/71A/`.

The paths in the report below refer to the run roots.

This is a diagnostic-only smoke on development data. It is not a numbered worklog. WL71, WL68 and the earlier smoke files were read-only inputs. Protocol: `drift_protocol.json`, written before any run.

**Labels under the predeclared rule:** **TRAINING DRIFT SUPPORTED (both arms R and N)**. **SCALING SENSITIVITY: not supported** (R/N ratios ≤ 1.05).

## Implementation Fact

**Machine.** LabServer63, as the user instructed. RTX 3080 Ti, torch 2.3.1+cu121. One server chain (`chain_drift.sh`): tests → R and N trained concurrently → evaluation → label.

**Root and code.**

| Item | Value |
|---|---|
| Server root | `~/rest_drift_smoke` |
| Local root | `C:/Users/dna10/rest_drift_smoke` |
| Code snapshot | `repo/`, `git archive` of HEAD `e96a8d6` |
| Driver | `drift.py` (sha `d21a2403…`) |
| Protocol sha | `a6e521be…` |

Before the run, `frozen_hashes_before_run.txt` recorded the hashes of the driver, the protocol, the chain script and the historical inputs. They were re-verified on the server before launch.

**Data path.** Training reused WL71's data path unchanged:

- `fit71.Problem` with population S4, replicas 0–3, rest and pt0;
- init `fit71.model_from('P_5902')`, i.e. P_5902 with the fixed WL68 net_all_5902 A/O;
- group 'net' (A/O frozen; 155,104 of 157,024 parameters trainable).

**Optimizer.** Adam lr 3e-3 → 3e-5, cosine over 20,000 steps, `clip_grad_norm_` 1.0, float32, TF32 off, `torch.manual_seed(0)`.

**Arms.**

- **R:** loss = J. This is the WL71 contract.
- **N:** loss = J / J₀, with J₀ = 1.982082 computed once at the common init.

Checkpoints were saved after 0, 500, 2,000, 6,000, 10,000 and 20,000 updates. No checkpoint was selected on held-out results.

**Evaluation.** The eval68 estimators, unchanged: unbiased cross-replica Grams, jackknife upper 95%, worst channel.

| Group | Contents |
|---|---|
| FIT | Slots 0–3, replicas 4–7 |
| H1 | Slots 4–7 at rest and at pt0, all 8 replicas |
| H3 | cf3 slots 4–7 |
| H2 | cf3 slots 0–3 |
| DEF | Signed changes, qualified at signal/SE ≥ 3: cf3 slots 4 and 5; pt0 slots 4 and 7; at the fit ports pt0 slots 0, 1 and 3 |

All abs matrices are qualified at signal/SE ≥ 10. The evaluation also covers:

- the WL71 Stage-1 flux fraction within 1 cm;
- relative T change from init;
- passivity;
- the WL59 historical full fields.

**Focused tests (`out/tests.json`): all passed.**

- The two arms start from the same init (tsha `a5d1c087…`), equal to the earlier smoke's init.
- The A/O are frozen, and the A/O are bitwise equal across net_all_5902, smoke student A and WL71 C_5902.
- The S4 targets are equal to WL68 `fit_targets` and to the smoke's targets: `73b49656…` / `d9971065…`.
- **Objective identity.** J₀ is the same through `fit71` and `family68`: 1.982081651687622 both.
- **Scaling equivalence.** grad(J/J₀) = grad(J)/J₀ to 1e-5.
- **Clipping accounting.** The value `clip_grad_norm_` returns equals the pre-clip norm, and the post-clip norm is ≤ 1.
- **Prediction equivalence.** Batched and per-port columns agree to 3.3e-7.
- **Perturbed passivity.** The maximum column sum is 0.9939.
- **Isolation.** Training opened only `fit71`'s allowed files, verified by a `torch.load` spy.

## Measurement

### Stage A — reconciliation

Smoke student A and WL71 C_5902 were re-scored with this evaluator on H3 (cf3 slots 4–7).

| Model | Re-scored (coarse / fine16 / contrast) | Reported | Max difference | Check |
|---|---|---|---|---|
| smoke A | .0702 / .1540 / .2676 | .070 / .154 / .268 | 4e-4 | **passed** |
| C_5902 | .4935 / .5496 / .9273 | WL71 | 9e-9 | **passed** |

**Contract differences recorded:**

| Factor | Smoke A | WL71 C_5902 |
|---|---|---|
| Schedule | Cosine over 2,000 steps | Cosine over 20,000 steps |
| Loss | J / J₀ | J |
| Software | torch 2.8 on an RTX 5080 | torch 2.3.1 on LabServer63 |

**Matched:** data, init, A/O, objective, optimizer and clipping.

### Stage B/C — trajectories

Values are upper 95%, worst channel: coarse / fine16 / contrast. Criteria are .10 / .15 / .25.

| Checkpoint | Training J | FIT rest | H1 rest (unseen) | H1 pt0 (unseen) | H3 cf3 (unseen) | H2 cf3 (trained positions) |
|---|---|---|---|---|---|---|
| init | 1.982 | .141/.324/1.15 | .189/.319/1.09 | .191/.321/1.10 | .191/.322/1.09 | .141/.324/1.15 |
| R@500 | .0254 | .043/.118/.16 | **.080/.136/.33** | .086/.140/.33 | .081/.138/.33 | .044/.118/.16 |
| R@2000 | .0182 | .038/.094/.14 | .089/.200/.32 | .095/.206/.33 | .089/.200/.33 | .038/.095/.14 |
| R@6000 | .0170 | .037/.089/.13 | .127/.272/.57 | .129/.277/.58 | .129/.272/.58 | .038/.089/.13 |
| R@10000 | .0168 | .037/.088/.13 | .410/.452/.87 | .410/.461/.90 | .406/.450/.86 | .038/.088/.13 |
| R@20000 | .0165 | .037/.087/.13 | **.511/.566/.92** | .510/.578/.99 | .508/.565/.90 | .039/.087/.14 |
| N@500 | .0246 | .046/.119/.16 | .082/.136/.33 | .089/.141/.33 | .083/.139/.33 | .047/.119/.16 |
| N@2000 | .0202 | .040/.095/.14 | .088/.194/.34 | .095/.202/.34 | .090/.194/.35 | .041/.096/.14 |
| N@6000 | .0174 | .038/.090/.13 | .126/.265/.58 | .127/.269/.58 | .126/.266/.58 | .039/.090/.14 |
| N@10000 | .0168 | .038/.088/.13 | .398/.427/.81 | .401/.435/.83 | .396/.428/.81 | .039/.088/.13 |
| N@20000 | .0166 | .038/.087/.13 | .492/.542/.89 | .500/.554/.95 | .491/.544/.89 | .039/.087/.13 |
| smoke A | — | .039/.098/.14 | .069/.152/.26 | .080/.162/.27 | .070/.154/.27 | .040/.098/.14 |
| WL71 C_5902 | — | .037/.087/.13 | .497/.551/.94 | .496/.563/1.00 | .493/.550/.93 | .039/.087/.14 |
| frozen P_5902 (WL59 A/O) | — | .104/.255/.90 | .154/.254/.95 | .154/.253/.95 | .155/.258/.95 | .104/.255/.90 |

**Drift rule** (H1 worst; ratio of the 20,000-step value to the 2,000-step value):

| Arm | Coarse | Contrast | Training J | FIT worst |
|---|---|---|---|---|
| R | .095 → .511 (×5.37) | .327 → .990 (×3.02) | .0182 → .0165 | .138 → .134 |
| N | .095 → .500 (×5.26) | .345 → .945 (×2.74) | .0202 → .0166 | .144 → .134 |

**Scaling rule** (max ratio R/N or N/R, H1):

| Checkpoint | Coarse | Contrast |
|---|---|---|
| 2,000 | 1.003 | 1.053 |
| 20,000 | 1.023 | 1.047 |

**Per source, rest.** Coarse / contrast at H1 slots:

| Checkpoint | Slot 4 (port 2148, near the fit ports) | Slot 5 (1025) | Slot 6 (2707, bottom corner) | Slot 7 (2422) |
|---|---|---|---|---|
| R@500 | .07 / .28 | .09 / .20 | .13 / .44 | .03 / .17 |
| R@2000 | .06 / .27 | .07 / .24 | .18 / .43 | .03 / .17 |
| R@6000 | .07 / .24 | .08 / .32 | .26 / .89 | .04 / .40 |
| R@10000 | .08 / .28 | .21 / .66 | .90 / 1.38 | .06 / .77 |
| R@20000 | .10 / .46 | .52 / .91 | 1.07 / 1.27 | .11 / 1.25 |
| frozen | .17 / .96 | .17 / .96 | .22 / .96 | .06 / .75 |

pt0 shows the same pattern. The N arm is similar: slot 6 goes .13 → 1.05 and slot 5 goes .09 → .43.

**Near-source flux concentration** (fraction within 1 cm, Stage-1 definition, rest):

| Ports | Physical | init | R@500 / 2,000 / 6,000 / 10,000 / 20,000 |
|---|---|---|---|
| Fit ports 0–3 | .898–.900 | .902–.908 | .90 → .91–.92 (stable) |
| Slot 5 | .899 | .903 | .917 / .916 / .916 / .872 / .827 |
| Slot 6 | .900 | .908 | .904 / .915 / **.834 / .475 / .304** |
| Slot 7 | .899 | .903 | .894 / .902 / .894 / .885 / .861 |

N is similar: slot 6 goes .906 → .383. WL71 C_5902 has .316 at slot 6.

**Relative T change from init** (rest):

- At the fit ports it reaches .72–.78 by 500 steps and .81–.82 by 20,000.
- At slot 6 it is .56 (500), .70 (2,000), .56 (6,000), .80 (10,000) and .94 (20,000).

**Signed deformation** (worst-channel error / minimum correlation):

| Checkpoint | cf3 − rest, slots 4–5 | pt0 − rest, slots 4 and 7 | pt0 − rest, fit ports |
|---|---|---|---|
| init | .83 / .65 | 1.35 / .57 | .53 / .86 |
| R@500 | .86 / .53 | 1.23 / .71 | .22 / .98 |
| R@2000 | .85 / .53 | 1.03 / .77 | .21 / .98 |
| R@6000 | 1.02 / .52 | 1.02 / .83 | .20 / .98 |
| R@10000 | 1.30 / .35 | 1.02 / .70 | .19 / .98 |
| R@20000 | 1.87 / .20 | 2.74 / .64 | .19 / .98 |
| N@20000 | 2.65 / .22 | 2.90 / .19 | .20 / .98 |
| frozen | .84 / .68 | .93 / .79 | .31 / .97 |

**Historical full fields** (rest coarse mean, R/G/B):

| Model | Values |
|---|---|
| frozen | .062 / .063 / .066 |
| init | .083 / .094 / .123 |
| R@500 | .14 |
| R@2000 | .14 |
| R@6000 | .16 |
| R@10000 | .29 / .34 / .49 |
| R@20000 | .45 / .49 / .97 |
| N@20000 | .41 / .44 / .67 |

**Passivity.** Passed for every checkpoint at rest, pt0 and cf3. The maximum column sum is ≤ 0.99990.

**Gradients and clipping.**

| Arm | Clipped steps | Steps clipped in 0–50 | First 5 norms |
|---|---|---|---|
| R | 0.02% overall | 4 of 50 | 2.14 → 0.92 |
| N | 0.0% overall | 1 of 50 (2%) | 1.08 → 0.47 |

After step 500 neither arm clips any step. The median raw-J gradient norm is .052 / .032 / .034 / .028 over windows 0–500 / 500–2,000 / 2,000–6,000 / 6,000–10,000, then .0041–.0047 over 10,000–20,000.

**Cost.** Each arm took 1,025 / 1,032 s, running concurrently, with a peak of 1.06 GB. Evaluation took 595 s.

## Observation

1. **The two contracts are reconciled.** The same evaluator reproduces both historical numbers, with H3 to 4e-4 and 9e-9. Under one identical contract, the 20,000-step R arm reproduces WL71 C_5902 (H1 rest .511 vs .497; slot-6 flux .304 vs .316).
2. **Cross-source error is non-monotone.** Training-port error and training J improve monotonically to 20,000 steps. Cross-source error on H1, H3 and pt0 falls to its lowest observed value at 500–2,000 steps, at about smoke A's level, then rises strongly after ~6,000 steps: coarse ×5.3–5.4 and contrast ×2.7–3.0 from 2,000 to 20,000. Trained positions at a changed geometry (H2) stay at the fit level throughout.
3. **The degradation is concentrated at the source farthest from the fit sources.** Slot 6 is in a bottom corner and shares no receivers with the fit sources. There the predicted near-source flux concentration collapses from the physical .90 to .30–.38, while the fit ports keep .91–.92. Slot 5, which also shares no receivers, degrades next.
4. **Loss scaling does not change the trajectory materially.** R/N differs by ≤ 5% at matched checkpoints. Clipping is active only in the first ~50 steps (R 4, N 1) and never after step 500.
5. **Signed deformation and full-field fidelity degrade with training.** Deformation off the fit ports and the historical full-field fidelity both worsen most after 6,000–10,000 steps; at the fit ports the deformation stays good (.19–.22 / .98). No checkpoint meets the contrast criterion at the unseen ports: the best is .32–.35. No checkpoint matches frozen WL59's off-fit deformation correlation.

## Interpretation

These are inferences, not measurements.

- **The WL71 / smoke discrepancy is explained by training progression.** Under a matched contract, continued optimisation of the sparse physical objective first improves and then destroys cross-source transfer, while the fit keeps improving. This is generalisation degradation during training.
- **Loss scaling and clipping are not the cause.** The distinction between J and J/J₀ does not explain the discrepancy. The smoke's better transfer comes from its short schedule (2,000 steps, learning rate annealed early), which stops the trajectory near the early, still-transferable regime.
- **The pattern fits the WL71 hypothesis, but does not establish it.** The degradation appears late, when the fitting objective is nearly flat (gradient norm about 0.004, against Adam's constant-size steps). It is largest where the fit sources give no constraint, and it removes the physical near-source concentration only there. This is consistent with progressive specialisation to the supervised source positions, but it does not prove absolute-position memorisation.
- **The early regime is not a qualified generalising rule either.** Its unseen-port contrast (.32–.35) and its off-fit deformation still fail the criteria. Early stopping would therefore not deliver a passing model on this evidence. The best checkpoint was not selected; the early values are reported only as observed.

## Unresolved Question

1. **What drives the late drift?** Is it the low-gradient Adam phase, an implicit drift along directions the S4 objective does not constrain, or late fitting of residual target noise? Lower learning rates or a different optimizer were not tested and are not authorised here.
2. **Does the same drift occur under S16 coverage?** WL71's D1 final result suggests a milder version, but its trajectory was not observed.
3. **Is the early regime's moderate transfer a property of the P_5902 initialisation** (frozen-map smoothness) **or of the objective?** A single init was used.
4. **Seed and nondeterminism variability.** There is one run per arm, and CUDA `index_add` is nondeterministic.

## Architecture Verdict

Findings:

- **TRAINING DRIFT SUPPORTED** in both arms.
- **SCALING SENSITIVITY not supported.**

**Completion question.** *Does sparse-source physical fine-tuning destroy an initially transferable transport rule, and is the observed failure explained by training progression or objective scaling?*

- **Yes, in this setting.** The P_5902-initialised map, briefly fine-tuned, transfers moderately: H1 about .08–.09 / .14–.20 / .33. Continued sparse fine-tuning destroys that transfer, most at the uncovered corner source, while the fit keeps improving.
- **The failure is explained by training progression, not by objective scaling.**
- **The early regime still misses the criteria.** It never meets the unseen-port contrast criterion or the off-fit deformation criterion, so this does not show that an early-stopped map is a qualified reusable rule.

No architecture, K, WL71 or living-document change follows. No result here is evidence about relative-geometry kernels.

## Reproducibility

**Commands.** `chain_drift.sh` on LabServer63. Logs are in `logs/`, including the 15 s GPU sampler.

**Inputs** (read-only):

- WL68 inputs, manifest `2abf3ba6…`;
- `fit_targets.pt` `3c21b619…`;
- `net_all_5902` `132303b2…`;
- WL71 `C_5902` `d17f1876…` and `fit_targets71.pt`;
- smoke `student_A.pt` `0a49f98d…`;
- WL71 `analysis.json` copy `d8aeaa88…`.

**Outputs** (`out/`):

- `tests.json`;
- `train_R.pt` / `train_N.pt`, containing the per-step J and gradient norm, window statistics, J₀ and the opened-file list;
- `ckpt_{R,N}_{00000…20000}.pt`, containing the model, optimizer state and tsha;
- `eval.json`;
- `label.json`.

Checkpoint sha256 prefixes:

| Step | R | N |
|---|---|---|
| 0 | `146aa3f6` | `a01336e9` |
| 500 | `ef9f963a` | `9d321f48` |
| 2,000 | `fcc4caca` | `6e6f97d8` |
| 6,000 | `dbedb7ec` | `36529326` |
| 10,000 | `2f8c0866` | `174e0972` |
| 20,000 | `ee1a5011` | `3d0b3c00` |

The step-0 files differ only by the arm/scale metadata; their model tsha is identical, `a5d1c087…`.

**Seeds.** `torch.manual_seed(0)`. Training is full batch, so it makes no random draws.
