# Worklog — Streaming Multi-Packet Walker Engine, WL72A (2026-11-03)

## Numbering and authorisation

**Number.** This is worklog 72A under the user's letter-suffix rule: it is the first small unnumbered task after the most recent numbered worklog (72). Before this worklog existed, the work was referred to as "WL71C" in chat. Its code directory, `experiments/fnto_walker71c/`, keeps that name so that the validated file hashes stay valid.

**User instructions:**

- **2026-11-02.** While watching WL71 reference generation (about 43% mean SM, one CPU core at 100%), the user said: "일단 지금은 그대로 두되, 현재 작업이 끝나면 해당 구조를 개편해서 GPU Utilization 효율을 올리도록 해" (leave it for now; once the current work is done, restructure it to raise GPU utilisation).
- **2026-11-03.** The user chose to run this alongside WL72 ("병행") and asked: "코드 작성 후 로컬에서 검증하면 안돼?" (can't you validate locally after writing the code?).

The validation therefore ran on the local PC.

## What was slow

`r57_fastwalk.walk` handles each 32M-photon chunk as follows:

1. About 1 s of dense GPU work while the whole chunk is in flight.
2. About 2 s in which the few remaining photons of the long high-albedo path tail are advanced one collision per Python loop iteration.

During the second phase each iteration launches small kernels and synchronises. The GPU sits mostly idle while one CPU thread is saturated.

Every packet pays this tail once. The tail cannot overlap anything, because packets run one after another.

## Engine (`experiments/fnto_walker71c/engine71c.py`)

**Schedule.** Several packets share one active photon pool. New photons are emitted whenever the pool falls below a low-water mark:

- pool 16M, emission 4M;
- up to 6 packets in flight, each with its own accumulator slot.

One packet's tail therefore overlaps the next packet's burst.

**Per-photon physics, identical to the walker and to the WL63 tagged action:**

- one-hot importance source;
- weight 1/N;
- exponential free flight;
- Henyey–Greenstein phase;
- albedo per collision;
- first exit into (port, 8×8 bin) with first and second moments;
- zero-collision (B) / collided (Scat) split, decided by the photon's own collision count;
- leaks, wrong-side exits and truncation;
- Russian roulette with the WL57 rule. It is applied per photon age, which is identical to the per-chunk iteration counter for photons emitted together.

**What changes.** Only the execution order and the random stream: one generator per run, a master seed and a fixed packet order. Equivalence with the WL57 walker is therefore statistical, not bitwise.

## Validation (`validate71c.py`, local PC)

**Setup.** RTX 5080, torch 2.8 (cu128), Mitsuba `cuda_ad_rgb`.

- **Packets:** 12 at 32M photons: 8 on rest (ports 2085 ×3 bins, 2707 ×2, 1123, 2422, 1025) and 4 on pt0.
- **Comparison:** the unchanged WL57 walker plus WL63 tags (one 32M chunk per packet, as in WL71/WL72) against the engine, with independent seeds.
- **Rule** (fixed before the 32M run):
  - |energy z| ≤ 4 per channel for B and Scat;
  - coarse-cell mean z² in [0.5, 1.5] where ≥ 30 cells have variance;
  - both physical gates pass.

**Result: PASSED** (`results/fnto_walker72a/validation.json`, engine sha `f224dedc…`).

| Component | Statistic | Range over the 12 packets |
|---|---|---|
| Scat | energy z | −0.97 … +0.38 |
| Scat | coarse mean z² | 0.978–1.035 (4,746–7,488 cells) |
| B | energy z | −2.35 … +1.27 |

B has 3–27 cells, so its z² rule does not apply. B energies are 1e-6 to 1.8e-3 of the input, and the two walkers agree to within those z.

**Throughput** (same packets, wall time incl. synchronisation; GPU utilisation is the mean of 0.5 s `nvidia-smi` samples under Windows):

| Geometry | WL57 walker | Engine | Speed-up | GPU utilisation |
|---|---|---|---|---|
| rest, 8 packets | 19.39 s | 15.04 s | 1.29× | 48% → 72% |
| pt0, 4 packets | 7.88 s | 6.10 s | 1.29× | 56% → 68% |

The engine's peak memory is 8.4 GB, against 10.0 GB for the walker.

**Tuning attempts, all disclosed and none adopted:**

- Pool 12M and emission 2M give the same time (15.1–15.5 s).
- Pool 24M exceeded the 16 GB GPU. Under Windows WDDM this pages to host memory and becomes extremely slow; the run was stopped.
- A "sync-free" variant that replaced the host branches with full-pool masked operations raised utilisation to 85% but took 26.6 s, so it was reverted. Utilisation alone is not the target; throughput is.
- The retained version checks packet completion every 8 steps while emitting.
- The first 32M validation, run before that change, passed with identical per-packet values. It is preserved as `validation_attempt1_pre_sync_tuning.json`.

## Limits and status

- **Machine.** Validated only on the local PC (torch 2.8, RTX 5080). It has not been validated on LabServer63 (torch 2.3.1, RTX 3080 Ti), whose GPU is busy with another project's job. The engine is copied to `~/rest_walker71c/` on LabServer63 with its record named `validation.json`.
- **WL72 does not use it.** WL72's frozen reference code accepts only a drop-in walker. Its tag instrumentation requires exactly one sink call per step with a common step index, which a multi-packet engine cannot satisfy. WL72 therefore uses the unchanged WL57 walker. The file name `validated.json` that WL72 checks for is deliberately not created.
- **Not yet used for any evidence.** Using the engine in a future batch needs its own integration (a packet runner calling `Engine.run`), plus a server-side identity check on the execution machine.
- **The remaining idle time is not attributed.** It is about 28% on the 5080 and comes from per-step compaction and synchronisation plus Mitsuba intersection launches.
