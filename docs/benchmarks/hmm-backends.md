# Choosing an HMM backend

**Measured 2026-09-17** on a 4-vCPU Intel Xeon @ 2.20 GHz, Python 3.12.3, with an NVIDIA L4
present and torch running on the CPU (4 threads). Read
[`README.md`](README.md) first: these numbers are dated observations, not promises.

## What is fixed, and what is measured

`pfsmgraph.hmm`'s kernels come in the [ADR 0002](../design/adr/0002-three-phase-algorithm-lifecycle.md)
lifecycle phases, chosen per call by name
([ADR 0021](../design/adr/0021-runtime-backend-selection.md)). Two statements about them are
contracts rather than measurements, and both are tested:

- **`python`, `cython`, `cpu_parallel` and `cuda` return bit-identical results on one host**
  ([ADR 0020](../design/adr/0020-scaled-probability-domain-forward-backward.md)). Choosing
  among them is a speed choice and nothing else.
- **`torch` is held to a tolerance**, `N · eps · max(1, x)` per element, because its E-step
  derives counts as gradients. It is the only backend whose result can differ at all.

Nothing falls back: a backend that cannot run here raises rather than quietly running
something else.

## Guidance

| Backend | Needs | Result | Choose it when |
|---|---|---|---|
| `cython` | the compiled extension (every published wheel) | bit-identical | the default at these model sizes: fastest measured phase at `S ≤ 8`, 120–330× the numpy forward pass |
| `python` | nothing | the reference | debugging, or comparing against a phase you suspect |
| `cpu_parallel` | `numba` (`cpu-parallel` extra) | bit-identical | large `S`, where one timestep's `S²` work covers a parallel region's cost; it loses badly at small `S` |
| `cuda` | `numba-cuda` and a device (`gpu` extra) | bit-identical | large `S`, or batched decoding; about 500 ms per corpus forward pass at `S ≤ 8`, where a launch per timestep dominates |
| `torch` | `torch` (`torch` extra) | within tolerance | gradient-based work, or training on a GPU through `baum_welch(..., device=)`; **not** for topology search, see below |

Two practical consequences of the table:

- **A topology search should run on `cython` today.** Both measurements below were taken at
  `S ≤ 8` on a 1,268-symbol corpus, where every parallel phase is slower than Cython.
- **A search cannot put torch on a GPU.** `device=` belongs to `baum_welch`; the private
  search and trial functions do not carry it, so `backend="torch"` there runs on CPU
  threads. `backend="cuda"` and `score_backend="cuda"` do reach the device.

## Measurement 1: where a search round's time goes

Script: `.scratch/hmm-lush/measurements/search_round_profile.py`, SHA-256
`ef4ed52139dfc29c6b426cdc4de0a345aece8f3eee04ebe84f44f5d4397ec730`.

One `_suggest_move` round on `m001_0005_005` (5 states, 10 merges and 10 splits) over the
`set02a_200` corpus as one record, `cython` throughout, a 200-cycle split floor. "Before" is
the code as it stood until [ADR 0024](../design/adr/0024-search-compiled-work.md) §2, when
`baum_welch`'s convergence check always ran the numpy forward pass; the script rebuilds it
by patching that one call, so both configurations run in one process.

| Configuration | Wall time | Under `cProfile` | numpy `_forward_backward` | `_corpus_step` (Cython E-step) |
|---|---|---|---|---|
| Before: check on numpy | 9.38 s | 12.65 s | 10.63 s over 236 calls (84%) | 1.14 s over 2,360 calls |
| After: check on `score_backend` | **1.70 s** | 2.04 s | not called | 1.13 s over 2,360 calls |

**A round got 5.5× faster, and the E-step did not move.** After the change the Cython E-step
is two thirds of a round, so the next saving is fewer EM cycles or parallel trials, not more
compiling.

```
host: x86_64, 4 CPUs, python 3.12.3
before (check on numpy): 20 moves in 9.38 s (12.65 s under cProfile); numpy _forward_backward 10.63 s over 236 calls (84%); _corpus_step 1.14 s over 2360 calls
after (check on score_backend): 20 moves in 1.70 s (2.04 s under cProfile); numpy _forward_backward 0.00 s over 0 calls (0%); _corpus_step 1.13 s over 2360 calls
same ranked moves and bit-identical totals: True
```

## Measurement 2: does `torch` change the answer, or only the clock?

Script: `.scratch/hmm-lush/measurements/torch_search_path.py`, SHA-256
`f239a50143106179a38b4ea0690e0b9fb44883759b8cb05563b60f4e6c808c5e`.

`torch` is the one backend whose result may differ, and [ADR 0023](../design/adr/0023-topology-search-loop.md)'s
Open item gives the mechanism by which a small difference could matter: near the optimum a
candidate's total is a step function of its converged parameters, because the best integer
`d` steps as EM runs on, and each step moves the model half by 58–71 bits. So a search on
`torch` might rank candidates differently from the bit-identical phases.

Every comparison below sets `score_backend="cython"` on both sides, so the scan and every
convergence check are bit-identical and **only the E-step differs**. Candidates are matched
by identity `(kind, states, trial)`, never by rank, so a reordering cannot pair different
candidates.

| Comparison | Candidates | `cython` | `torch` | EM cycles summed (cython / torch) | `d` differs | Totals |
|---|---|---|---|---|---|---|
| Round from `m001_0001_001`, S = 1 | 2 splits | 0 s | 124 s | 400 / 400 | 0 | tie in 2 of 2 |
| Round from `m001_0005_005`, S = 5 | 10 merges, 10 splits | 2 s | 742 s | 2,340 / 2,340 | 0 | tie in 20 of 20 |
| Round from `m008_0001_008`, S = 8 | 28 merges, 16 splits | 8 s | 2,768 s | 8,500 / 8,500 | 0 | tie in 44 of 44 |
| 3-round search from the one-state start | 3 rounds | 2 s | 802 s | equal at the start and every round | 0 | same moves, best 1774.0328 |

**Torch agreed everywhere, and cost 350–400× more.** All 66 candidates chose the same `d`,
converged in the same number of EM cycles — which vary from 30 to 490, so the split floor is
not masking the comparison — and scored totals equal to the last bit. The rankings, the
winners and all three search moves matched, including at `m008`, where the winner led the
runner-up by 0.0513 bits.

**Why the totals are exactly equal, when the E-step is only within a tolerance.** The
unrounded data lengths did differ, by at most 1e-11 bits. The criterion quantizes the
parameters at `d` decimals before scoring (`_mdl.py`'s `_quantize`, and
[ADR 0023](../design/adr/0023-topology-search-loop.md) §5 for how `d` is chosen), so a
difference that small falls inside the rounding grain and disappears from the total. A
trial's result keeps both quantities, so the difference is visible in `data_bits` and gone
from `total_bits`. ADR 0023's step sensitivity needs EM to stop somewhere *materially*
different; an ulp is not that. The mechanism is real, but this perturbation is ten orders of
magnitude too small to trigger it.

### The logs

Two runs, because the script grew between them. Run 1 measured the 1- and 5-state rounds;
the script was then extended with the EM-cycle sums, the per-candidate "which backend scored
lower" column and a `--models` option, and run 2 measured the 8-state round and the search.
**Run 1's version of the script was not preserved**, so the hash above is run 2's; run 1's
output differs only in lacking the `lower` field and the summary line.

Run 1:

```
host: x86_64, 4 CPUs, python 3.12.3
m001_0001_001.hmm S=1 cython: 2 moves in 0 s
m001_0001_001.hmm S=1 torch: 2 moves in 124 s
  ('split', (0,), 0): d 10/10, cycles 200/200, data +4.55e-13, total 2434.0131/2434.0131
  ('split', (0,), 1): d 10/10, cycles 200/200, data +4.55e-13, total 2434.0131/2434.0131
  summary: d differs in 0, cycles in 0, max |data| 4.55e-13 bits, max |total| 0 bits; same ranking True, same winner True, cython winner's lead 0 bits
m001_0005_005.hmm S=5 cython: 20 moves in 2 s
m001_0005_005.hmm S=5 torch: 20 moves in 742 s
  ('merge', (0, 4), 0): d 3/3, cycles 30/30, data +0, total 1774.0328/1774.0328
  ('split', (1,), 0): d 4/4, cycles 200/200, data -9.09e-13, total 1808.3381/1808.3381
  ('split', (1,), 1): d 4/4, cycles 200/200, data -1.36e-12, total 1808.3381/1808.3381
  ('split', (3,), 0): d 5/5, cycles 200/200, data -7.73e-12, total 1820.2345/1820.2345
  ('split', (3,), 1): d 5/5, cycles 200/200, data +0, total 1820.2345/1820.2345
  ('split', (4,), 0): d 6/6, cycles 200/200, data +1.14e-12, total 1836.4732/1836.4732
  ('split', (4,), 1): d 6/6, cycles 200/200, data +1.14e-12, total 1836.4732/1836.4732
  ('split', (0,), 0): d 6/6, cycles 200/200, data -2.27e-13, total 1838.4732/1838.4732
  ('split', (0,), 1): d 6/6, cycles 200/200, data +0, total 1838.4732/1838.4732
  ('split', (2,), 0): d 4/4, cycles 200/200, data -6.14e-12, total 1843.7815/1843.7815
  ('split', (2,), 1): d 4/4, cycles 200/200, data +6.82e-13, total 1844.5268/1844.5268
  ('merge', (0, 3), 0): d 2/2, cycles 40/40, data -2.27e-13, total 1939.9115/1939.9115
  ('merge', (3, 4), 0): d 2/2, cycles 40/40, data +0, total 1943.0814/1943.0814
  ('merge', (0, 1), 0): d 4/4, cycles 40/40, data -2.27e-13, total 1960.2274/1960.2274
  ('merge', (1, 4), 0): d 4/4, cycles 40/40, data -4.55e-13, total 1964.2274/1964.2274
  ('merge', (0, 2), 0): d 4/4, cycles 40/40, data +4.55e-13, total 2062.7206/2062.7206
  ('merge', (2, 4), 0): d 4/4, cycles 40/40, data +0, total 2068.7206/2068.7206
  ('merge', (1, 3), 0): d 3/3, cycles 30/30, data -2.27e-13, total 2097.6538/2097.6538
  ('merge', (2, 3), 0): d 4/4, cycles 30/30, data +0, total 2103.3672/2103.3672
  ('merge', (1, 2), 0): d 4/4, cycles 30/30, data -3.64e-11, total 2406.4304/2406.4304
  summary: d differs in 0, cycles in 0, max |data| 3.64e-11 bits, max |total| 0 bits; same ranking True, same winner True, cython winner's lead 34.3 bits
m008_0001_008.hmm S=8 cython: 44 moves in 8 s
```

Run 2:

```
host: x86_64, 4 CPUs, python 3.12.3
m008_0001_008.hmm S=8 cython: 44 moves in 8 s
m008_0001_008.hmm S=8 torch: 44 moves in 2768 s
  ('merge', (0, 1), 0): d 3/3, cycles 40/40, data +0, total 1932.7242/1932.7242, lower tie
  ('merge', (4, 5), 0): d 3/3, cycles 100/100, data -9.09e-13, total 1932.7754/1932.7754, lower tie
  ('merge', (0, 6), 0): d 3/3, cycles 40/40, data -2.27e-13, total 1933.5882/1933.5882, lower tie
  ('merge', (1, 5), 0): d 3/3, cycles 90/90, data -1.14e-12, total 1934.3079/1934.3079, lower tie
  ('merge', (4, 7), 0): d 3/3, cycles 100/100, data -4.55e-13, total 1936.7444/1936.7444, lower tie
  ('merge', (2, 4), 0): d 3/3, cycles 100/100, data -4.55e-13, total 1937.5276/1937.5276, lower tie
  ('merge', (4, 6), 0): d 3/3, cycles 70/70, data +4.55e-13, total 1940.6479/1940.6479, lower tie
  ('merge', (1, 6), 0): d 3/3, cycles 70/70, data +4.55e-13, total 1941.9957/1941.9957, lower tie
  ('merge', (0, 4), 0): d 3/3, cycles 60/60, data -2.27e-13, total 1942.3468/1942.3468, lower tie
  ('merge', (1, 3), 0): d 3/3, cycles 50/50, data -4.55e-13, total 1945.6883/1945.6883, lower tie
  ('merge', (3, 5), 0): d 3/3, cycles 50/50, data -4.55e-13, total 1951.8343/1951.8343, lower tie
  ('merge', (2, 3), 0): d 3/3, cycles 50/50, data +2.27e-13, total 1951.9883/1951.9883, lower tie
  ('merge', (3, 7), 0): d 3/3, cycles 50/50, data +2.27e-13, total 1951.9883/1951.9883, lower tie
  ('merge', (3, 6), 0): d 3/3, cycles 40/40, data -1.14e-12, total 1956.0387/1956.0387, lower tie
  ('merge', (1, 2), 0): d 3/3, cycles 90/90, data -2.96e-12, total 1956.2125/1956.2125, lower tie
  ('merge', (1, 7), 0): d 3/3, cycles 80/80, data -2.27e-13, total 1956.2125/1956.2125, lower tie
  ('merge', (0, 3), 0): d 3/3, cycles 40/40, data -9.09e-13, total 1959.2497/1959.2497, lower tie
  ('merge', (0, 2), 0): d 3/3, cycles 40/40, data +1.14e-12, total 1959.5169/1959.5169, lower tie
  ('merge', (0, 5), 0): d 3/3, cycles 30/30, data -6.82e-13, total 1959.5169/1959.5169, lower tie
  ('merge', (0, 7), 0): d 3/3, cycles 40/40, data +1.59e-12, total 1959.5169/1959.5169, lower tie
  ('merge', (2, 5), 0): d 3/3, cycles 40/40, data -2.27e-13, total 1959.5169/1959.5169, lower tie
  ('merge', (5, 7), 0): d 3/3, cycles 40/40, data -6.82e-13, total 1959.5169/1959.5169, lower tie
  ('merge', (5, 6), 0): d 4/4, cycles 60/60, data -2.27e-12, total 1959.6421/1959.6421, lower tie
  ('merge', (2, 6), 0): d 3/3, cycles 50/50, data +9.09e-13, total 1979.1888/1979.1888, lower tie
  ('merge', (6, 7), 0): d 3/3, cycles 50/50, data -2.27e-13, total 1979.1888/1979.1888, lower tie
  ('merge', (1, 4), 0): d 3/3, cycles 100/100, data -2.27e-13, total 1979.7156/1979.7156, lower tie
  ('merge', (3, 4), 0): d 4/4, cycles 70/70, data +0, total 1984.0508/1984.0508, lower tie
  ('split', (7,), 0): d 5/5, cycles 400/400, data +4.55e-13, total 2054.6615/2054.6615, lower tie
  ('split', (7,), 1): d 5/5, cycles 440/440, data -3.18e-12, total 2054.6615/2054.6615, lower tie
  ('split', (0,), 0): d 5/5, cycles 450/450, data +1.36e-12, total 2097.7784/2097.7784, lower tie
  ('split', (0,), 1): d 5/5, cycles 450/450, data -1.59e-12, total 2104.7056/2104.7056, lower tie
  ('split', (4,), 1): d 5/5, cycles 470/470, data +1e-11, total 2111.8883/2111.8883, lower tie
  ('split', (4,), 0): d 5/5, cycles 490/490, data -1.82e-12, total 2113.8056/2113.8056, lower tie
  ('split', (5,), 1): d 6/6, cycles 490/490, data -3.18e-12, total 2114.1999/2114.1999, lower tie
  ('split', (5,), 0): d 6/6, cycles 450/450, data -4.55e-13, total 2114.4449/2114.4449, lower tie
  ('split', (2,), 0): d 6/6, cycles 350/350, data -2.27e-13, total 2131.2116/2131.2116, lower tie
  ('split', (2,), 1): d 6/6, cycles 350/350, data -4.55e-13, total 2131.2116/2131.2116, lower tie
  ('split', (1,), 1): d 5/5, cycles 410/410, data -5.23e-12, total 2131.4425/2131.4425, lower tie
  ('split', (3,), 1): d 5/5, cycles 420/420, data +3.41e-12, total 2132.2956/2132.2956, lower tie
  ('split', (3,), 0): d 5/5, cycles 430/430, data +6.82e-12, total 2135.3610/2135.3610, lower tie
  ('split', (6,), 1): d 5/5, cycles 420/420, data -3.87e-12, total 2135.4192/2135.4192, lower tie
  ('split', (1,), 0): d 5/5, cycles 410/410, data +9.09e-13, total 2136.0489/2136.0489, lower tie
  ('split', (6,), 0): d 5/5, cycles 400/400, data -9.09e-13, total 2150.0107/2150.0107, lower tie
  ('merge', (2, 7), 0): d 4/4, cycles 30/30, data +4.55e-13, total 2408.6601/2408.6601, lower tie
  cycles to converge, summed over candidates: cython 8500, torch 8500; lower total: cython 0, torch 0, tie 44; winners' totals 1932.7242/1932.7242, lower tie
  summary: d differs in 0, cycles in 0, max |data| 1e-11 bits, max |total| 0 bits; same ranking True, same winner True, cython winner's lead 0.0513 bits
path cython: 3 rounds in 2 s
path torch: 3 rounds in 802 s
  start: total 3187.9100/3187.9100, d 13/13, cycles 40/40
  round 0: ('split', (0,), 0)/('split', (0,), 0), total 2434.0131/2434.0131, d 10/10, cycles 200/200, lower tie, same move True
  round 1: ('split', (0,), 0)/('split', (0,), 0), total 2130.6592/2130.6592, d 4/4, cycles 200/200, lower tie, same move True
  round 2: ('split', (2,), 0)/('split', (2,), 0), total 1774.0328/1774.0328, d 3/3, cycles 200/200, lower tie, same move True
  first differing round: None; stop max_rounds/max_rounds; best total 1774.0328/1774.0328, lower tie
```

## What these numbers do not say

- **One corpus, small models.** `set02a_200` is 1,268 symbols over 6 symbols, and every
  model here has 1 to 8 states. The parallel phases are built for the opposite regime, and
  `viterbi_batch` measurements on the same L4 show `cuda` winning from `S = 64`.
- **Torch ran on the CPU**, because the private search takes no `device=`. A torch search on
  a GPU is therefore not measured here, and plumbing `device=` through the search waits on a
  model size where a GPU E-step beats `cython`.
- **`cuda` and `cpu_parallel` were not re-timed for this page.** The per-forward-pass figures
  quoted in the guidance table come from `total_forward_cost.py` as recorded in ADR 0024 §4.
- **A tie is not a proof.** Torch agreeing on 66 candidates bounds the perturbation at these
  sizes; it does not make torch bit-identical, and nothing should be built on the assumption
  that it is.
