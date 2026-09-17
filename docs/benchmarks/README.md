# Benchmarks

**Nothing here is a contract.** These pages record measurements taken on one host on one
day, with the versions of everything that were installed then. They go stale by design, and
a number that disagrees with a fresh run on your machine is not a defect in either.

What is meant to survive is the shape of each comparison — which backend wins where, by
roughly how much, and why — and the script that produced it, so the measurement can be
re-run rather than believed.

Where the durable statements live instead:

- [`docs/api/`](../api/) is normative for contracts: what a call promises, and which
  backends return identical results. Every code block there is executed and its output
  pasted ([ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md)), so
  timings cannot live there — they would fail on the next machine.
- [`docs/design/adr/`](../design/adr/) records decisions, each with the evidence that
  decided it.
- `.scratch/hmm-lush/measurements/` holds the scripts. Each page below names the script it
  ran and the SHA-256 of that script's contents; the page is stale when the hash no longer
  matches, as a `FORMALIZATION.md` is stale when its kernel's hash moves.

## Pages

- [`hmm-backends.md`](hmm-backends.md) — choosing among `pfsmgraph.hmm`'s backends for
  training and topology search: what each phase promises, and what two measurements found.
