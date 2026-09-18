# 0025. The topology search ships private at 0.3.0, and `__all__` stays at ten names

- **Status:** Accepted — 2026-09-18.
- **Date:** 2026-09-18
- **Source:** none in the PRD — postdates it. Raised on `feat/hmm-search-log` (PR #50) in
  revision `04-hmm-v0.3.0`, while asking where the search's training log should be
  documented. `docs/api/hmm/README.md`'s public-surface rule resolved that as *nowhere*,
  which is what surfaced the question this record answers.

## Context

Revision 04 ports topology search by state merge and split, and **no subgoal of it makes
the search public.** As planned, 0.3.0 ships what `core.md` and the PRD both name as the
revision's headline capability with `pfsmgraph.hmm.__all__` unchanged at the ten names it
has carried since `viterbi_batch`, and `_search`, `_trials`, `_topology` and `_mdl` all
private and out of contract.

That may well be right, but it fell out of the subgoal list rather than being chosen, and
two documents leaned the other way. One has since been turned: PR #50 gave
`docs/api/hmm/baum_welch.md` the paragraph stating that the search "is **private**: nothing
under `pfsmgraph.hmm` exports it, and the public surface is still exactly ten names". The
other is revision 04's own audit subgoal, which says the API check covers "whatever the
search loop and training log export" — a phrase that presumes a surface.

The decision is ordered **before the audit**, which checks `docs/api/` against what 0.3.0
actually ships, and **before the release**, since a published `__all__` cannot be narrowed
without a breaking change.

## Decision

### 1. `__all__` stays at ten names

`search` is not exported, and neither is any type reachable from it. The four modules stay
private.

### 2. Exporting is a one-way door and not exporting is not

`__all__` may **widen** at any later version at no cost to any consumer, and may never
narrow. So the two options are not symmetric in what they commit the project to, and the
asymmetry — not a judgement that the current surface is wrong — is what decides this.
Nothing here says the search should never be public. It says 0.3.0 is the wrong version to
promise it, because 0.3.0 is the version that cannot take the promise back.

### 3. Exporting the search would export five names, not one

`SearchResult` holds `TrialResult` twice and a `tuple[Round, ...]`; `Round.move` is a
`Move` whose `result` is a `TrialResult`. The whole frozen dataclass graph is reachable
from the return value, so `search`, `SearchResult`, `Round`, `Move` and `TrialResult`
would become contract together.

### 4. Publishing `TrialResult` would assert the two-part code, which `_mdl` refuses to assert

`TrialResult` carries `d` and `data_bits`. `_total_description_length` returns a bare
`float` **deliberately**, and its docstring gives the reason in as many words:

> a one-part code has no data/model split, so a return type carrying those two fields
> would assert the very structure that may be replaced

That is PRD §8's open research question, and the seam exists precisely so the criterion can
be swapped. Publishing `TrialResult` would re-assert in a frozen `__all__` exactly what the
criterion's seam was built to leave open — the same project contradicting itself one layer
up. `baum_welch` never faced this, which is why revision 03 could export its result on
terms revision 04 cannot: `BaumWelchResult.description_lengths` is a corpus description
length, and a corpus length survives a one-part code intact.

### 5. Three further promises sit on ADR 0023's own Open section

A public signature would freeze `split_min_cycles=200`, and that default **changes which
model comes back**: near the optimum a candidate's total is decided by where EM stopped
rather than by the criterion, moving up to 72 bits on `m008` across floors of 200, 400 and
800 while its unrounded data length moved by at most 2.4.
[ADR 0023](0023-topology-search-loop.md) records that the floor was never measured on
another corpus and that the floor needed depends on the model. It would also freeze §2's
`spawn_key` seeding contract as a consumer-visible promise, and the stop set
`Literal["max_rounds", "patience", "dead_end"]`, which a held-out score — explicitly
outside 0023's record — would have to widen.

### 6. Private means out of contract, not unreachable

`pfsmgraph.hmm._search` imports and runs. Nothing is hidden and nothing is prevented; a
researcher who wants the search at 0.3.0 can call it. What this record withholds is the
promise that it will keep working, which is the thing that cannot be withdrawn later.

### 7. The cost is named rather than mitigated

0.3.0 ships its headline capability with no supported way to call it. That is a real cost,
it is accepted rather than argued away, and 0.3.0's release notes should say so plainly
rather than leaving a reader to discover it from `__all__`.

## Consequences

### Positive

- The project keeps the ability to answer PRD §8 by substituting one function, which a
  published `TrialResult` would have removed.
- Revision 04's audit gets a smaller job: `docs/api/` is checked against ten names, the
  same ten 0.2.0 shipped, and the audit subgoal's "whatever the search loop and training
  log export" resolves to nothing.
- 0.4.0 inherits an open question rather than a frozen answer, which matters because
  `DEFERRED.md` leaves the alignment seed's own exposure undecided.
- No `docs/api/hmm/search.md` is owed at this release, and the search's log format does not
  become contract by being documented under [ADR 0013](0013-api-documentation-layout-and-tooling.md).

### Negative / costs

- A pip user installing `pfsmgraph-hmm` 0.3.0 gets a release whose advertised feature they
  cannot reach through the public API.
- Anyone who calls `_search` anyway is unprotected, and a future rename will break them
  silently. That is the intended meaning of private, but it is still a cost someone pays.
- The decision must be re-made later, and a deferral that is never revisited becomes a
  default by accident. `DEFERRED.md` carries the trigger for exactly this reason.

## Alternatives considered

1. **Export the search as it stands.** Rejected on §3 and §4 together: it is five names,
   and one of them contradicts a decision already written two modules away. It would also
   freeze §5's three unresolved promises at a version that cannot narrow them.
2. **Export a narrowed surface** — a public result carrying `params`, `total_bits`,
   `cycles` and `converged`, withholding `d` and `data_bits`, so the two-part structure
   stays unasserted. This answers §4 squarely and is the strongest alternative; it is
   rejected because it still freezes the signature, the seeding contract and the stop rules
   that §5 lists as unresolved, and because designing a second result type is work this
   revision did not budget and would be redesigning at 0.4.0 anyway. Worth re-proposing
   when the trigger fires.
3. **Defer without recording.** The status quo, and the thing this record exists to
   prevent: the next revision would rediscover the question rather than find its answer.

## Evidence

- `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_search.py:95-109` — `SearchResult`'s fields,
  from which the five-name reachability in §3 follows.
- `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_trials.py:80-94, 189-203` — `TrialResult`'s
  `d` and `data_bits`, and `Move.result`.
- `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_mdl.py:262-279` — the bare-`float` return and
  its stated reason, quoted in §4.
- [ADR 0023](0023-topology-search-loop.md), **Open** — the 72-bit step, the unmeasured
  floor, and the held-out score left out of the record.
- `docs/api/hmm/README.md`, *The public surface* — the rule that resolved the training
  log's documentation as *nowhere* and so raised this question.
- `docs/plan/DEFERRED.md`, `## Trigger: align able to produce a multiple alignment` — the
  alignment seed's exposure is itself undecided ("a caller who can still start merge/split
  from a single state may never need `align`"), so a signature frozen now would answer it
  by accident.

## Open

- **Whether the narrowed surface of alternative 2 is the right public shape**, if the
  search is ever exported. This record rejects it for 0.3.0 on timing, not on merit.
- **Whether `d` belongs in any public result at all.** It is the quantization parameter and
  genuinely useful to a caller, and it is also half of what makes the two-part structure
  visible. A one-part criterion would not have it to return.
