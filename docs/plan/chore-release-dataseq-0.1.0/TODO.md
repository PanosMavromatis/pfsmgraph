# chore/release-dataseq-0.1.0

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Release `pfsmgraph-dataseq` 0.1.0, replacing the `0.0.0` placeholder, and set honest lower bounds on the intra-family dependencies that name it (revision `01-dataseq-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Settle ADR 0003's sdist/wheel question
  > **Q:** Tests ship in the sdist without the policy that makes them honest (measurement
  > reproduced: sdist carries `tests/` but no `-ra` and no `pytest_report_header`; wheel
  > carries no tests at all). Ship the policy per member, stop shipping tests, or record
  > the policy as repo-local?
  > **A:** Repo-local, with an `align` trigger. Keep shipping tests; state in ADR 0003
  > that the mechanism does not travel in any sdist; file the obligation to revisit at
  > the first member that registers a real backend.
  > **Found:** the sdist also ships `.gitignore` and carries **no README and no LICENSE
  > file**, though `license = "MIT"` is declared. The PyPI page would be blank on first
  > publish. Added to goal 4 rather than folded in here.
  - [x] Re-read the measurement and the three candidate remedies as the record states them
  - [x] Put the choice to the user and log the answer inline
  - [x] Amend ADR 0003, moving the question out of `Open` per ADR 0010's convention
  - [x] Implement whichever remedy was chosen, or record explicitly that none was
  > **Done:** the chosen remedy is documentary, so there is no packaging change to make
  > and that is stated rather than left as an apparent omission. ADR 0003's sdist bullet
  > moved `Open` -> `Resolved`; `DEFERRED.md` closes the release-trigger entry and opens
  > a replacement under the `align` migration.

- [ ] Sweep the prose claims about repository state, semantically
  - [ ] `README.md` — the front page, and the surface with no other source of truth
  - [ ] `docs/agents/core.md` and `codex.md`, then regenerate the `AGENTS.*` artifacts
  - [ ] `docs/design/PRD.md` and each ADR's `Status`, plus `docs/design/adr/README.md`
  - [ ] `docs/api/dataseq/` — every executed code block still matching its pasted output

- [ ] Set honest version lower bounds
  - [ ] List every intra-family dependency naming `pfsmgraph-dataseq` and what it currently declares
  - [ ] Decide the bound `0.1.0` actually earns, and write it
  - [ ] Confirm the workspace source is not what makes it resolve

- [ ] Release: bump, build, publish, tag
  - [ ] Drop `.dev0` from `pfsmgraph-dataseq` alone, leaving the other four untouched
  - [ ] Add a README and a LICENSE file to `pfsmgraph-dataseq` and make the sdist carry them, not `.gitignore`
  - [ ] `uv build --package pfsmgraph-dataseq` and inspect both artifacts before anything leaves the machine
  - [ ] Publish to PyPI — the user runs this; it is irreversible
  - [ ] Tag `pfsmgraph-dataseq-v0.1.0` by hand and push the tag
  - [ ] Close the `DEFERRED.md` entries this branch discharged, leaving the recurring lower-bounds one open
