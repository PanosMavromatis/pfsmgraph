# chore/release-dataseq-0.1.0

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Release `pfsmgraph-dataseq` 0.1.0, replacing the `0.0.0` placeholder, and set honest lower bounds on the intra-family dependencies that name it (revision `01-dataseq-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Settle ADR 0003's sdist/wheel question
  - [ ] Re-read the measurement and the three candidate remedies as the record states them
  - [ ] Put the choice to the user and log the answer inline
  - [ ] Amend ADR 0003, moving the question out of `Open` per ADR 0010's convention
  - [ ] Implement whichever remedy was chosen, or record explicitly that none was

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
  - [ ] `uv build --package pfsmgraph-dataseq` and inspect both artifacts before anything leaves the machine
  - [ ] Publish to PyPI — the user runs this; it is irreversible
  - [ ] Tag `pfsmgraph-dataseq-v0.1.0` by hand and push the tag
  - [ ] Close the `DEFERRED.md` entries this branch discharged, leaving the recurring lower-bounds one open
