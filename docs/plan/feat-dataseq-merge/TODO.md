# feat/dataseq-merge

**Status**: active
**Created**: 2026-08-31
**Subgoal**: revision `01-dataseq-v0.1.0`, subgoals 1 (merge the three implementations)
and 2 (settle the encoder API and promote ADR 0010)

## Tasks

- [ ] Import the three implementations and tabulate their divergences
  - [ ] All three are readable in-tree (or in a scratch location) and their provenance is recorded
  - [ ] A written comparison of container semantics, encoder shape, and vocabulary handling
  - [ ] Every point where the `dl` base must be overridden by another implementation is named, with why

- [ ] Land the merged container in `packages/pfsmgraph-dataseq/`
  - [ ] `dl` version is the base; divergences resolved per the comparison above
  - [ ] Conforms to `torch.utils.data.Dataset`; a stock `DataLoader` works without subclassing
  - [ ] No `pfsmgraph/__init__.py` introduced anywhere; PEP 420 namespace intact
  - [ ] `uv sync && uv run python -c "import pfsmgraph.dataseq"` succeeds, and the other four still import

- [ ] Settle and implement the encoder API
  - [ ] Constructor signature decided and recorded inline as Q&A
  - [ ] Spelling of the strictness switch decided; strict is the default, `UNK` fallback explicit opt-in
  - [ ] How `align` consumes the mapping at its boundary decided — encode-at-the-boundary must stay mechanical for Cython/CUDA
  - [ ] Reserved block hard-coded per ADR 0011: `PAD`=0 … `MSK`=5, user symbols from 6, not configurable
  - [ ] Encoder and decoder implemented; unseen symbols raise by default

- [ ] Promote ADR 0010 to `Accepted`
  - [ ] Record the settled API in the ADR's decision section
  - [ ] Status changed `Proposed` → `Accepted` with the date
  - [ ] Row updated in `docs/design/adr/README.md`
