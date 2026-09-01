# `segalign` against the `dl` base

The third `dataseq` container, read the way `hmm-lush/COMPARISON.md` reads the second. It
bears on the **container**; the encoder half of goal 4 is `align-poc/COMPARISON.md`, and the
reserved block for all four sources is `../RESERVED-BLOCK.md`. Kept separate because merging
the accounts would obscure which implementation supports which claim.

Two repositories are in scope, contributing asymmetrically. `segalign` (`ca97809`) is the
implementation. `SegAlign-Draft` (`9dc37b9`) is the predecessor it was refactored from, and
what it contributes is an **absence** — §5.

## 1. The axis: a container with no item access

`dl` and Lush are both dense vocabulary indices (`hmm-lush/COMPARISON.md` §1), and so is
`segalign` — but the comparison that matters here is not the encoder. It is that
`Dataset` **has `__len__` and no `__getitem__`** (`dataset.py:87-89`).

That single fact decides most of this document. The class is not indexable, not iterable
over items, and therefore not `torch.utils.data.Dataset`-compatible in the duck-typed sense
`ANALYSIS.md` §2.1 establishes is all that is required — which is a low bar it still does
not clear. What it is instead is a **corpus loader with derived views**: a list of
`pd.DataFrame` plus two lazily-computed parallel lists (`sequences`, `encoded_sequences`)
reached as whole-collection properties, never per item.

So where `dl` bakes the training objective into the container (`ANALYSIS.md` §2.2),
`segalign` does not bake in an objective at all — it simply has no item boundary to attach
one to. That is not restraint; it is a stage earlier.

## 2. Where the base must be overridden

Among the points the ADRs leave open, on the terms set at the top of the branch plan.

### 2.1 The focus-column indirection is the useful idea, and it generalises

`Dataset` holds DataFrames and derives sequences from **one named column** chosen at
construction (`focus_column`, default `'Pitch'`), with `Token`, `Boundary`, `Text` and
`Stress` all available from the same loaded data (`dataset.py:223`). Change the column
and the same corpus yields a different sequence alphabet.

This is the same separation `hmm-lush/COMPARISON.md` §4 arrives at from the other side —
the container holds the data, a *view* projects the sequence a consumer wants — and it is
the first of the three containers to express it as a first-class constructor parameter
rather than as an accident. `dl` has no analogue: its column is fixed at the point the
corpus is read.

**Carry the idea, not the mechanism.** The mechanism is a mutable public attribute plus a
hand-managed cache (§2.2). What generalises is that *which sequence you get* is a property
of the view, not of the container, and that several views coexist over one loaded corpus.

### 2.2 Lazy caching must not be invalidated by hand

`sequences` and `encoded_sequences` cache into `_sequences` / `_encoded_sequences`, and the
cache is cleared only by an explicit `invalidate_cache()` call (`dataset.py:184-201`). Every
input the cache depends on — `dframes`, `focus_column`, `toks_enc` — is a plain public
mutable attribute. Setting `focus_column` and reading `sequences` therefore returns the
*previous* column's data, silently and forever, unless the caller remembers a second call.

The class documents this rather than fixing it: `invalidate_cache`'s own docstring is a
worked example of the trap ("Change focus column / Clear stale cache"), and two tests pin
the behaviour (`test_invalidate_cache`, `test_cache_independence`). A documented footgun
with tests around it is still a footgun.

The merged container must make staleness unrepresentable rather than documented — a frozen
view, or derivation keyed on the inputs. This is the container-side counterpart of
`hmm-lush/COMPARISON.md` §2.4's "frozen must be an explicit state, not a structural
accident", and it is the second independent arrival at that conclusion.

### 2.3 Vocabulary ordering is frequency-descending — a third scheme, and the best of them

Codes are assigned from `Counter.most_common()` (`dataset.py:298`), so the most frequent
symbol gets the lowest user code. `dl` assigns by iterating a `set`; Lush by first
appearance.

`hmm-lush/COMPARISON.md` §2.1 makes ordering a **correctness** constraint and settles it as
first-appearance, on reproducibility grounds — `dl`'s `set` iteration is a live
reproducibility bug because CPython randomises `str` hashing per process. Measured against
that bar, `segalign` is *not* buggy: `most_common()` breaks ties by insertion order, and
insertion is driven by `sorted(tract_dirs)` and `sorted(verse_dirs)` (`dataset.py:266, 274`),
so the result is deterministic across processes.

It is therefore the only one of the three whose ordering is both deterministic and
*meaningful*, and it is worth recording why it still loses. Frequency ordering makes the
code of a symbol a function of the whole corpus, so adding one file can renumber the entire
alphabet — which is precisely what `hmm-lush/COMPARISON.md` §2.4 needs `frozen` to prevent,
and what makes a persisted vocabulary necessary rather than optional. First-appearance
ordering has the same determinism with a far weaker dependency. **Not a candidate to adopt;
a candidate to offer**, as an explicit reordering applied when a vocabulary is built, never
as the default assignment rule.

### 2.4 The container must not be the corpus reader

`from_directories` (`dataset.py:204-303`) is a classmethod that walks a package-data
directory tree, matches `V*` verse subdirectories, reads `All.csv` with `sep='\t'`, builds
the vocabulary and constructs the object — 100 lines in which corpus layout, file format,
vocabulary construction and container construction are one operation. `ANALYSIS.md` §2.5
reaches the same verdict about `MiniCorpus` from the `dl` side.

Two containers, independently, made the loader a constructor. That is the strongest evidence
in this comparison that `dataseq` must not: the ingestion path is the part that differs most
between the projects that will use it, and it is the part neither implementation could
reuse from the other.

It also fails quietly in a way worth naming. A missing base path returns an **empty
`Dataset`** rather than raising (`dataset.py:248-249`), and an unreadable CSV is caught and
printed as a `Warning:` to stdout before continuing (`dataset.py:289-290`). A typo'd corpus
path produces a valid, empty, silent result.

## 3. Divergences that are *not* candidates

### 3.1 The reserved block and strictness

`:EOS` 0, `:PAD` 1, user symbols from 2, and unknown symbols mapped to `:PAD` on purpose and
under test. Settled by ADR 0011 and tabulated in `../RESERVED-BLOCK.md` §1-2; the
unknown → `:PAD` collapse is the failure the separate `UNK`=1 exists to prevent, and this is
the only one of the four implementations that chose it deliberately rather than fell into it.

### 3.2 `toks` and `toks_enc` as two parallel public attributes

The vocabulary is a `list` and a `dict` the caller must keep consistent; nothing enforces
that `toks_enc` is `{tok: i for i, tok in enumerate(toks)}`, and the constructor accepts any
pair. `hmm-lush/COMPARISON.md` §2.5 and `ANALYSIS.md` §3.5 already require a vocabulary
*object*; this is a third instance of the same gap, not a new finding.

### 3.3 The decode direction is documented but absent

The class docstring says `toks_enc` is a "Dictionary mapping integer indices to token strings
for decoding" (`dataset.py:27`). It maps `str → int`, and nothing in the class decodes. This
is `ANALYSIS.md` §3.3 again, with the aggravating detail that here the docs claim otherwise —
worth recording only because a reader trusting the docstring would believe the gap is filled.

## 4. What `segalign` contributes that the others have no analogue for

**A tested container.** 422 lines across 22 tests (`tests/seq/test_dataset.py`), against
`ANALYSIS.md` §3.7's finding that `dl` has **no** container tests at all. Regardless of what
the merged container inherits in structure, this is the only one of the three that
demonstrates what a container test suite covers: empty and custom construction, the
length-mismatch invariant, missing columns, null values, unknown symbols, cache invalidation
and cache independence, and one test against real corpus data.

Those cases transfer even though the code does not, and they should seed the merged
container's suite. Two of them invert: `test_encoded_sequences_with_unknown_tokens` becomes
an assertion that encoding **raises**, and the two cache tests should become unnecessary
rather than rewritten, since §2.2 removes the state they exercise.

## 5. `SegAlign-Draft`, and the argument from absence

Tracked at a single file, deliberately. `ss2_alignment.py` is the Altschul & Erickson (1986)
all-optimal-alignments implementation, and its signature is the whole contribution:

```python
def ss2_alignment(seq1: List[Any], seq2: List[Any],
                  subst_cost: Callable = None,
                  gap_opening_cost: float = 1.0,
                  gap_extension_cost: float = 1.0) -> Dict:
```

There is no sequence type, no alphabet, no encoding step, and no `seq` submodule anywhere in
the source tree. Sequences are bare lists and symbol comparison is a caller-supplied
callable over raw objects.

The annotation is `List[Any]`, not `List[str]` — so the tree does not even *declare* that its
elements are strings; the docstring calls them "lists of musical tokens" and the string-ness
is a convention held in the caller's head. The absence is total, which makes the pair a
before/after: `SegAlign-Draft` aligns raw lists, `segalign` introduces `seq/` with a
`Dataset` and a `toks_enc`, and the refactoring between them is the moment a sequence
abstraction was first felt to be necessary in this lineage.

That is the evidentiary value, and it bears directly on the invariant in
`docs/agents/core.md`: **encode at the boundary**. `ss2_alignment` is the counter-example —
with `subst_cost` a callable over `Any`, the comparison of two symbols reaches into Python
object space in the innermost loop of an O(mn) dynamic program, which is exactly the
arrangement that makes a Cython or CUDA backend impossible to write mechanically. The
proof-of-concept's answer to the same problem is `ScoringMatrix.score(i, j)` on a dense
`float64` array (`align-poc/COMPARISON.md` §4). Nothing here is a candidate; it is the
before picture for a decision already made.

## 6. What this hands forward

| Goal | Inherits |
|---|---|
| 5 (land the container) | §2.1 the focus-column idea as a view, not a mutable attribute; §2.2 staleness must be unrepresentable; §2.4 the loader is not a constructor, and must not fail silently; §4 the 22 test cases as the container suite's seed |
| 6 (encoder API) | §2.3 frequency ordering offered as an explicit reordering, never the default; §3.2 the vocabulary object, third instance |
| 7 (promote ADR 0010) | §2.4 is the strongest evidence in this comparison — two containers independently made the corpus loader a constructor; §5 is the before picture for "encode at the boundary" |
| `align` 0.1.0 (later) | §5 — `ss2_alignment.py` is an all-optimal-alignments implementation with affine gaps, and belongs to that migration, not this one |
