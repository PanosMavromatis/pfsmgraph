# The Lush `dataseq` implementation, in its own terms

**Subgoal 2 of goal 3**, `docs/plan/feat-dataseq-merge/TODO.md`. This document describes
what the Lush code *is*, before any translation and without reference to the `dl`
(MelodyHPO) base. The comparison against that base is `COMPARISON.md`; the Python
transliteration is `translation/`. Keeping them apart is deliberate — an account written
with the merge in view tends to describe the original as a deviation from the thing it
predates by fifteen years.

**Sources.** `Code/SeqData/dsource-seq.lsh` (130 lines, 2009-07-15),
`Code/SeqData/seq-state.lsh` (80 lines, 2009-07-15), `Code/SeqData/format-sds.lsh`
(52 lines, 2009-07-05), plus the helpers `member-pos`, `add-to`, `not-eof`,
`mkdir-on-need` and `blank-str` from `Code/Utility/util.lsh`. Two corpora are tracked as
specimens: `Training/set01z0/set01z0_100.sds/` and
`Training/set11a_dInt/set11a_dInt.sds/`. Every quantitative claim below was measured
against those two, and the measurements are collected in the appendix.

**Provenance is code, not history.** Nothing here claims what was *run*. Where the code
admits a behaviour that may never have been exercised, it is marked **provenance
unknown** rather than asserted as a bug that bit.

---

## 1. Three parts, and the seam between them

The implementation is three files that do not form a single object:

| File | Kind | Role |
|---|---|---|
| `format-sds.lsh` | free function (`de`) | Builds a corpus: reads raw text, assigns codes, writes `.sds` |
| `dsource-seq.lsh` | class | Loads a built corpus; holds it; flattens it for the trainer |
| `seq-state.lsh` | class | One sequence *plus* the HMM's annotation of it |

**The seam is the important structural fact.** `format-sds` is not a method on
`dsource-seq` — it is a bare `de` that shares no code with the class, communicating with
it only through the on-disk format. So in this design **the vocabulary is a build-time
artefact, not a runtime object**. Encoding happens once, when a corpus is created, in a
process that afterwards exits. A loaded `dsource-seq` can *decode* but has no encoding
path at all.

Much of what follows is downstream of that seam.

## 2. `.sds` is a directory

The on-disk format is a directory, not a file, and nothing in the tree documents it — the
two writers and the one reader are its only specification.

```
<basename>.sds/
  _alphabet_size    one integer
  _alphabet         "<code>\t<symbol>" per line
  _size             one integer: number of sequences
  _seq_size_max     one integer: length of the longest sequence
  _seq_sizes        "<index>\t<size>" per line
  _raw_data         the input, kept alongside the output (see below)
  0.seq, 1.seq, …   one file per sequence, one integer code per line
```

`_raw_data` is the *input* to `format-sds`, and it stays in the directory beside the
output it produced. The format is Lush source text read with `(read)`: tokens separated
by any whitespace, `;` to end-of-line comments, `()` terminating each sequence, and
`|…|` bar-quoting for symbols containing spaces. Line structure is not significant —
`set11a_dInt`'s `_raw_data` uses one line per musical measure with a `; ---- m. |1|`
header above each, purely for reading by eye.

## 3. Vocabulary construction (`format-sds`)

```lisp
(let* ((alphabet '(begin end))
       (alphabet-size 2)
       (current-sequence-size 2)
       (seq-size-max 2))
  (while (<> (skip-char) "\e")            ; until EOF
    (setq current-symbol (read))
    (if current-symbol
        (progn
          (unless (member current-symbol alphabet)
            (add-to alphabet current-symbol)
            (incr alphabet-size))
          (add-to current-sequence (member-pos current-symbol alphabet))
          (incr current-sequence-size))
      … terminate the current sequence …)))
```

**The reserved block is two codes: `0 begin`, `1 end`. User symbols are numbered from 2.**
That is the whole of it — there is no padding code, no unknown code, no gap code, no mask.

**Codes are assigned in first-appearance order.** `add-to` appends
(`(setq lst (append lst (list elt)))`), and `member-pos` returns
`(- (length lst) (length sublst))`, a 0-based position. So the first novel symbol
encountered becomes 2, the next 3, and so on. The ordering is a property of the corpus
text, deterministic given the same input, and carries no meaning — `set11a_dInt`'s
alphabet begins `_`, `+4th`, `-2nd`, `+2nd`, `+3rd` because that is the order those
intervals happen to occur in the first measures.

**The vocabulary is unbounded and grows on demand.** Every symbol read is admitted. There
is no notion of a closed alphabet, so there is nothing an unknown symbol could fail
against — the concept does not exist here, rather than existing and being lenient.

**Both operations are linear scans over a list.** `member` and `member-pos` each walk the
alphabet, and `add-to` copies it to append. Building a corpus of *N* tokens over a
vocabulary of *V* symbols therefore costs O(N·V) in scanning plus O(V²) in appends, and
each sequence of length *L* costs O(L²) to accumulate the same way. For the sizes
involved — *V* = 4 and 25, *N* = 582 and 1447 — this is invisible, and it is the natural
thing to write in a list language. There is no hash table anywhere in the implementation.

**`begin` and `end` are added at serialization, not during accumulation.** The sequence
list holds only user codes; the writer emits them wrapped:

```lisp
(writing (open-write (concat basename (str i-seq) ".seq"))
  (print 0)
  (each ((current-symbol-n current-sequence)) (print current-symbol-n))
  (print 1))
```

`current-sequence-size` is initialised to 2 to account for the pair, so the recorded size
always matches the number of lines actually written. Both invariants hold in the
specimens: `set11a_dInt`'s single `.seq` is 1449 lines, opens with `0` and closes with
`1`, and `_seq_sizes` records `0	1449`.

**Sequence termination is a falsy read.** `(read)` returning `()` — the empty list in
`_raw_data` — closes the current sequence. A consequence worth naming: **if the file ends
without a final `()`, the last sequence is silently dropped**, because the accumulator is
only flushed on that branch and the loop exits on EOF. *Provenance unknown; the two
specimens both terminate properly.*

**Empty sequences are representable and present.** Two consecutive `()` yield a sequence
of size 2 containing only `begin` and `end`. This is not hypothetical: **35 of the 100
sequences in `set01z0_100.sds` are empty**, and `0.seq` is literally the two lines `0`
and `1`.

## 4. The container (`dsource-seq`)

```lisp
(defclass dsource-seq object
  ((-str-) name)
  ((-int-) alphabet-size)
  ((-idx1- (-gptr-)) alphabet)      ; array of C string pointers
  ((-int-) size)
  ((-int-) seq-size-max)
  ((-idx1- (-int-)) seq-sizes)
  ((-idx2- (-int-)) seq-data))      ; size x seq-size-max, dense
```

The constructor is `load`: `(new dsource-seq <basename>)` reads a `.sds` directory and
nothing else. There is no way to build one in memory.

**The alphabet is an array of raw C pointers.** Decoding is
`(ptr-str (alphabet code))` — an O(1) index into an array of `char *`. This is the
`-gptr-` type, chosen so the compiled `fprop-all` can carry the alphabet across the
Lush/C boundary. It also means the alphabet has no ownership story: it is aliased, never
copied, wherever it is passed.

**`seq-data` is dense and rectangular.** `load` allocates
`(int-matrix size seq-size-max)`, which is zero-filled, then writes only the first
`(seq-sizes i-seq)` cells of each row:

```lisp
(setq seq-data (int-matrix size seq-size-max))
(for (i-seq 0 (1- size))
     (reading (open-read (concat basename (str i-seq) ".seq"))
       (for (j-symb 0 (1- (seq-sizes i-seq)))
            (seq-data i-seq j-symb (read)))))
```

**So the unused tail of every short row holds `0`, which in this alphabet is `begin` — a
real symbol, not a marker for absence.** Padding and data are indistinguishable by
inspection of the matrix. The design is safe only because `seq-sizes` is consulted at
every read, and it is: `save` and `fprop-all` both loop to the row's own size rather than
the matrix width. But the correctness lives in every reader remembering to do that, not
in the representation.

The cost is not marginal. `set01z0_100.sds` holds 582 real entries in a 100×20 matrix —
**1418 of 2000 cells, 71%, are `begin`-valued padding.** The waste scales with
`size × (seq_size_max − mean)`, so it vanishes for `set11a_dInt`, which is a single
sequence of 1449 (a 1×1449 matrix, exactly full), and grows with corpus raggedness.

**Two loaders self-index from the file.** Both `_seq_sizes` and `_alphabet` are read as
(index, value) pairs and stored at the index the file names:

```lisp
(for (i 0 (1- size)) (seq-sizes (read) (read)))
```

The loop variable `i` is a counter, not the destination. Line order therefore does not
matter; a duplicated index silently overwrites; and a missing index silently leaves the
zero the matrix was allocated with — which for `seq-sizes` means a sequence of length 0,
and for the alphabet a null pointer. Nothing validates that the indices form a complete
range, and the consistency check the code intends is an unimplemented comment:
`; [ Will call the 'check consistency' script here. ]`.

**`save` is the near-inverse of `format-sds`, and differs on one line.** It writes the
same seven-file layout, and the `.seq` files round-trip exactly. But the two writers
spell `_alphabet` differently:

| Writer | Directive | Argument | `E. -2nd` written as |
|---|---|---|---|
| `format-sds` | `%l` | the symbol | `17	\|E. -2nd\|` |
| `dsource-seq save` | `%s` | `(ptr-str …)`, a string | `17	E. -2nd` *(inferred)* |

**The corpora on disk are fine, and the mechanism that makes them fine is worth stating
explicitly.** `format-sds` prints with `%l`, the Lisp representation, so any symbol
needing it is written with the multiple-escape delimiters `|…|`. The reader treats
everything between the bars as a single symbol name irrespective of its contents —
whitespace included — so `load`'s `(read)` recovers `E. -2nd` whole. `set11a_dInt`'s
`_alphabet` uses this for exactly two of its 25 entries, `17 |E. -2nd|` and `19 |H _|`,
and round-trips correctly. **Nothing about the tracked specimens is broken, and
whitespace in a symbol is not by itself a problem** — the escape mechanism is precisely
what handles it.

The open question is narrower, and concerns only output that **does not exist anywhere in
this tree**: a `_alphabet` written by `dsource-seq save` rather than by `format-sds`.
`save` prints `(ptr-str (alphabet i-symb))` — a string, the symbol's name with the
delimiters already stripped by `symbol->string` at load time — with `%s`. If `%s` emits
that string bare, the delimiters are not restored, and re-reading such a file with
`(read)` would take `E.` as the symbol and `-2nd` as the next index.

**That last step is an inference, not a measurement, and is flagged as such.** No
`save`-written `_alphabet` exists in the tree to check, and Lush is not available here to
run, so the claim rests on `%s` printing a string without re-escaping it — which is what
`%s` conventionally means, but was not verified. What *is* measured is only that the two
writers pass different things to different directives: a symbol to `%l` in one, a
`ptr-str` string to `%s` in the other.

**Why the two differ, which is not arbitrary.** Neither writer is compiled — `format-sds`
is a free `de` with no `dhc-make`, and `save` is explicitly marked "Not compilable". The
cause is one level down, in the *representation* compilation forced. The `alphabet` slot
is `-idx1- (-gptr-)`, an array of raw C pointers, because `fprop-all` and
`set-alphabet` are the two methods `dhc-make` compiles and the alphabet must cross that
boundary with them. That slot type obliges every class-side access to go through
`str-ptr` / `ptr-str`, and `load` line 83 — `(alphabet n (str-ptr (symbol->string s)))` —
is exactly where symbolhood is discarded. After it, the alphabet holds names, and nothing
records that any of them ever needed delimiters. `format-sds`, sitting outside the class
and never touching a `gptr`, keeps Lush symbols throughout and can still print with `%l`.

The file dates fit: `format-sds.lsh` is 2009-07-05 and `dsource-seq.lsh` 2009-07-15, so
the symbol-based writer is the earlier one, and the pointer-based representation arrived
with the class that had to be compiled. **The escaping asymmetry is therefore a
downstream cost of the compile boundary described in §7**, not an oversight in either
writer taken alone — which is also why it is a hazard worth carrying into the merge, where
the same boundary exists for the same reason.

*Provenance unknown* on the other half as well: whether `save` was ever run, and if so
whether on a corpus containing such a symbol, is not recoverable from the tree.

So this is carried forward as a **question for the merge, not a defect of the original** —
worth one regression test in the merged package (round-trip a symbol containing a space),
since a serializer and its parser disagreeing about escaping is a failure mode that
survives translation into any language.

## 5. `fprop-all` — the only consumer, and what it wants

```lisp
(defmethod dsource-seq fprop-all ()
  (let ((out-seq-state (new seq-state name ((idx-sum seq-sizes))))
        (current-pos 0))
    (idx-bloop ((current-size seq-sizes) (current-seq seq-data))
      (for (i 0 (1- (current-size)))
           (:out-seq-state:symbol-data current-pos (current-seq i))
           (incr current-pos)))
    (==> out-seq-state set-alphabet alphabet)
    out-seq-state))
```

It concatenates every sequence into **one flat stream** of `(idx-sum seq-sizes)` symbols,
stepping over the padding by looping to each row's own size. The `begin`/`end` codes stay
inline, and are the only thing marking where one sequence ends and the next begins.

`hmm-trainer.lsh:66` calls it once at setup — `(setq data-seq (==> data-source fprop-all))`
— and the trainer works on that `seq-state` alone. **It is the only caller of
`fprop-all` in the tree, and nothing else reads `seq-data`.**

That inverts how the dense matrix should be understood. It is not the container's
representation of the corpus so much as a staging buffer between the `.seq` files and the
flat stream: `load` unpacks ragged data into a rectangle, and its sole consumer
immediately repacks it into a ragged stream. The 71% zero-fill on `set01z0` buys nothing —
no code indexes `seq-data` by `[i][j]` for its own sake, and no code uses `seq-size-max`
except to allocate the matrix. The structure is genuinely there; the reason for it is not
in this tree.

Note also that `set-alphabet` **aliases**: the `seq-state` and the `dsource-seq` share one
`-gptr-` array. Freeing or reloading the source invalidates the derived state's alphabet.

## 6. `seq-state` — a sequence and its annotation

```lisp
(defclass seq-state object
  ((-str-) name)
  ((-int-) size)
  ((-idx1- (-gptr-)) alphabet)
  ((-idx1- (-int-)) symbol-data)     ; size
  ((-idx1- (-int-)) path-states)     ; size + 1
  ((-idx1- (-float-)) path-entropy)) ; size + 1
```

This is the design idea most worth carrying forward, and it is not a container idea at
all. `seq-state` holds the observed sequence **and the model's reading of it, in parallel
arrays, in one object**. The `+1` is exact rather than defensive: a path through an HMM
that emits *N* symbols visits *N+1* states, so `path-states` and `path-entropy` are
indexed by the *gaps between* symbols while `symbol-data` is indexed by the symbols
themselves.

`view-string` renders that offset directly, and is the clearest statement of the intent —
states on one line in brackets, symbols on another, interleaved by construction:

```
    a    b    a
 [0]  [2]  [1]  [0]
```

`resize` keeps all three arrays consistent (`midx-m1resize`, with `1+` on two of them);
`set-alphabet` is the aliasing setter described above. There is no decode method: rendering
goes through `(ptr-str (alphabet (symbol-data pos-i)))` inline in `view-string`.

## 7. What is compiled, and what that implies

`dhc-make` — the DH compiler, which emits the C found in `Code/SeqData/C/` — is applied to
exactly three methods:

```lisp
(dhc-make () (dsource-seq fprop-all))
(dhc-make () (seq-state seq-state set-alphabet))
```

The docstrings mark the rest explicitly: the constructor, `load` and `save` all say **"Not
compilable."** So the boundary is drawn between I/O and computation — anything touching a
file stays interpreted, and only the flattening step and the `seq-state` constructor cross
into C. The typed slots (`-int-`, `-idx1-`, `-gptr-`) and the inline type declarations
inside `fprop-all` exist to make that crossing possible.

This is the same principle the family now states as *encode at the boundary*, arrived at
under a different constraint: the DH compiler could not handle strings or file handles, so
strings had to become `gptr`s and integers before any compiled code saw them.

## 8. What the original does not have

Stated plainly, because the account's job is to describe the original rather than to
anticipate the merge:

- **No padding, unknown, gap or mask code.** Two reserved codes, both structural markers
  for sequence boundaries.
- **No runtime encoder.** Encoding exists only inside `format-sds`, at corpus-build time.
- **No decode method.** Decoding is an inline `ptr-str` index wherever it is needed.
- **No hash-based lookup.** Linear scan throughout.
- **No vocabulary object.** The alphabet is an array of pointers with a separately-stored
  size, reconstructed from a text file on every load.
- **No validation.** The intended consistency check is a comment; `alphabet-size`, `size`
  and the file contents are all trusted.
- **No train/test split, and no way to encode a second corpus against a first's
  vocabulary** — which is the same limitation seen from a different side, since sharing a
  vocabulary would require an encoder that outlives corpus construction.
- **No batching, no shuffling, no iteration protocol.** The trainer takes one flat stream.

---

## Appendix: measurements

Both specimens, measured directly rather than inferred.

| | `set01z0_100.sds` | `set11a_dInt.sds` |
|---|---|---|
| `_size` | 100 | 1 |
| `_alphabet_size` | 4 (`begin`, `end`, `a`, `b`) | 25 |
| `_seq_size_max` | 20 | 1449 |
| Total entries (Σ `_seq_sizes`) | 582 | 1449 |
| Dense matrix cells | 2000 | 1449 |
| Padding cells (`begin`-valued) | **1418 (71%)** | 0 |
| Empty sequences (size 2) | **35** | 0 |
| Bar-quoted alphabet entries | 0 | 2 (`\|E. -2nd\|`, `\|H _\|`) |

`set01z0` sequence-length distribution (including the `begin`/`end` pair): 35×2, 20×4,
16×6, 12×8, 4×10, 3×12, 4×14, 2×16, 2×18, 2×20.

The two specimens are near-opposite shapes — many short sequences over a toy alphabet
versus one long sequence over a real one — which is why both were tracked. Anything true
of the container has to hold for both.
