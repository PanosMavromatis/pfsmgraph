"""Dataset classes for melody sequence modeling.

Provides corpus loading, encoding, and dataset implementations for
symbolic melody data stored as tab-separated files. The module contains:

- :class:`MiniCorpus` — loads and encodes a filtered subset of melody files,
  caching symbol-to-code mappings for efficient lookup.
- :class:`DatasetSW` — sliding-window dataset for next-token prediction.
- :class:`DatasetDoc` — whole-document dataset for next-token prediction.
"""

import logging
import os
import re

import pandas as pd
import torch.utils.data

from melody_hpo.data.encoder.control import _control_code_decode, _control_sym_encode

logger = logging.getLogger(__name__)


class MiniCorpus:
    """Filtered collection of symbolic melody tab-separated files.

    Loads tab-separated files from a mini-corpus data root directory, selects
    columns according to a ``filters`` mapping, and keeps only rows where **every**
    column matches its own regex pattern. Each surviving row is joined by tab
    characters to form a symbol string. Symbols are encoded to integer codes
    via the supplied ``encoder`` and wrapped with BOS / EOS control tokens.

    Encoding results are stored in ``encoder_map`` / ``decoder_map`` so
    that ``encoder.encode()`` is called at most once per unique symbol across
    all documents. The ``alphabet`` set tracks which symbols have already been
    mapped.

    Args:
        minicorp_def: A dictionary defining the mini-corpus with the following keys:

            - **data_dir** (*str*): The mini-corpus data root directory.
            - **doc_paths** (*list[str]*): Directory paths relative to *data_dir*,
              each containing a tab-separated file named *df_name*.
            - **df_name** (*str*): The filename of the tab-separated file to load
              within each path directory.
            - **filters** (*dict[str, str]*): Column names mapped to regex patterns.
              Only columns present as keys are kept; only rows where every column
              matches its regex are retained.
            - **encoder** (:class:`~melody_hpo.data.encoder.music.MusicCode`):
              Encoder used to convert each filtered row into a single integer code.

    Attributes:
        data: A ``dict[str, pd.Series]`` keyed by relative path. Each Series
            contains the encoded integer sequence (BOS + encoded tokens + EOS)
            for that file.
        max_doc_length: The length of the longest encoded sequence.
        alphabet: The set of all symbol strings seen so far (control symbols
            plus every unique data symbol across all loaded documents).

    Example::

        layer01 = {
            "data_dir": "data/MelodyData/content/Chant/Gregorian/Tractus",
            "doc_paths": ["Mode02/DeusMeus/V01"],
            "df_name": "All.csv",
            "filters": {"Token": r"^[A-G]"},
            "encoder": my_encoder,
        }
        mc = MiniCorpus(minicorp_def=layer01)
        print(mc.data["Mode02/DeusMeus/V01"])
    """

    def __init__(self, minicorp_def: dict) -> None:
        data_dir = minicorp_def["data_dir"]
        doc_paths = minicorp_def["doc_paths"]
        df_name = minicorp_def["df_name"]
        filters = minicorp_def["filters"]
        encoder = minicorp_def["encoder"]

        # Encoding maps: seeded with control symbols and grown as new data
        # symbols are encountered. The three structures are kept in sync —
        # alphabet tracks which symbols have been mapped so that
        # encoder.encode() is called at most once per unique symbol.
        self.encoder_map: dict[str, int] = _control_sym_encode.copy()
        self.decoder_map: dict[int, str] = _control_code_decode.copy()
        self.alphabet: set[str] = set(_control_sym_encode.keys())

        self.doc_lengths: dict[str, int] = {}
        self.max_doc_length = 0

        # Pre-compile one regex per column
        compiled = {col: re.compile(pat) for col, pat in filters.items()}
        columns = list(filters.keys())

        # Initialize data store with one DataFrame per path
        self.data: dict[str, pd.Series] = {}

        for path in doc_paths:
            df_source_path = path + '/' + df_name
            filepath = os.path.join(data_dir, df_source_path)
            logger.info("Loading %s", filepath)

            df = pd.read_csv(filepath, sep="\t")

            # Keep only the requested columns
            df = df[columns]

            # Filter rows: keep only those where each column matches its own pattern
            # Boolean Series matching df's length, all True; the loop below ANDs each filter in
            mask = pd.Series(True, index=df.index)
            for col, regex in compiled.items():
                # Cast to str, test each value against the regex (NaN → False),
                # then AND into the mask so only rows matching every filter survive
                mask &= df[col].astype(str).str.contains(regex, na=False)
            # Keep only matching rows and renumber from 0 (drop=True discards the old index)
            df = df[mask].reset_index(drop=True)

            # Encode each row into a single integer code.
            # df.itertuples(index=False) yields each row as a named tuple of cell
            # values (without the row index). We join the values with tab characters
            # to produce a single string per row (e.g. "C4\tq"), which the encoder
            # receives as its input.
            symbols = pd.Series(
                ["\t".join(row) for row in df.itertuples(index=False)]
            )
            # Prepend BOS and append EOS
            symbols = pd.concat([pd.Series(['BOS']), symbols, pd.Series(['EOS'])], ignore_index=True)

            # Map any symbols not yet seen. Only genuinely new symbols
            # trigger encoder.encode(); previously seen ones (including BOS/EOS
            # after the first document) are served from the existing maps.
            symbol_set = set(symbols.tolist())
            new_symbol_set = symbol_set - self.alphabet
            for new_symbol in new_symbol_set:
                new_symbol_code = encoder.encode(new_symbol)
                self.encoder_map[new_symbol] = new_symbol_code
                self.decoder_map[new_symbol_code] = new_symbol
            self.alphabet.update(new_symbol_set)

            # Apply integer codes to the symbolic sequence
            codes = symbols.map(self.encoder_map)

            # Register the document's length and update
            # the mini-corpus's max document length
            doc_length = len(codes)
            self.doc_lengths[path] = doc_length
            if self.max_doc_length < doc_length:
                self.max_doc_length = doc_length

            self.data[path] = codes
            logger.info("Loaded %s: %d rows after filtering", path, len(df))


class DatasetSW(torch.utils.data.Dataset):
    """Sliding-window dataset (SW) over a :class:`MiniCorpus`.

    Document selection is delegated to :class:`MiniCorpus`. This class
    implements the sliding-window logic and exposes fixed-length windows
    for next-token prediction training.

    Args:
        mini_corpus: A :class:`MiniCorpus` instance to draw documents from.
        window_size: The length of each input window.
        stride: Step size between consecutive windows. Must not exceed
            *window_size*, or tokens between windows will be lost.

    Each window produces an ``(input_ids, target_ids)`` pair where
    ``target_ids`` is shifted one position ahead of ``input_ids``.
    Documents shorter than ``window_size + 1`` tokens are zero-padded to
    fit a single window. All tensors are ``torch.long``.
    """

    def __init__(self, mini_corpus: MiniCorpus, window_size: int, stride: int) -> None:
        if stride > window_size:
            raise ValueError(
                f"stride ({stride}) must not exceed window_size ({window_size}), "
                f"or notes between windows will be lost"
            )

        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # Iterate over documents in the minicorpus
        for key, series in mini_corpus.data.items():
            token_ids = series.tolist()

            if len(token_ids) < window_size + 1:
                logger.warning(
                    "Document %s has %d tokens, fewer than window_size + 1 (%d); "
                    "padding to fit a single window",
                    key,
                    len(token_ids),
                    window_size + 1,
                )
                pad_len = window_size - len(token_ids)
                input_chunk = token_ids + [0] * pad_len
                target_chunk = token_ids[1:] + [0] * (pad_len + 1)
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
                continue

            for i in range(0, len(token_ids) - window_size, stride):
                input_chunk = token_ids[i : i + window_size]
                target_chunk = token_ids[i + 1 : i + window_size + 1]
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


class DatasetDoc(torch.utils.data.Dataset):
    """Document-based dataset (Doc) over a :class:`MiniCorpus`.

    Document selection is delegated to :class:`MiniCorpus`. This class
    pads every document to the corpus maximum length and exposes
    fixed-length sequences for next-token prediction training.

    Args:
        mini_corpus: A :class:`MiniCorpus` instance to draw documents from.

    Each document produces an ``(input_ids, target_ids)`` pair where
    ``target_ids`` is shifted one position ahead of ``input_ids``.
    Documents shorter than ``max_doc_length`` tokens are zero-padded.
    All tensors are ``torch.long``.
    """

    def __init__(self, mini_corpus: MiniCorpus) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        max_doc_length = mini_corpus.max_doc_length

        # Iterate over documents in the minicorpus
        for series in mini_corpus.data.values():
            token_ids = series.tolist()

            pad_len = max_doc_length - len(token_ids)
            input_chunk = token_ids + [0] * pad_len
            target_chunk = token_ids[1:] + [0] * (pad_len + 1)
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]
