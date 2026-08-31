# Mini-corpus definition: Gregorian Tractus, Mode 8, Layer 01.
#
# This file is not imported as a regular module. It is executed via
# runpy.run_path() from notebooks (see notebooks/explore/datasets.ipynb,
# cells [3]–[4]). The call returns all module-level variables as a dict,
# which is then passed to MiniCorpus(minicorp_def=...).

from pathlib import Path

import pandas as pd

from melody_hpo.data.encoder import PitchCode
from melody_hpo.paths import DATA_ROOT

# Derive the corpus group path (e.g. "Chant/Gregorian/Tractus/Mode08/")
# from this file's location under minicorps/, so the definition stays
# location-aware without hardcoding.
_parts = Path(__file__).resolve().parent.parts
minicorp_groups_root = "/".join(_parts[_parts.index("minicorps") + 1:]) + "/"

# Absolute path to the data directory where the CSVs for this group live.
data_dir = str(DATA_ROOT / "MelodyData/content" / minicorp_groups_root)

# Build doc_paths from groups.csv: select chants belonging to this layer's
# group and generate a sub-path for each of their sections (V01, V02, …).
group_label = 'L1'

_groups = pd.read_csv(Path(data_dir) / "groups.csv")
_layer = _groups[_groups["Group"] == group_label]

doc_paths: list[str] = [
    f"{row['Chant Name']}/V{s:02d}"
    for _, row in _layer.iterrows()
    for s in range(1, row["Sections"] + 1)
]


# CSV filename to load from each doc_path directory.
df_name = 'All.csv'

# Column-level regex filters applied when loading data.
# Only rows where the "Pitch" column matches PitchCode.pattern are kept.
filters = {
    "Pitch": PitchCode.pattern
}

# Encoder instance used by MiniCorpus to map pitch strings to integer codes.
encoder = PitchCode()
