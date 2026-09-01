# MelodyHPO

Hyperparameter optimization for transformer and RNN models of melody. This research project prioritizes **interpretability and knowledge-level modeling** over raw generation quality.

## Status

**Phase 1: Local Proof of Concept** — establishing a working training loop locally before cloud migration.

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/PanosMavromatis/MelodyHPO.git
cd MelodyHPO

# Create environment and install dependencies
uv sync

# Install dev dependencies (pytest, ruff, mypy)
uv sync --extra dev

# Install all optional dependencies
uv sync --all-extras
```

## Development

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/unit/test_pitch_encoder.py

# Format and lint
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/

# Start TensorBoard
uv run tensorboard --logdir=experiments/
```

## Notebooks

Running JupyterLab through `uv` ensures the project's virtual environment is active, so
`melody_hpo` is importable without `sys.path` hacks:

```bash
uv run jupyter lab
```

## Project Structure

```
MelodyHPO/
├── src/
│   ├── components/    # Shared building blocks (attention variants, etc.)
│   ├── data/          # Shared data loading, tokenizer, preprocessing
│   │   └── encoder/   # String-to-integer encoders (pitch, etc.)
│   │       ├── control.py # Control-token mappings (PAD, BOS, EOS)
│   │       └── pitch.py   # PitchCode class: binomial encoder/decoder (ASA notation ↔ integer codes)
│   ├── generation/    # Shared autoregressive decoding, sampling strategies
│   ├── training/      # Shared training loop, loss computation, validation
│   ├── evaluation/    # Test-set evaluation, metrics, post-training analysis
│   ├── utils/         # Shared utilities (weight loading, checkpointing, plotting)
│   └── models/        # Model-specific code
│       └── transf01/  # Transformer variant 01 (config + model definition)
├── configs/           # YAML experiment configs
├── data/              # Local data copy (gitignored)
├── experiments/       # Experiment outputs (gitignored)
├── tests/             # Unit and integration tests
├── notebooks/         # Exploratory analysis
└── docs/decisions/    # Architectural Decision Records
```

### Pitch encoding

The `PitchCode` class (`src/data/encoder/pitch.py`) converts between ASA pitch strings and integer codes using a binomial encoding scheme via `PitchCode.encode()` and `PitchCode.decode()`. Each code packs both chromatic and diatonic information: the last two digits represent the diatonic code, and the leading digits represent the chromatic code (e.g., `C4` → `6035`). Codes start at 1207 (C0), leaving the 0–1206 range free for control tokens (`[PAD]`, `[BOS]`, `[EOS]`, etc.). The full MIDI range fits under 14,000, within a 16-bit integer.

### Shared vs. model-specific code

The directories directly under `src/` (`components/`, `data/`, `generation/`, `training/`, `evaluation/`, `utils/`) contain **shared modules** — code that is common across all models. Each model package under `src/models/` (e.g., `transf01/`) contains **only the code specific to that model**: its configuration and model definition. Shared infrastructure like data loading, training loops, attention building blocks, and decoding strategies live in the corresponding shared directories, so adding a new model means adding only a new subdirectory under `models/` with its config and architecture, reusing everything else.

## Reference Code (`LLMsFS/`)

The `LLMsFS/` directory contains reference code from Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch). It is **gitignored** by this repository and tracked independently at [github.com/PanosMavromatis/LLMsFS](https://github.com/PanosMavromatis/LLMsFS). Inside it, a read-only sparse checkout of the upstream book repository provides the `llms_from_scratch` package (chapters 02–07, plus Llama 3 and Qwen 3 implementations).

This code serves as the **architectural reference** for the transformer models developed in MelodyHPO. The core reference point is the `GPTModel` / `GPTModelFast` decoder-only transformer in chapter 04, along with its config-dict initialization pattern and supporting components (attention mechanisms, data loading, training loops, weight loading). When adapting patterns for this project, `GPTModelFast` — which uses `nn.LayerNorm`, `nn.GELU`, and `scaled_dot_product_attention` — is the preferred baseline.

**The reference code informs but does not fully align with this project's architecture.** The book organizes code by chapter, interleaving concerns across files (e.g., dataloaders in ch02, attention in ch03, model definition in ch04, training in ch05). MelodyHPO instead follows conventional practice — organizing by functional concern (`models/`, `data/`, `training/`, `evaluation/`, `utils/`), as production and research LLM codebases typically do. The `LLMsFS/` README includes a detailed chapter-to-concern mapping for reference.

The upstream code is **read-only** — it is never edited or committed from this project. See `LLMsFS/README.md` for setup instructions and update procedures.

## Documentation

- `CLAUDE.md` — AI assistant instructions and coding guidelines
- `PROJECT_INSTRUCTIONS.md` — Research philosophy and experiment structure
- `ARCHITECTURE_REFERENCE.md` — Technology stack and deployment architecture
- `TODO.md` — Current task list with status tracking
- `docs/environments/python-313-dependencies.md` — Python 3.13 environment package guide
- `docs/decisions/` — Architectural Decision Records (ADRs)

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.14+ |
| PyTorch | 2.10+ |
| Experiment tracking | TensorBoard |
| Package manager | uv |
| Linting/Formatting | ruff |
| Type checking | mypy |