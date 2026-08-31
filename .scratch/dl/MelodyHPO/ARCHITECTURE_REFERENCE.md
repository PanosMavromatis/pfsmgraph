# Architecture Reference: Hyperparameter Optimization for Music Models

**Document Purpose:** Reference guide for technology stack and architectural decisions for HPO experiments on melody models. Intended for project memory and future conversations.

**Last Updated:** January 2026

---

## Executive Summary

This document captures architectural decisions for a research project on hyperparameter optimization for AI models of melody. The project prioritizes **interpretability and knowledge-level modeling** over raw performance, using small, curated, stylistically coherent corpora.

### Key Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Local environment | Native Python with `uv` | Fast iteration, no container overhead for PoC |
| Cloud platform | Vertex AI Training | Managed infrastructure, native HPO via Vizier |
| Container strategy | None locally → single multi-platform Dockerfile for cloud | Simplicity now, consistency when scaling |
| Experiment tracking | TensorBoard (Vertex AI managed) | Embedding projector for interpretability; W&B as future option |
| Data versioning | Git (current) → SQLite catalog (future) | Pragmatic for small corpus; scales to metadata queries |
| Code/data coupling | Decoupled repositories, path configuration | Flexibility to run different models on same data |

---

## Development Environment

### Local Setup (M1 Mac)

**Hardware:** MacBook Pro, M1 Max, 64GB RAM

**Python Environment:**
- Use `uv` for dependency management
- No containers for local development (faster iteration)
- MPS backend available for modest GPU acceleration if needed

**Why no local containers?**
- Proof-of-concept phase prioritizes speed
- Small dataset (tens of MB) loads instantly
- Container overhead unnecessary until cloud migration

**Recommended local workflow:**
```bash
# Project setup with uv
uv init melody-hpo
cd melody-hpo
uv add torch tensorboard pandas

# Activate environment
source .venv/bin/activate

# Run experiments
python train.py --config configs/local_test.yaml
```

### Dependency Pinning

Maintain a `pyproject.toml` with pinned versions for reproducibility:

```toml
[project]
name = "melody-hpo"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.10",
    "tensorboard>=2.20",
    "pandas>=3.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
    "mypy>=1.13",
]
```

When preparing for cloud deployment, export frozen requirements:
```bash
uv pip compile pyproject.toml -o requirements.txt
```

---

## Cloud Platform: Vertex AI

### Why Vertex AI?

- **Managed compute:** No raw GCE instance management
- **Native HPO:** Vizier integration for Bayesian optimization
- **Experiment tracking:** Built-in metrics logging, TensorBoard integration
- **Flexibility:** Custom containers, custom training code

### Recommended GPU Targets

For small datasets and simple transformer/RNN models:

| GPU | Use Case | Cost Tier |
|-----|----------|-----------|
| T4 | Development, small experiments | Low |
| L4 | Moderate training, HPO sweeps | Medium |
| A100 (40GB) | Only if scaling significantly | High |

**Start with T4** for initial cloud experiments. The dataset size (tens of MB) and model complexity (simple transformers, GRUs) don't require high-end hardware.

### Vertex AI Components

```
┌─────────────────────────────────────────────────────────┐
│                     Vertex AI                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Training   │  │   Vizier    │  │   TensorBoard   │ │
│  │   Jobs      │  │    (HPO)    │  │    (Managed)    │ │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         │                │                   │          │
│         └────────────────┼───────────────────┘          │
│                          │                              │
│                    ┌─────▼─────┐                        │
│                    │    GCS    │                        │
│                    │  Bucket   │                        │
│                    └───────────┘                        │
│                    (data, checkpoints, logs)            │
└─────────────────────────────────────────────────────────┘
```

### Getting Started with Vertex AI

**Prerequisites:**
1. GCP project with billing enabled
2. Vertex AI API enabled
3. Service account with appropriate roles
4. `gcloud` CLI installed and configured

**Initial setup commands:**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create service account (if needed)
gcloud iam service-accounts create melody-hpo-sa \
    --display-name="MelodyHPO Service Account"

# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:melody-hpo-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

---

## Container Strategy

### Phase 1: Local Development (Current)

No containers. Run directly in `uv`-managed virtual environment.

### Phase 2: Cloud Deployment

Single multi-platform Dockerfile supporting both local testing and GCP deployment.

**Directory structure:**
```
docker/
├── Dockerfile           # Multi-platform build
├── .dockerignore
└── entrypoint.sh        # Training script wrapper
```

**Dockerfile template:**
```dockerfile
# Multi-platform Dockerfile for melody HPO
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.4-cudnn9-runtime

FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY src/ ./src/
COPY configs/ ./configs/

# Entry point
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
```

**Build commands:**
```bash
# For GCP (AMD64 with CUDA)
docker build -t gcr.io/YOUR_PROJECT/melody-hpo:latest \
    --platform linux/amd64 \
    -f docker/Dockerfile .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT/melody-hpo:latest

# For local testing on M1 (optional, CPU-only)
docker build -t melody-hpo:local \
    --platform linux/arm64 \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.10.0-cpu \
    -f docker/Dockerfile .
```

---

## Data Management

### Current State: Hierarchical CSV Structure

```
data/
├── Gregorian/
│   └── Tract/
│       └── Mode8/
│           └── SicutCervus/
│               └── Melody.csv
├── Medieval/
│   └── ...
└── Renaissance/
    └── ...
```

**Characteristics:**
- Metadata embedded in directory structure
- Each document is a single CSV file
- Total size: tens of MB
- Resides in separate GitHub repository

### Versioning Strategy

**Current (small corpus):** Git directly

- Simple, familiar workflow
- Adequate for infrequently changing data
- Track data repo commit hash in experiment logs

**Future (hundreds of works):** SQLite catalog with flattened structure

```
data/
├── catalog.db           # SQLite: uuid, title, composer, date, mode, genre, path
├── documents/
│   ├── 550e8400-e29b-41d4-a716-446655440000.csv
│   ├── 6ba7b810-9dad-11d1-80b4-00c04fd430c8.csv
│   └── ...
└── schema.sql           # Catalog schema definition
```

**Catalog schema:**
```sql
CREATE TABLE works (
    uuid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    composer TEXT,
    date_composed INTEGER,  -- Year, nullable
    mode TEXT,
    genre TEXT,
    tradition TEXT,         -- Gregorian, Medieval, Renaissance, etc.
    source TEXT,            -- Original database/collection
    filepath TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_date ON works(date_composed);
CREATE INDEX idx_mode ON works(mode);
CREATE INDEX idx_tradition ON works(tradition);
```

**Query examples:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/catalog.db')

# All works before 1500
pre_1500 = pd.read_sql(
    "SELECT * FROM works WHERE date_composed < 1500",
    conn
)

# All Mode 8 Gregorian tracts
mode8_tracts = pd.read_sql("""
    SELECT * FROM works 
    WHERE mode = '8' AND tradition = 'Gregorian' AND genre = 'Tract'
""", conn)
```

### Repository Relationship

**Architecture:** Decoupled repositories with path configuration

```
~/projects/
├── melody-data/         # Data repository (GitHub)
│   ├── data/
│   └── catalog.db       # (future)
│
└── MelodyHPO/           # Training repository (GitHub)
    ├── src/
    ├── configs/
    └── experiments/
```

**Configuration approach:**
```yaml
# configs/base.yaml
data:
  root: ${DATA_ROOT:/home/user/projects/melody-data}
  catalog: ${DATA_ROOT}/catalog.db  # future
  
# Environment variable or CLI override
# DATA_ROOT=/path/to/data python train.py
```

**Cloud data access:**
1. Sync data repo to GCS bucket (one-time or periodic)
2. Training jobs mount/download from GCS
3. Record data commit hash in experiment metadata

```bash
# Sync data to GCS
gsutil -m rsync -r ./melody-data gs://YOUR_BUCKET/data/

# In training job, data path becomes:
# gs://YOUR_BUCKET/data/ or /gcs/YOUR_BUCKET/data/ (mounted)
```

---

## Experiment Tracking: TensorBoard

### Why TensorBoard?

1. **Embedding Projector:** Visualize learned representations (pitch embeddings, rhythm encodings) with PCA/t-SNE/UMAP—directly supports interpretability goals
2. **Vertex AI Integration:** Managed TensorBoard instances with GCS log storage
3. **Mature PyTorch Support:** `torch.utils.tensorboard` is stable and well-documented
4. **No Additional Vendor:** Stays within GCP ecosystem

### What to Log

| Category | What to Log | TensorBoard Feature |
|----------|-------------|---------------------|
| Training progress | Loss, learning rate, gradient norms | Scalars |
| Model internals | Weight distributions, activation stats | Histograms |
| Learned representations | Pitch/rhythm embeddings | Embedding Projector |
| Attention patterns | Attention heatmaps | Images |
| Architecture | Model graph | Graphs |
| Hyperparameters | Full config | HParams |

### Implementation Pattern

```python
from torch.utils.tensorboard import SummaryWriter
import torch

class ExperimentLogger:
    def __init__(self, log_dir: str, config: dict):
        self.writer = SummaryWriter(log_dir)
        self.config = config
        
        # Log hyperparameters at start
        self.writer.add_hparams(
            hparam_dict=self._flatten_config(config),
            metric_dict={},  # Filled at end
            run_name='.'
        )
    
    def log_scalars(self, step: int, **kwargs):
        """Log scalar metrics."""
        for name, value in kwargs.items():
            self.writer.add_scalar(name, value, step)
    
    def log_embeddings(self, tag: str, embeddings: torch.Tensor, 
                       metadata: list[str], step: int):
        """Log embeddings for projector visualization."""
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            tag=tag,
            global_step=step
        )
    
    def log_attention(self, tag: str, attention: torch.Tensor, step: int):
        """Log attention heatmap as image."""
        # Normalize to [0, 1] for visualization
        attn_normalized = (attention - attention.min()) / (attention.max() - attention.min())
        self.writer.add_image(tag, attn_normalized, step, dataformats='HW')
    
    def log_histograms(self, model: torch.nn.Module, step: int):
        """Log weight and gradient histograms."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def close(self):
        self.writer.close()
```

### Vertex AI Managed TensorBoard

```python
from google.cloud import aiplatform

# Create managed TensorBoard instance (one-time)
tensorboard = aiplatform.Tensorboard.create(
    display_name="melody-hpo-tensorboard",
    project="YOUR_PROJECT",
    location="us-central1"
)

# When submitting training job, link to TensorBoard
job = aiplatform.CustomJob.run(
    display_name="gru_bach-chor_rhythm_20250120",
    script_path="train.py",
    container_uri="gcr.io/YOUR_PROJECT/melody-hpo:latest",
    tensorboard=tensorboard.resource_name,
    # ... other params
)
```

### Future Option: Weights & Biases

If TensorBoard's comparison UI proves limiting during intensive HPO sweeps, W&B can supplement:

- Superior run comparison (parallel coordinates, custom tables)
- Hyperparameter importance analysis
- Free tier sufficient for individual research

**When to consider adding W&B:**
- Running 50+ HPO trials and need better comparison tools
- Want automated hyperparameter importance plots
- Need to share results with collaborators outside GCP

---

## Storage Strategy

### GCS Bucket Structure

```
gs://YOUR_BUCKET/
├── data/                    # Synced from data repo
│   ├── documents/
│   └── catalog.db
│
├── experiments/             # Training outputs
│   └── {experiment-name}/
│       ├── checkpoints/
│       │   ├── epoch_010.pt
│       │   ├── epoch_020.pt
│       │   └── best_model.pt
│       ├── logs/
│       │   └── events.out.tfevents.*
│       └── outputs/
│           └── evaluation_results.json
│
└── tensorboard/             # Managed TensorBoard logs
    └── {tensorboard-instance}/
```

### Checkpoint Strategy

**During training:**
- Save every N epochs (configurable, e.g., every 10)
- Always save "best so far" based on validation metric
- Include optimizer state for resumption

**After training:**
- Keep only top K models (e.g., 3-5) based on final evaluation
- Delete intermediate checkpoints
- Archive final models with full metadata

**Checkpoint contents:**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_metric': val_metric,
    'config': config,
    'data_commit': data_commit_hash,  # Reproducibility
    'random_state': {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
}
torch.save(checkpoint, path)
```

### Cost Management

**GCS storage classes:**

| Class | Use Case | Cost |
|-------|----------|------|
| Standard | Active experiments, recent checkpoints | Higher |
| Nearline | Completed experiments (access < 1x/month) | Lower |
| Coldline | Archived models (access < 1x/quarter) | Lowest |

**Lifecycle policy example:**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30, "matchesPrefix": ["experiments/"]}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 180, "matchesPrefix": ["experiments/*/checkpoints/epoch_"]}
      }
    ]
  }
}
```

---

## Workflow Summary

### Phase 1: Local Proof of Concept

```
┌─────────────────────────────────────────────────────────┐
│                   Local Machine (M1 Mac)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐    ┌──────────────┐                 │
│   │  MelodyHPO   │    │  melody-data │                 │
│   │    (code)    │───▶│    (data)    │                 │
│   └──────┬───────┘    └──────────────┘                 │
│          │                                              │
│          ▼                                              │
│   ┌──────────────┐                                     │
│   │ uv venv      │                                     │
│   │ Python 3.14  │                                     │
│   │ PyTorch 2.10 │                                     │
│   └──────┬───────┘                                     │
│          │                                              │
│          ▼                                              │
│   ┌──────────────┐    ┌──────────────┐                 │
│   │   Training   │───▶│ TensorBoard  │                 │
│   │    Loop      │    │   (local)    │                 │
│   └──────────────┘    └──────────────┘                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Goals:**
- Validate training loop
- Test data pipeline
- Establish baseline metrics
- Iterate quickly without cloud costs

### Phase 2: Cloud Migration

```
┌─────────────────────────────────────────────────────────┐
│                     Google Cloud                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐         ┌──────────────────────┐    │
│   │   GitHub     │────────▶│  Container Registry  │    │
│   │   (code)     │  build  │  (training image)    │    │
│   └──────────────┘         └──────────┬───────────┘    │
│                                       │                 │
│   ┌──────────────┐                    ▼                 │
│   │     GCS      │◀────────┌──────────────────────┐    │
│   │   (data)     │         │   Vertex AI          │    │
│   └──────────────┘────────▶│   Training Jobs      │    │
│          ▲                 └──────────┬───────────┘    │
│          │                            │                 │
│          │         ┌──────────────────┼───────────┐    │
│          │         ▼                  ▼           │    │
│          │  ┌────────────┐    ┌─────────────┐     │    │
│          └──│ Checkpoints│    │ TensorBoard │     │    │
│             │   & Logs   │    │  (managed)  │     │    │
│             └────────────┘    └─────────────┘     │    │
│                                       │           │    │
│                  ┌────────────────────┘           │    │
│                  ▼                                │    │
│           ┌────────────┐                         │    │
│           │   Vizier   │◀────────────────────────┘    │
│           │   (HPO)    │                              │
│           └────────────┘                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Triggers for migration:**
- Training time exceeds ~10 minutes locally
- Need to run multiple experiments in parallel
- HPO sweeps require more compute

---

## Quick Reference: Commands

### Local Development

```bash
# Setup
uv init melody-hpo && cd melody-hpo
uv add torch tensorboard pandas pyyaml

# Run training
python src/train.py --config configs/experiment.yaml

# Start TensorBoard
tensorboard --logdir=experiments/
```

### Cloud Deployment

```bash
# Build and push container
docker build -t gcr.io/PROJECT/melody-hpo:v1 -f docker/Dockerfile .
docker push gcr.io/PROJECT/melody-hpo:v1

# Sync data to GCS
gsutil -m rsync -r ./melody-data gs://BUCKET/data/

# Submit training job (via Python SDK - see Vertex AI docs)
python scripts/submit_job.py --config configs/cloud_experiment.yaml
```

### Data Management

```bash
# Current: track data version
cd melody-data && git rev-parse HEAD  # Record this in experiment config

# Future: query catalog
sqlite3 data/catalog.db "SELECT * FROM works WHERE mode = '8'"
```

---

## Appendix: Future Considerations

### When to Revisit Decisions

| Decision | Revisit If... |
|----------|---------------|
| No local containers | Local/cloud parity issues emerge |
| Git for data | Corpus exceeds ~500MB or needs complex versioning |
| TensorBoard only | HPO comparison becomes painful (50+ runs) |
| T4 GPUs | Training time prohibitive, model complexity increases |
| SQLite catalog | Need concurrent writes or complex relational queries |

### Not Covered (Intentionally Deferred)

- CI/CD automation (manual builds for now)
- Model serving/deployment (research focus, not production)
- Multi-region replication (single-researcher project)
- Advanced data pipelines (current scale doesn't justify)

---

## Document History

| Date | Change |
|------|--------|
| January 2026 | Initial version based on architecture discussion |
