# IDSPY

IDSPY is a modular framework for building intrusion detection pipelines. It combines event-driven orchestration, reusable pipeline components, and seamless machine learning integration to help you prototype and deploy detection workflows quickly.

## Highlights

- Modular pipeline engine with shared storage and event hooks
- Batteries-included data tooling for tabular intrusion detection datasets
- PyTorch helpers for training, evaluating, and deploying custom models
- Production-ready implementations of common steps, handlers, and utilities

## Architecture Overview

Each module in `src/idspy/` plays a specific role in the pipeline ecosystem:

### Core (`src/idspy/core/`)
- **Pipeline** coordinates multi-step execution with shared storage and event routing.
- **Step** defines composable units of work that can be chained together.
- **Storage** provides persistent key-value storage for passing artifacts across steps.
- **Events** enables monitoring, logging, and extension through an event bus.

### Data (`src/idspy/data/`)
- **Repository** reads and writes pandas DataFrames with automatic format detection (Parquet, CSV, pickle).
- **Schema** validates expected data contracts before processing.
- **Partition** offers reusable splitting and partitioning strategies.
- **TabAccessor** extends pandas with intrusion detection–friendly access patterns.

### Neural Networks (`src/idspy/nn/`)
- **Torch helpers** simplify device management and reproducible training.
- **Model integration** wraps custom neural networks for pipeline execution.
- **Loss functions** provide tailored losses for detection scenarios.

### Builtins (`src/idspy/builtins/`)
- **Handlers** supply production-ready event handlers for pipelines.
- **Steps** includes curated data processing and ML workflow steps.

### Common (`src/idspy/common/`)
- **Logging** centralizes configuration for consistent diagnostics.
- **Predicates** implements a functional predicate system for filtering and validation.
- **Profiler** captures timing and performance data across runs.
- **Utils** gathers shared helpers used by multiple modules.

## Repository Layout

```text
idspy/
├── src/idspy/
│   ├── core/        # Pipeline engine and execution primitives
│   ├── data/        # Tabular data tooling
│   ├── common/      # Shared utilities and diagnostics
│   ├── builtins/    # Ready-to-use steps and handlers
│   └── nn/          # PyTorch integration layer
├── examples/        # Jupyter notebooks with end-to-end guides
├── resources/       # Sample datasets and trained models
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- `pip`

### Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/Simone-Albero/idspy.git
   cd idspy
   ```
2. Create a virtual environment.
   ```bash
   python3 -m venv .venv
   ```
3. Activate the environment.
   - macOS/Linux
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell)
     ```bash
     .venv\\Scripts\\Activate.ps1
     ```
4. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Examples

Explore the [`examples/`](examples/) directory for Jupyter notebooks that demonstrate typical workflows:

- `data.ipynb` — Data loading, schemas, repositories, and partitioning.
- `steps.ipynb` — Building and composing custom processing steps.
- `storage.ipynb` — Sharing intermediate artifacts through the storage layer.
- `pipeline.ipynb` — Orchestrating full ML pipelines end to end.
- `events.ipynb` — Extending pipelines with custom event handlers.
- `training.ipynb` — Training and evaluating models within the pipeline.

Each notebook walks through practical scenarios so you can adapt the patterns to your own intrusion detection projects.
