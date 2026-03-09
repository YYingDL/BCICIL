## Overview

BCLCIL (Brain computer interfaces Class-Incremental Learning) - A PyTorch framework for continual learning on BCI classification tasks. Supports multiple datasets (SEED-3, MI, BCI Benchmark) and implements various CL algorithms.

## Commands

```bash

conda activate cil
# Train/evaluate agents (main entry point)
python main_config_bci.py --agent <AGENT> --data <DATASET> --encoder <BACKBONE> [options]

# Hyperparameter tuning
python main_tune.py --agent <AGENT> --data <DATASET> --encoder <BACKBONE>

# Common agent choices: SFT, Offline, LwF, EWC, SI, MAS, DT2W, ER, ASER, DER, CLOPS, FastICARL, ACIL, GR
# Dataset choices: seed3, mi, benchmark
# Encoder choices: Emotion_Net_ln2, MI_EEGNet, FBSSVEPFormer
```

## Architecture

### Core Components

```
agents/          # CL algorithm implementations (base.py defines BaseLearner interface)
  ├── base.py    # BaseLearner abstract class with learn_task(), evaluate() methods
  ├── acil.py    # Analytic Class-Incremental Learning (two-phase: base training + RLS)
  └── ...        # Other agents (ER, EWC, LwF, DER, etc.)

data/            # Dataset loaders (benchmark.py, mi.py, seed.py)
models/          # Neural network architectures
  ├── base.py    # SingleHeadModel setup
  ├── encoders.py # Backbones: FBSSVEPFormer, EEGNet, etc.
  └── classifier.py # Head types: Linear, CosineLinear, SplitCosineLinear

utils/
  ├── stream.py  # IncrementalTaskStream - creates task sequences
  ├── buffer/    # Memory buffer strategies (reservoir, MIR, ASER, etc.)
  └── setup_elements.py # Dataset configs (n_classes, n_tasks, input_size)
```

### Data Flow

1. `IncrementalTaskStream` loads data and splits into tasks based on class order
2. Each task: `(x_train, y_train), (x_val, y_val), (x_test, y_test)`
3. Agents implement `learn_task(task)` - trains on task, optionally uses memory buffer
4. After each task: `evaluate()` measures accuracy on all learned tasks

### Key Patterns

**Dynamic Head Expansion**: Model head grows as new classes arrive:
- `before_task()` expands head for new classes
- `SplitCosineLinear` concatenates old weights with new class weights

**Memory Buffer** (for replay methods):
- `Buffer` class in `utils/buffer/buffer.py`
- Pluggable update/retrieve strategies via `name_match.py`
- DER stores logits for Dark Experience Replay

**ACIL Special Case**: Two-phase learning
- Task 0: Train backbone with gradient descent
- Task 1+: Freeze backbone, use Recursive Least Squares for classifier updates

### Configuration Keys (`utils/setup_elements.py`)

```python
input_size_match = {'seed3': [62, 5], 'mi': [1001, 22], 'benchmark': [3, 250, 9]}
n_classes = {'seed3': 3, 'mi': 4, 'benchmark': 40}
n_tasks = {'seed3': 2, 'mi': 3, 'benchmark': 5}
```

## Dependencies

Install from `environment.yml` (requires conda):
```bash
conda env create -f environment.yml
```

Key: PyTorch 1.13.1, CUDA 11.6, scikit-learn, pandas, matplotlib, seaborn
