# BCICIL Experiment Framework

This directory is a clean, self-contained snapshot of the BCI class-incremental learning experiments in the parent project. It supports the three paradigms used in the paper:

- Affective BCI / SEED-3: `seed3` with `Emotion_Net_ln2`
- Motor imagery / BNCI2014001: `mi` with `MI_ADFCNN`
- SSVEP Benchmark: `benchmark` with `SSVEPFormer`

## Included methods

The paper baselines are available through the main entry point: `ER`, `DER`, `ASER`, `CLOPS`, `FastICARL`, `EWC`, `LwF`, `PASS`, `IL2A`, and `ADR`. The repository additionally includes `SFT`, `SI`, `MAS`, `DT2W`, `GR`, and `DERS`.

The proposed method is exposed as `--agent ADR` and uses analytic RLS with score-based target relaxation, including the over-confidence threshold. The underlying implementation file retains its historical internal name for stability.

## Quick start

Create the environment from `environment-github.yml` (the copied `environment.yml` is retained as the original research environment), then check the CLI:

```bash
python main_config_bci.py --help
```

Run a minimal smoke experiment after placing prepared pickles under `data/saved/<dataset>/`:

```bash
python main_config_bci.py --agent ADR --data mi --encoder MI_ADFCNN \
  --device cpu --epochs 1 --runs 1 --cf_matrix False --verbose False
```

For the paper-aligned settings, use `configs/paper_protocol.yaml` as a reference and run `python scripts/check_data.py --dataset mi` before training.

## Data layout

The framework does not redistribute EEG data. Each dataset directory should contain `x_train.pkl`, `state_train.pkl`, `x_test.pkl`, and `state_test.pkl`. Subject-level runs additionally require `subject_label_train.pkl` when the selected method needs it.

The expected shapes and class/task protocol are recorded in `configs/paper_protocol.yaml`. The conversion utilities in `data/seed.py`, `data/mi.py`, and `data/benchmark.py` retain the original preprocessing logic; update their input paths locally rather than committing private data paths.

## Reproducibility

Use `--seed`, `--fix_order True`, an explicit `--device`, and `--runs` greater than one for reported means and confidence intervals. Outputs are written below `result/`; this directory is ignored by Git by default.

## Paper-to-code map

| Paper component | Framework location |
|---|---|
| Unified incremental task stream | `utils/stream.py` |
| Analytic recursive classifier | `agents/utils/analytic_linear.py` |
| ADR analytic learner | `agents/adr.py` with the tested core in `agents/acil.py` |
| Replay baselines | `agents/er.py`, `agents/der.py`, `agents/aser.py`, `agents/clops.py`, `agents/fast_icarl.py` |
| Regularization/prototype baselines | `agents/ewc.py`, `agents/lwf.py`, `agents/pass_agent.py`, `agents/il2a_agent.py` |
| Encoders | `models/encoders.py`, `models/ADFCNN.py`, `models/IFNet.py`, `models/EISATC.py` |
| Experiment entry point | `main_config_bci.py` |
