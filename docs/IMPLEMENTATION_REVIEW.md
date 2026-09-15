# Implementation review

## Scope distinction

The attached paper is treated as a source of experimental protocol and method requirements. Only the user's request authorizes repository changes. The framework therefore preserves code that exists in the repository and labels compatibility entries explicitly.

## Paper coverage

The paper evaluates exemplar replay methods (`ER`, `DER`, `ASER`, `CLOPS`, `FastICARL`), privacy-preserving regularization/prototype methods (`EWC`, `LwF`, `PASS`, `IL2A`), and the proposed `ADR` method. The current repository contains all named baselines. The public `ADR` entry uses analytic RLS with discriminative target relaxation and a score threshold; the underlying module retains its historical internal filename for stability.

## Current implementation inventory

| Area | Current implementation |
|---|---|
| Experiment orchestration | `main_config_bci.py`, `experiment/exp.py` |
| Paradigms | `seed3`, `mi`, `benchmark` |
| Encoders | `Emotion_Net_ln2`, `MI_EEGNet`, `MI_IFNet`, `MI_ADFCNN`, `MI_EISATC`, `SSVEPFormer` |
| Replay | `ER`, `DER`, `DERS`, `ASER`, `CLOPS`, `FastICARL`, `GR` |
| Regularization/distillation | `EWC`, `LwF`, `SI`, `MAS`, `DT2W` |
| Prototype/augmentation | `PASS`, `IL2A` |
| Analytic | `ADR` via `agents/adr.py` |

## Known reproducibility risks inherited from the source snapshot

- EEG data are not included and must be prepared as local pickle files under `data/saved/<dataset>/`.
- The original conversion scripts contain machine-specific source paths; those paths are intentionally not generalized without access to the datasets.
- The complete paper protocol should be run with the paper's optimizer, epoch, batch-size, weight-decay, class order, and cross-subject settings rather than relying on CLI defaults.
- `--cf_matrix False` is recommended for headless smoke tests; plotting is an optional output.

## Recommended claim boundary

The GitHub snapshot is a runnable BCICIL experiment framework containing the paper baselines and the public `ADR` analytic learner. The paper's explicit BMR loss is not separately implemented in this source snapshot and should be described accurately in any publication.
