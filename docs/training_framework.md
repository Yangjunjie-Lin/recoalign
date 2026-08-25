# ReCoAlign training framework

## Status: inactive historical/toy framework

The research pivot blocks model development. TRAIN001–TRAIN003 and the commands below remain for
reproducibility and implementation tests, but they are not the next experiment and cannot generate
claim-eligible evidence under the current decision.

## Scientific contract

ReCoAlign learns structure tokens from visual tokens. Scene graphs are permitted
only as Stage-1 target labels; the trainer rejects `graph`, `oracle_graph`,
`scene_graph`, and `graph_tokens` batch inputs. The same `forward(visual_tokens)`
contract is used during training and inference.

Training experiments are registered separately in
`research/experiments/training_registry.yaml`. This preserves the EXP001–EXP004
governance rule that those mechanism-validation experiments have training disabled.

## Stages

| Training experiment | Stage | Scientific purpose | Required supervision |
| --- | --- | --- | --- |
| TRAIN001 | Structured interface pretraining | Learn visual → latent structure | semantic and object/attribute/relation/composition targets |
| TRAIN002 | Reasoning alignment | Make structure tokens accessible to the QA path | image, question representation, answer |
| TRAIN003 | Instruction adaptation | Transfer the same interface to a preregistered real task | image, instruction, answer |

TRAIN003 is configuration-ready but scientifically pending selection of a real
dataset manifest and backbone checkpoint. The included CLI deliberately exposes
only `--toy-data`; it cannot silently substitute synthetic data for a real run.

## Pipeline

`ReCoAlignTrainer` applies the freeze policy, constructs the optimizer/scheduler,
checks required supervision and forbidden inputs, evaluates every epoch, records
JSONL metrics, renders SVG curves, and saves full resumable checkpoints.

Each run contains:

- `config.resolved.yaml`
- `environment.json`
- `git_commit.txt`
- `dataset_manifest.yaml`
- `checkpoint_manifest.yaml`
- `metrics.json`
- `training_provenance.json`
- `logs/training.jsonl`, `logs/validation.jsonl`, and `logs/loss_curves.svg`
- checkpoint model, optimizer, scheduler, metrics, configuration, and RNG state

## Commands

```text
recoalign validate-training-registry
recoalign validate-training-fairness
recoalign train-recoalign --config configs/training/recoalign_stage1.yaml --toy-data
recoalign run-training-sanity --config configs/training/recoalign_stage1.yaml
recoalign run-recoalign-ablation --config configs/ablations/no_structure.yaml --output RUN_DIR
```
