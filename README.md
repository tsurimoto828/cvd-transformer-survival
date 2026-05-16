# Transformer-Based Survival Model for Cardiovascular Risk Prediction

**Abbreviation:** CVD-Transformer

This repository provides the implementation of a Transformer-based discrete-time survival model for cardiovascular risk prediction using tabular health checkup data. The package now includes manuscript-aligned configuration templates based on the current study design: 43 baseline features, a 10-year prediction horizon, annual discrete-time intervals, seed 0, Adam with learning rate 0.001 and weight decay 1e-5, batch size 256, early stopping patience 10, and a three-block Transformer backbone with four attention heads and a two-layer MLP head.

## Author

**Shota Tsurimoto**  
Department of Cardiovascular Medicine, Kanazawa University Graduate School of Medical Sciences, Kanazawa, Japan  
Email: tsurimoto@staff.kanazawa-u.ac.jp

## Related manuscript

**Title:** Transformer-Based Survival Model for Cardiovascular Risk Prediction from 15-year Longitudinal Health Checkup Data  
**Status:** under preparation

**Authors:**  
Shota Tsurimoto, Akihiro Nomura, Yoshiki Nagata, Yuki Kosaka, Masahiro Noguchi, Tadayuki Hirai, Yasuaki Takeji, Hayato Tada, Kenji Sakata, Soichiro Usui, Shogo Okada, Masayuki Takamura

## Task

Discrete-time survival prediction

## Input data

Tabular health checkup data including 21 continuous and 22 categorical baseline variables.

## Output

Interval-wise conditional survival probabilities, cumulative survival probabilities, and 10-year CVD risk.

## Data availability

The original health checkup data are not publicly available due to privacy and institutional restrictions.

## Pretrained weights

Pretrained model weights are not publicly available.

## What is included

- package-ready source code under `src/cvd_transformer_survival/`
- training CLI: `python -m cvd_transformer_survival.train --config configs/train.yaml`
- inference CLI: `python -m cvd_transformer_survival.predict --config configs/predict.yaml`
- manuscript-aligned YAML config files for training and prediction
- example inference CSV with the manuscript feature schema
- artifact saving for model weights and preprocessing metadata

## Manuscript-aligned feature schema

### Continuous columns (21)

`age`, `height`, `weight`, `bmi`, `waist`, `sbp`, `dbp`, `rbc`, `hemoglobin`, `hematocrit`, `ast`, `alt`, `ggt`, `total_chol`, `hdl_c`, `triglyceride`, `ldl_c`, `glucose`, `hba1c`, `creatinine`, `egfr`

### Categorical columns (22)

`sex`, `urine_protein`, `urine_glucose`, `ecg_abnormality`, `eye_exam`, `anti_ht_med`, `insulin_med`, `anti_chol_med`, `past_anemia`, `smoker`, `weight_gain`, `regular_exercise`, `daily_exercise`, `fast_walking`, `eating_speed`, `late_meals`, `skip_breakfast`, `habitual_drinker`, `alcohol_consumption`, `enough_sleep`, `lifestyle_change`, `health_interest`

### Outcome columns for training

- `time`: follow-up time in years from baseline
- `f`: event indicator (`1` = incident CVD, `0` = censored)

## Important preprocessing notes

- The manuscript used complete-case analysis. In the provided manuscript-aligned training config, `require_complete_cases: true` drops rows with missing required variables before fitting.
- Continuous variables are standardized using z-scores based on the training-set mean and standard deviation.
- Categorical variables are label-encoded from the training CSV.
- Annual bins from 0 to 10 years are used to represent the 10-year discrete-time survival horizon.

## Installation

```bash
git clone https://github.com/tsurimoto828/cvd-transformer-survival.git
cd cvd-transformer-survival
python -m pip install -U pip
python -m pip install -e .
```

## Training

After training, the following files are saved in `output.output_dir`:

- `model_survival.pth`: trained model weights
- `metadata.json`: preprocessing and model metadata
- `training_log.csv`: epoch-wise training and validation losses
- `loss_curve.png`: training and validation loss curve


```bash
python -m cvd_transformer_survival.train --config configs/train.yaml
```

Artifacts are saved under the directory specified by `output.output_dir`.

- `model_survival.pth`: trained model weights
- `metadata.json`: preprocessing and architecture metadata required for inference
- `training_summary.json`: training summary

## Inference

```bash
python -m cvd_transformer_survival.predict --config configs/predict.yaml
python examples/example_inference.py
```

The prediction file contains both interval-wise conditional survival probabilities and cumulative survival probabilities. The example script also computes:

- `predicted_10y_cvd_risk = 1 - cumulative_survival_interval_10`

## Main file structure

```text
src/cvd_transformer_survival/model.py      # Transformer-based survival model
src/cvd_transformer_survival/dataset.py    # dataset and dataloader preparation
src/cvd_transformer_survival/loss.py       # survival likelihood loss
src/cvd_transformer_survival/trainer.py    # training and checkpointing
src/cvd_transformer_survival/train.py      # training entrypoint
src/cvd_transformer_survival/predict.py    # inference entrypoint
src/cvd_transformer_survival/utils.py      # preprocessing and helper functions
configs/train.yaml                         # manuscript-aligned training configuration
configs/predict.yaml                       # manuscript-aligned inference configuration
examples/manuscript_example_input.csv      # example inference input in manuscript format
```

## Notes

- `dim: 256` with `heads: 4` corresponds to 64 dimensions per attention head.
- `mlp_hidden_layers: [128, 64]` matches the two-layer MLP description used in the manuscript.
- If you need to match the exact cross-validation and benchmark workflow in the manuscript, add your study-specific data split and evaluation scripts on top of this packaging-friendly training path.
- `num_threads: 1` is the safest default for portable CPU execution; increase it after confirming local stability.

## Citation

If you use this repository, please cite the software repository and the related manuscript when available.

## License

BSD-3-Clause

## Contact

Shota Tsurimoto  
Department of Cardiovascular Medicine, Kanazawa University Graduate School of Medical Sciences, Kanazawa, Japan  
tsurimoto@staff.kanazawa-u.ac.jp
