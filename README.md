ID-PaS+: Identity-Aware Predict-and-Search for General Mixed-Integer Linear Programs

This repository contains research code for training and evaluating graph-based prediction models on MILP instances, then using those predictions inside downstream search procedures.

## Repository layout

- `collect_data.py`: data collection pipeline for Gurobi runs
- `train.py`: train the GAT-based prediction model
- `evaluate_pas.py`: run the original binary-only PaS search
- `evaluate_idpas.py`: run the ID-PAS+ search with separate binary and integer zero-fixing tracks
- `MIPDataset.py`: dataset loading and graph feature construction
- `GAT.py`: graph attention network policy

## Training and search

- `train.py` trains on all discrete variables (binary and general integer) using zero vs. non-zero targets.
- `evaluate_idpas.py` performs the paper-style search over separate binary and integer zero-fixing percentages with a trust-region budget defined as a percentage of `|X0|`.

Example training command:

```bash
python train.py --taskName my_task --instance_dir data/instances --result_dir data/solutions --var_nfeats 32 --learning_rate 1e-5 --batch_size 16 --max_train_hours 48
```

Example original PaS command:

```bash
python evaluate_pas.py --instance path/to/instance.mps --expname my_eval --model pretrain/binary_pas/model_best.pth --k0-pcts 0,10,20,30,40,50 --k1-pcts 0,10,20,30,40,50 --delta-pcts 1,3,5
```

Example ID-PAS+ command:

```bash
python evaluate_idpas.py --instance path/to/instance.mps --expname my_eval --methods IM,IM_id --kb0-pcts 0,10,30,50,70,90 --ki0-pcts 0,10,30,50,70,90 --delta-pcts 1,3,5
```

## Notes

For Middle Mile and SLAP instances, the authors note that benchmark data has also been added to Distributional MIPLIB:
[Distributional MIPLIB](https://sites.google.com/usc.edu/distributional-miplib/)
