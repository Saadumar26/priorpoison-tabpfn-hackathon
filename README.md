# PriorPoison: poisoning attacks on an in-context tabular foundation model

TabPFN-3.5 never updates weights: the training rows are part of the model input
(in-context learning). So whoever controls the training rows can steer predictions.
PriorPoison measures how vulnerable TabPFN-3.5 is, compared with XGBoost, LogReg and kNN,
and tests two lightweight defenses.

## Attacks
1. **Label flipping** (untargeted): flip x% of training labels, track test accuracy.
2. **Targeted copy injection**: insert k noisy copies of a chosen test point with the wrong label.
   Metric: fraction of targets whose prediction flips to the attacker's label (k=0 is the natural error rate).
3. **Backdoor**: stamp a trigger value on one feature in x% of rows, relabel them as class 1.
   Metric: attack success rate (ASR) on trigger-stamped inputs vs. the same inputs without trigger.

## Defenses
- `oof`: out-of-fold filtering. Drop rows whose label gets low probability from a model trained on the other folds.
- `dedup`: drop rows inside tight near-duplicate clusters (targets copy injection).
Both are evaluated after targeted poisoning (`--defend_k`).

## Run
```bash
pip install -r requirements.txt
# local weights (accept the TabPFN license / set token as in the Prior Labs docs)
python priorpoison.py --dataset breast_cancer --backend local
# or the hosted API
pip install tabpfn-client
python priorpoison.py --dataset blood-transfusion-service-center --backend client
```
Outputs: `results/results_<dataset>.csv` and `results/priorpoison_<dataset>.png`.
`--backend dummy` runs the pipeline with a RandomForest stand-in (smoke test only, not TabPFN).

## Results
(paste your table / plot here and describe honestly, including cases where TabPFN is robust)

## Limitations
Binary tabular datasets, small training sets (500 rows), one trigger feature, one seed by default,
simple (non-adaptive) defenses. An attacker who knows the defense can adapt.

License: Apache-2.0
