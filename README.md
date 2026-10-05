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

Both are evaluated after targeted poisoning with 3 copies per target (`--defend_k 3`).

## Results (breast_cancer, 1 seed, 20 targets)

| Attack | TabPFN | XGBoost | LogReg | kNN |
|---|---|---|---|---|
| Label flip 30%: test accuracy | **0.930** | 0.760 | 0.854 | 0.795 |
| Targeted, 2 copies: attack success | 0.75 | 0.85 | 0.10 | 0.25 |
| Targeted, 3 copies: attack success | 1.00 | 1.00 | 0.25 | 1.00 |
| Backdoor 1% poison: ASR | 1.00 | 0.09 | 0.48 | 0.03 |
| Backdoor 2% poison: ASR | 1.00 | 0.94 | 0.72 | 0.31 |

Defenses against targeted poisoning (3 copies per target), attack success after filtering:

| Defense | TabPFN | XGBoost | LogReg | kNN |
|---|---|---|---|---|
| none | 1.00 | 1.00 | 0.25 | 1.00 |
| `dedup` | 0.05 | 0.05 | 0.05 | 0.10 |
| `oof` | 0.95 | 0.75 | 0.15 | 0.95 |

![results](results/priorpoison_breast_cancer.png)

Raw numbers: `results/results_breast_cancer.csv`.

### Findings
1. **TabPFN is the most robust model to random label noise** (93% accuracy at 30% flipped labels vs. 76-85% for the baselines).
2. **It is highly vulnerable to structured poisoning.** Three injected copies per target flip 100% of targets,
   and a backdoor trigger reaches 100% ASR with only ~3 poisoned rows (1%), while behaviour without the trigger
   stays normal (error on non-triggered inputs stays at 5-8%). The same in-context mechanism that lets TabPFN
   learn quickly from few rows also lets an attacker steer it with few rows.
3. **Out-of-fold filtering fails against copy injection** (95% attack success remains for TabPFN), because poison
   copies support each other across folds. Near-duplicate filtering (`dedup`) cuts attack success to 5%.

## Run
```bash
pip install -r requirements.txt
# local weights (accept the TabPFN license / set token as in the Prior Labs docs)
python priorpoison.py --dataset breast_cancer --backend local
# or the hosted API (set TABPFN_TOKEN first)
pip install tabpfn-client
python priorpoison.py --dataset breast_cancer --backend client
```
Outputs: `results/results_<dataset>.csv` and `results/priorpoison_<dataset>.png`.
`--backend dummy` runs the pipeline with a RandomForest stand-in (smoke test only, not TabPFN).

## Limitations
- One dataset (breast_cancer, ~398 training rows, 171 test rows), one seed, 20 targets (5% granularity).
  With ~398 training rows, 1% poison is only ~3 rows. Results should be repeated on more datasets and seeds.
- The backdoor trigger is an out-of-distribution value (feature max + 1 std), which is easy for any model to pick up.
- `dedup` is non-adaptive: an attacker who adds larger noise to the copies can evade it. Its clean-accuracy cost was not measured.
- Binary tabular datasets only; one trigger feature.

License: Apache-2.0
