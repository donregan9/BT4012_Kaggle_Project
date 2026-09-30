# LightGBM SSL Notebook Explanation

This notebook builds a semi-supervised LightGBM model for a Kaggle competition. The key idea is:

- use the rows with known labels to train a baseline model,
- use the rows with missing labels as extra unlabeled data,
- keep only high-confidence pseudo-labels,
- retrain the model with those pseudo-labels added,
- generate a final submission file.

---

## 1) Loading the competition data

```python
import os
import pandas as pd
import numpy as np

data_dir = r"C:\Users\regan\.cache\kagglehub\competitions\bt-4012-competition-2026"
print("Files:", os.listdir(data_dir))

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
edges_df = pd.read_csv(os.path.join(data_dir, "txs_edgelist.csv"))
```

This part:

- points to the Kaggle competition folder,
- lists all files available there,
- reads the training data, test data, and the edge list,
- prints dataset shapes so the data can be sanity-checked before training.

The data is likely structured as a transaction dataset where each row represents a record and `label` is the target for the labeled set.

---

## 2) Choosing model features

```python
features = [col for col in train_df.columns if col not in ['txId', 'label']]
X = train_df[features].copy()
y = train_df['label']
X_test = test_df[features].copy()
```

This creates the feature matrix `X` and target vector `y`.

- `txId` is excluded because it is an identifier, not a useful predictive feature.
- `label` is removed from the feature columns because it is the target variable.
- The same feature columns are reused for the test set.

This is standard machine learning practice: train on all useful features, while keeping the target separate.

---

## 3) Creating a temporal train/validation split

```python
cutoff_step = train_df['time_step'].quantile(0.75)

labeled_mask = y.notna()
train_mask = (train_df['time_step'] <= cutoff_step) & labeled_mask
val_mask = (train_df['time_step'] > cutoff_step) & labeled_mask
unlabeled_train_mask = (train_df['time_step'] <= cutoff_step) & (~labeled_mask)

X_tr, y_tr = X.loc[train_mask], y.loc[train_mask].astype(int)
X_val, y_val = X.loc[val_mask], y.loc[val_mask].astype(int)
X_unlabeled_train = X.loc[unlabeled_train_mask]
```

This is a time-based split, which is important for Kaggle-style temporal data.

- `cutoff_step` is the 75th percentile of `time_step`.
- Rows before that cutoff are used as the training pool.
- Rows after that cutoff are used as validation data.
- Rows with missing labels are treated as unlabeled data for pseudo-labeling.

The notebook intentionally uses a temporal split instead of random splitting to reduce leakage and simulate real-world forecasting conditions.

---

## 4) Handling class imbalance

```python
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
```

This computes the class imbalance ratio.

If the positive class is much rarer than the negative class, LightGBM may underweight the minority class during training. `scale_pos_weight` helps balance that by giving minority-class examples more influence.

---

## 5) Training the base LightGBM classifier

```python
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

pos_ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

model = LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    scale_pos_weight=pos_ratio,
    random_state=42,
    n_jobs=-1
)
```

This defines a binary classification model using LightGBM.

Important settings:

- `objective='binary'`: the target is binary (0 or 1).
- `metric='auc'`: the model is evaluated using ROC AUC, which is appropriate for class imbalance.
- `n_estimators=1000`: up to 1000 boosting trees.
- `learning_rate=0.03`: slower learning, often more stable.
- `scale_pos_weight=pos_ratio`: corrects class imbalance.
- `random_state=42`: makes results reproducible.

This is a tuned gradient boosting model suitable for tabular data.

---

## 6) Stage 1: train on labeled rows only

```python
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    eval_names=['train', 'val'],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)
```

This fits the LightGBM model using only rows that are genuinely labeled.

- `eval_set` tracks performance on both the training set and validation set.
- `eval_metric='auc'` tells LightGBM to monitor ROC AUC.
- `early_stopping` stops training if validation performance stops improving.
- `log_evaluation` prints progress every 50 boosting rounds.

This gives a baseline model before using pseudo-labels.

---

## 7) Pseudo-labeling the unlabeled rows

```python
unlabeled_probs = model.predict_proba(X_unlabeled_train)[:, 1]
confidence = np.maximum(unlabeled_probs, 1 - unlabeled_probs)
pseudo_mask = confidence >= 0.95

X_pseudo = X_unlabeled_train.loc[pseudo_mask]
y_pseudo = pd.Series((unlabeled_probs[pseudo_mask] >= 0.5).astype(int), index=X_pseudo.index)
```

This is the semi-supervised learning step.

- The trained model predicts probability for each unlabeled row.
- `confidence` is the distance from 0.5, meaning how sure the model is about the prediction.
- `pseudo_mask = confidence >= 0.95` keeps only highly confident predictions.
- For these rows, the model converts probabilities into binary pseudo-labels.

This approach is common in semi-supervised learning: use confident predictions as additional training labels.

---

## 8) Augmenting the training set

```python
X_aug = pd.concat([X_tr, X_pseudo])
y_aug = pd.concat([y_tr, y_pseudo])
```

Now the model combines:

- real labeled rows from `X_tr`, and
- high-confidence pseudo-labeled rows from `X_pseudo`.

This creates a larger and more informative training set without adding obviously unreliable labels.

---

## 9) Retraining on real + pseudo labels

```python
model.fit(
    X_aug, y_aug,
    eval_set=[(X_aug, y_aug), (X_val, y_val)],
    eval_names=['train_augmented', 'val'],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f'\n---> Temporal Validation ROC-AUC: {val_auc:.5f}')
```

This retrains the model after pseudo-labeling.

- `X_aug` includes both authentic labels and the selected pseudo-labels.
- The validation set remains untouched, so the evaluation is still honest.
- ROC AUC is computed to measure temporal generalization.

This second training pass is the main semi-supervised step and is often the reason the model improves beyond the purely supervised baseline.

---

## 10) Final model and submission generation

```python
X_labeled = X.loc[labeled_mask]
y_labeled = y.loc[labeled_mask].astype(int)
X_full_train = pd.concat([X_labeled, X_pseudo])
y_full_train = pd.concat([y_labeled, y_pseudo])

model.fit(X_full_train, y_full_train)
test_preds = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'index': test_df['index'],
    'target': test_preds
})
submission.to_csv('submission_pseudolabelled.csv', index=False)
```

This is the final prediction stage.

- All genuinely labeled rows are kept.
- The selected pseudo-labeled rows are also included.
- The final model is trained on this full set.
- Predictions are made for the test set.
- Results are saved to `submission_pseudolabelled.csv` so they can be submitted for evaluation.

---

## Why this is a semi-supervised learning approach

This notebook uses a common SSL strategy:

1. train a model on labeled data,
2. predict unlabeled data,
3. keep only high-confidence predictions,
4. add those predictions to the training set,
5. retrain and generate final forecasts.

This works well when:

- the labeled set is much smaller than the unlabeled set,
- the model is confident on many unlabeled rows,
- the data distribution is stable enough that pseudo-labels are useful.

It is especially useful in tabular competition settings where unlabeled data is available but ground truth is missing.

---

## Summary

This notebook is a practical example of a semi-supervised, time-aware LightGBM pipeline. It:

- loads competition data,
- chooses features,
- splits by time,
- trains a strong classifier,
- adds confident pseudo-labels,
- retrains for better performance,
- exports predictions for submission.

The main difference from a fully supervised pipeline is the pseudo-labeling step, which turns unlabeled data into extra training examples.
