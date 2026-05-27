# Metric Glossary

Use these definitions for model-iteration reports.

## Label-Level Metrics

- `support`: human positives for a risk label.
- `predicted`: model predictions for that risk label.
- `TP`: model predicts the label and human label is the same label.
- `FP`: model predicts the label but human label is another label or pass.
- `FN`: human label is the label but model predicts another label or pass.
- `precision`: `TP / (TP + FP)`.
- `recall`: `TP / (TP + FN)`.
- `F1`: harmonic mean of precision and recall.

## Aggregate Metrics

- `exact-label hit precision`: among all non-pass model predictions, the share whose exact risk label matches the human label.
- `exact-label risk recall`: among all non-pass human labels, the share caught with the exact same risk label.
- `binary risk precision`: among all non-pass model predictions, the share where human label is any non-pass risk label.
- `binary risk recall`: among all non-pass human labels, the share where model predicts any non-pass risk label.
- `overall accuracy including pass`: exact match rate across both risk labels and pass.

## Delta Terms

- `少误杀`: false positives removed by the new model.
- `漏召回`: true positives lost by the new model.
- `新增正确`: true positives newly caught by the new model.
- `新增误杀`: false positives newly introduced by the new model.

## Interpretation Notes

Precision gains are not enough for replacement when high-risk recall drops. Prefer label-level rollout when some labels improve but other labels regress.

If the sample is an old-model hit pool, the old model may have 100% recall by construction. State that recall is measured within the labeled hit-pool universe and request background/non-hit samples before claiming online full-traffic recall.
