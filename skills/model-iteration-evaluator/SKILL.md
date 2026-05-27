---
name: model-iteration-evaluator
description: Evaluate model iterations by comparing old-model labels, new-model labels, and human adjudication labels from CSV/XLSX moderation or classification samples. Use when Codex needs to calculate precision, recall, F1, confusion matrices, label-level deltas, false-positive reductions, recall regressions, rollout/replacement recommendations, or generate reusable strategy evaluation reports for new-vs-old machine-review models.
---

# Model Iteration Evaluator

## Overview

Use this skill to turn model-iteration samples into a repeatable replacement decision. The standard input is a CSV containing one row per sample with a human label, old model label, and new model label.

The bundled script `scripts/evaluate_model_iteration.py` is the default implementation. Prefer running it instead of rewriting metric code.

## Expected Inputs

Default columns:

```text
url
旧机审结果
人审结果
新机审结果
```

Default negative/pass label:

```text
通过
```

If the file uses different names, pass explicit column arguments:

```bash
python3 scripts/evaluate_model_iteration.py \
  --input sample.csv \
  --id-col data_id \
  --human-col human_label \
  --old-col old_model_label \
  --new-col new_model_label \
  --negative-label pass \
  --output-dir outputs/model_eval
```

For XLSX input, first export the relevant sheet to CSV or use the spreadsheet skill to inspect/export. Ignore unnamed blank columns unless they contain actual labels or weights.

## Workflow

1. Inspect files with `rg --files`, then inspect CSV shape, columns, head rows, and label distributions.
2. Confirm the evaluation unit. Default to sample/content-level rows; deduplicate only when the user explicitly defines a key and business rule.
3. Identify the human adjudication label column. If both initial review and QA labels exist, use the final adjudicated label when it contains risk classes; if QA columns only contain consistency markers such as `是/否/不确定/True/False`, keep the original human risk-label column.
4. Run `scripts/evaluate_model_iteration.py` into a versioned output directory such as `outputs/model_eval_v2`.
5. Review `model_report.md`, `summary_metrics.csv`, and `label_delta.csv`. Check that the headline precision/recall deltas match the label-level deltas.
6. State the rollout decision with label-level guardrails before aggregate gains.

## Metrics

Compute each risk label one-vs-rest:

```text
TP = model predicts label and human label is label
FP = model predicts label and human label is not label
FN = model does not predict label and human label is label
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Also report:

- exact-label hit precision: risk predictions whose label exactly matches human label
- exact-label risk recall: human risk rows caught with the same risk label
- binary risk precision/recall: any non-pass prediction vs any non-pass human label
- confusion matrices for old and new models
- old -> new -> human transition paths

Read `references/metric-glossary.md` if a user asks about metric definitions or interpretation.

## Decision Rules

Default replacement thresholds:

- block full replacement if overall exact-label recall drops by more than 2 percentage points
- block label-level replacement if a supported label's recall drops by more than 3 percentage points
- treat labels with fewer than 30 human positives as low-support; report them, but avoid over-weighting them
- block or require human-review fallback when high-risk labels show severe recall loss, even if overall precision improves

Use these terms in the report:

- `少误杀`: old model predicted a label, new model did not, and human label is not that label
- `漏召回`: old model predicted a label, new model did not, and human label is that label
- `新增正确`: new model predicted a label, old model did not, and human label is that label
- `新增误杀`: new model predicted a label, old model did not, and human label is not that label

If the old model never predicts the negative/pass label, explicitly caveat that the sample likely represents an old-model hit pool. In that case, recall is valid within the labeled hit-pool universe, but online full-traffic recall still needs background or non-hit samples.

## Script Output

The script writes:

```text
model_report.md
summary_metrics.csv
label_metrics.csv
label_delta.csv
confusion_old.csv
confusion_new.csv
old_new_human_transition.csv
old_new_transition.csv
regression_samples.csv
improvement_samples.csv
changed_samples.csv
```

Use `model_report.md` as the primary deliverable. Link key CSVs only when useful.

## Report Shape

Use `references/report-template.md` for the standard report structure. Keep the final user answer short:

- link the report
- link the reusable script
- quote the core precision/recall deltas
- state whether full replacement is recommended

## Validation

Before finalizing:

1. Run the script successfully on the target CSV.
2. Run `python3 -B -m py_compile scripts/evaluate_model_iteration.py` or an equivalent path check.
3. Inspect the first 200 lines of `model_report.md`.
4. Verify the output directory contains all expected CSVs.
5. Remove generated `__pycache__` files if they appear.
