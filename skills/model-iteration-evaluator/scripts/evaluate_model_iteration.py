#!/usr/bin/env python3
"""Evaluate whether a new moderation classifier can replace an old one."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare old/new machine-review labels against human labels and "
            "emit precision/recall deltas plus a replacement recommendation."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output-dir", default="outputs/model_eval", help="Output directory.")
    parser.add_argument("--id-col", default="url", help="Sample identifier column.")
    parser.add_argument("--human-col", default="人审结果", help="Human adjudication label column.")
    parser.add_argument("--old-col", default="旧机审结果", help="Old model label column.")
    parser.add_argument("--new-col", default="新机审结果", help="New model label column.")
    parser.add_argument("--negative-label", default="通过", help="Negative/pass label.")
    parser.add_argument(
        "--max-overall-recall-drop",
        type=float,
        default=0.02,
        help="Maximum acceptable exact positive recall drop for full replacement.",
    )
    parser.add_argument(
        "--max-label-recall-drop",
        type=float,
        default=0.03,
        help="Maximum acceptable per-label recall drop for labels with enough support.",
    )
    parser.add_argument(
        "--min-label-support",
        type=int,
        default=30,
        help="Minimum human-label support for per-label rollout guardrails.",
    )
    return parser.parse_args()


def clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else math.nan


def f1_score(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        return math.nan
    return 2 * precision * recall / (precision + recall)


def wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    phat = success / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return ((centre - margin) / denom, (centre + margin) / denom)


def pct(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.2%}"


def signed_pct(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2%}"


def fmt_num(value: float) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:,.0f}"


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def unique_labels(df: pd.DataFrame, cols: Iterable[str], negative_label: str) -> list[str]:
    labels: set[str] = set()
    for col in cols:
        labels.update(df[col].dropna().map(clean_label).unique().tolist())
    labels.discard("")
    labels.discard(negative_label)
    return sorted(labels)


def label_metrics(
    df: pd.DataFrame,
    model_name: str,
    pred_col: str,
    human_col: str,
    labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(df)
    for label in labels:
        pred = df[pred_col] == label
        truth = df[human_col] == label
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        tn = int(n - tp - fp - fn)
        predicted = tp + fp
        support = tp + fn
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        p_low, p_high = wilson_ci(tp, predicted)
        r_low, r_high = wilson_ci(tp, support)
        rows.append(
            {
                "model": model_name,
                "label": label,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "precision_ci_low": p_low,
                "precision_ci_high": p_high,
                "recall": recall,
                "recall_ci_low": r_low,
                "recall_ci_high": r_high,
                "f1": f1_score(precision, recall),
                "fp_rate_among_hits": safe_div(fp, predicted),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(
    df: pd.DataFrame,
    model_name: str,
    pred_col: str,
    human_col: str,
    labels: list[str],
    negative_label: str,
    per_label: pd.DataFrame,
) -> dict[str, object]:
    pred_positive = df[pred_col] != negative_label
    human_positive = df[human_col] != negative_label
    exact_positive = (df[pred_col] == df[human_col]) & human_positive
    binary_tp = int((pred_positive & human_positive).sum())
    binary_fp = int((pred_positive & ~human_positive).sum())
    binary_fn = int((~pred_positive & human_positive).sum())
    binary_precision = safe_div(binary_tp, binary_tp + binary_fp)
    binary_recall = safe_div(binary_tp, binary_tp + binary_fn)
    exact_tp = int(exact_positive.sum())
    exact_precision = safe_div(exact_tp, int(pred_positive.sum()))
    exact_recall = safe_div(exact_tp, int(human_positive.sum()))
    accuracy = safe_div(int((df[pred_col] == df[human_col]).sum()), len(df))
    precision_low, precision_high = wilson_ci(exact_tp, int(pred_positive.sum()))
    recall_low, recall_high = wilson_ci(exact_tp, int(human_positive.sum()))
    macro_precision = per_label["precision"].mean(skipna=True)
    macro_recall = per_label["recall"].mean(skipna=True)
    macro_f1 = per_label["f1"].mean(skipna=True)
    micro_tp = int(per_label["tp"].sum())
    micro_fp = int(per_label["fp"].sum())
    micro_fn = int(per_label["fn"].sum())
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    return {
        "model": model_name,
        "samples": len(df),
        "human_positive": int(human_positive.sum()),
        "human_negative": int((~human_positive).sum()),
        "pred_positive": int(pred_positive.sum()),
        "pred_negative": int((~pred_positive).sum()),
        "exact_positive_tp": exact_tp,
        "hit_precision_exact_label": exact_precision,
        "hit_precision_ci_low": precision_low,
        "hit_precision_ci_high": precision_high,
        "risk_recall_exact_label": exact_recall,
        "risk_recall_ci_low": recall_low,
        "risk_recall_ci_high": recall_high,
        "overall_accuracy_including_pass": accuracy,
        "binary_risk_tp": binary_tp,
        "binary_risk_fp": binary_fp,
        "binary_risk_fn": binary_fn,
        "binary_risk_precision": binary_precision,
        "binary_risk_recall": binary_recall,
        "binary_risk_f1": f1_score(binary_precision, binary_recall),
        "macro_label_precision": macro_precision,
        "macro_label_recall": macro_recall,
        "macro_label_f1": macro_f1,
        "micro_label_precision": micro_precision,
        "micro_label_recall": micro_recall,
        "micro_label_f1": f1_score(micro_precision, micro_recall),
    }


def build_label_delta(
    df: pd.DataFrame,
    old_label_df: pd.DataFrame,
    new_label_df: pd.DataFrame,
    labels: list[str],
    human_col: str,
    old_col: str,
    new_col: str,
) -> pd.DataFrame:
    old_lookup = old_label_df.set_index("label").to_dict("index")
    new_lookup = new_label_df.set_index("label").to_dict("index")
    rows: list[dict[str, object]] = []
    for label in labels:
        old_pred = df[old_col] == label
        new_pred = df[new_col] == label
        truth = df[human_col] == label
        old = old_lookup[label]
        new = new_lookup[label]
        rows.append(
            {
                "label": label,
                "support": int(truth.sum()),
                "old_predicted": int(old_pred.sum()),
                "new_predicted": int(new_pred.sum()),
                "predicted_delta": int(new_pred.sum() - old_pred.sum()),
                "old_tp": int(old["tp"]),
                "new_tp": int(new["tp"]),
                "tp_delta": int(new["tp"] - old["tp"]),
                "old_fp": int(old["fp"]),
                "new_fp": int(new["fp"]),
                "fp_delta": int(new["fp"] - old["fp"]),
                "old_fn": int(old["fn"]),
                "new_fn": int(new["fn"]),
                "fn_delta": int(new["fn"] - old["fn"]),
                "old_precision": float(old["precision"]),
                "new_precision": float(new["precision"]),
                "precision_delta": float(new["precision"] - old["precision"]),
                "old_recall": float(old["recall"]),
                "new_recall": float(new["recall"]),
                "recall_delta": float(new["recall"] - old["recall"]),
                "old_f1": float(old["f1"]),
                "new_f1": float(new["f1"]),
                "f1_delta": float(new["f1"] - old["f1"]),
                "old_only_false_positives_removed": int((old_pred & ~new_pred & ~truth).sum()),
                "old_only_true_positives_lost": int((old_pred & ~new_pred & truth).sum()),
                "new_only_correct_catches": int((new_pred & ~old_pred & truth).sum()),
                "new_only_false_positives_added": int((new_pred & ~old_pred & ~truth).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True if df.index.name else False, encoding="utf-8-sig")


def decision_text(
    summary_delta: dict[str, float],
    label_delta: pd.DataFrame,
    args: argparse.Namespace,
    old_has_negative_predictions: bool,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if summary_delta["risk_recall_delta"] < -args.max_overall_recall_drop:
        blockers.append(
            f"整体精确风险召回下降 {signed_pct(summary_delta['risk_recall_delta'])}，"
            f"超过阈值 {pct(args.max_overall_recall_drop)}"
        )

    guarded = label_delta[label_delta["support"] >= args.min_label_support].copy()
    recall_blockers = guarded[guarded["recall_delta"] < -args.max_label_recall_drop]
    for _, row in recall_blockers.sort_values("recall_delta").iterrows():
        blockers.append(
            f"{row['label']} 召回下降 {signed_pct(row['recall_delta'])} "
            f"({pct(row['old_recall'])}->{pct(row['new_recall'])})"
        )

    fp_added = guarded[guarded["new_only_false_positives_added"] > 0]
    for _, row in fp_added.sort_values("new_only_false_positives_added", ascending=False).iterrows():
        blockers.append(f"{row['label']} 新增误杀 {int(row['new_only_false_positives_added'])} 条")

    if not old_has_negative_predictions:
        blockers.append("旧模型没有`通过`预测，本样本更像旧模型命中池，线上全量召回仍需背景流量样本验证")

    if blockers:
        return (
            "不建议直接全量替换；建议按标签灰度/旁路，并对召回下降标签保留旧模型或进入人审兜底。",
            blockers,
        )
    return (
        "可以进入全量替换前灰度；当前样本下新模型精准率提升且未触发召回护栏。",
        blockers,
    )


def build_report(
    output_dir: Path,
    source_name: str,
    df: pd.DataFrame,
    labels: list[str],
    summaries: pd.DataFrame,
    label_delta: pd.DataFrame,
    transition_summary: pd.DataFrame,
    args: argparse.Namespace,
    recommendation: str,
    blockers: list[str],
) -> str:
    old_summary = summaries[summaries["model"] == "old"].iloc[0]
    new_summary = summaries[summaries["model"] == "new"].iloc[0]

    summary_delta = {
        "pred_positive_delta": int(new_summary["pred_positive"] - old_summary["pred_positive"]),
        "precision_delta": float(
            new_summary["hit_precision_exact_label"] - old_summary["hit_precision_exact_label"]
        ),
        "risk_recall_delta": float(
            new_summary["risk_recall_exact_label"] - old_summary["risk_recall_exact_label"]
        ),
        "accuracy_delta": float(
            new_summary["overall_accuracy_including_pass"]
            - old_summary["overall_accuracy_including_pass"]
        ),
    }

    key_rows = [
        {
            "指标": "风险命中量",
            "旧模型": fmt_num(float(old_summary["pred_positive"])),
            "新模型": fmt_num(float(new_summary["pred_positive"])),
            "变化": fmt_num(float(summary_delta["pred_positive_delta"])),
        },
        {
            "指标": "精确标签精准率",
            "旧模型": pct(float(old_summary["hit_precision_exact_label"])),
            "新模型": pct(float(new_summary["hit_precision_exact_label"])),
            "变化": signed_pct(summary_delta["precision_delta"]),
        },
        {
            "指标": "精确标签召回率",
            "旧模型": pct(float(old_summary["risk_recall_exact_label"])),
            "新模型": pct(float(new_summary["risk_recall_exact_label"])),
            "变化": signed_pct(summary_delta["risk_recall_delta"]),
        },
        {
            "指标": "含通过整体准确率",
            "旧模型": pct(float(old_summary["overall_accuracy_including_pass"])),
            "新模型": pct(float(new_summary["overall_accuracy_including_pass"])),
            "变化": signed_pct(summary_delta["accuracy_delta"]),
        },
        {
            "指标": "二分类风险召回",
            "旧模型": pct(float(old_summary["binary_risk_recall"])),
            "新模型": pct(float(new_summary["binary_risk_recall"])),
            "变化": signed_pct(float(new_summary["binary_risk_recall"] - old_summary["binary_risk_recall"])),
        },
    ]

    label_rows: list[dict[str, object]] = []
    for _, row in label_delta.sort_values("support", ascending=False).iterrows():
        label_rows.append(
            {
                "标签": row["label"],
                "人审量": int(row["support"]),
                "旧P": pct(float(row["old_precision"])),
                "新P": pct(float(row["new_precision"])),
                "P变化": signed_pct(float(row["precision_delta"])),
                "旧R": pct(float(row["old_recall"])),
                "新R": pct(float(row["new_recall"])),
                "R变化": signed_pct(float(row["recall_delta"])),
                "少误杀": int(row["old_only_false_positives_removed"]),
                "漏召回": int(row["old_only_true_positives_lost"]),
                "新增正确": int(row["new_only_correct_catches"]),
                "新增误杀": int(row["new_only_false_positives_added"]),
            }
        )

    changed_count = int((df[args.old_col] != df[args.new_col]).sum())
    old_correct = df[args.old_col] == df[args.human_col]
    new_correct = df[args.new_col] == df[args.human_col]
    both_correct = int((old_correct & new_correct).sum())
    new_fixed = int((~old_correct & new_correct).sum())
    new_regressed = int((old_correct & ~new_correct).sum())
    both_wrong = int((~old_correct & ~new_correct).sum())

    transition_rows = transition_summary.head(12).to_dict("records")
    for row in transition_rows:
        row["count"] = int(row["count"])

    blocker_text = "\n".join(f"- {item}" for item in blockers) if blockers else "- 未触发主要替换护栏"

    return f"""# 新旧机审模型替换评估报告

## 1. 数据与口径

- 数据源：`{source_name}`
- 样本量：{len(df)}
- 人工风险样本：{int((df[args.human_col] != args.negative_label).sum())}
- 人工通过样本：{int((df[args.human_col] == args.negative_label).sum())}
- 风险标签：{", ".join(labels)}
- 负类/通过标签：`{args.negative_label}`

> 指标口径：对每个风险标签按 one-vs-rest 计算 TP/FP/FN、精准率、召回率和 F1；“精确标签精准率”表示模型命中的风险标签中，与人审标签完全一致的比例；“精确标签召回率”表示人工风险样本中，被模型以相同风险标签命中的比例。

## 2. 核心结论

{recommendation}

主要原因：
{blocker_text}

## 3. 整体指标

{markdown_table(key_rows, ["指标", "旧模型", "新模型", "变化"])}

新旧模型预测发生变化的样本有 {changed_count} 条。按精确标签是否命中人审看：

{markdown_table([
    {"类型": "新旧均正确", "样本数": both_correct},
    {"类型": "旧错新对", "样本数": new_fixed},
    {"类型": "旧对新错", "样本数": new_regressed},
    {"类型": "新旧均错", "样本数": both_wrong},
], ["类型", "样本数"])}

## 4. 分标签表现

{markdown_table(label_rows, ["标签", "人审量", "旧P", "新P", "P变化", "旧R", "新R", "R变化", "少误杀", "漏召回", "新增正确", "新增误杀"])}

## 5. 主要迁移路径

{markdown_table(transition_rows, ["old_label", "new_label", "human_label", "count"])}

## 6. 建议

1. 不做全量无兜底替换。新模型在当前样本中显著减少误杀、提升整体准确率和命中精准率，但代价是部分标签召回大幅下降。
2. 可优先考虑对 `坦克`、`国家领土`、`极端服饰` 做灰度替换或“新模型通过时降级人审”策略，因为这些标签的召回下降较小或有提升，但仍需监控漏召回。
3. `敏感制服`、`敏感事件集会`、`人民大会堂` 不建议直接替换；当前样本下新模型几乎不再命中这些标签，应保留旧模型兜底或补充训练后复测。
4. 由于旧模型在该 CSV 中没有 `通过` 预测，样本很可能来自旧模型命中池；如要判断线上全量替换，需要再补充未命中/背景流量的人审样本，验证新模型对旧模型未命中风险的召回。

## 7. 输出文件

- `summary_metrics.csv`：整体指标
- `label_metrics.csv`：新旧模型分标签 TP/FP/FN、P/R/F1
- `label_delta.csv`：分标签差异、少误杀、漏召回、新增正确和新增误杀
- `confusion_old.csv`、`confusion_new.csv`：新旧模型混淆矩阵
- `old_new_human_transition.csv`：旧模型 -> 新模型 -> 人审标签迁移
- `regression_samples.csv`：旧模型正确但新模型错误样本
- `improvement_samples.csv`：旧模型错误但新模型正确样本
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = [args.human_col, args.old_col, args.new_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    for col in set(required + [args.id_col]):
        if col in df.columns:
            df[col] = df[col].map(clean_label)

    labels = unique_labels(df, [args.human_col, args.old_col, args.new_col], args.negative_label)
    old_metrics = label_metrics(df, "old", args.old_col, args.human_col, labels)
    new_metrics = label_metrics(df, "new", args.new_col, args.human_col, labels)
    all_label_metrics = pd.concat([old_metrics, new_metrics], ignore_index=True)

    old_summary = summary_metrics(df, "old", args.old_col, args.human_col, labels, args.negative_label, old_metrics)
    new_summary = summary_metrics(df, "new", args.new_col, args.human_col, labels, args.negative_label, new_metrics)
    summaries = pd.DataFrame([old_summary, new_summary])

    label_delta = build_label_delta(
        df,
        old_metrics,
        new_metrics,
        labels,
        args.human_col,
        args.old_col,
        args.new_col,
    )
    summary_delta = {
        "precision_delta": float(new_summary["hit_precision_exact_label"] - old_summary["hit_precision_exact_label"]),
        "risk_recall_delta": float(new_summary["risk_recall_exact_label"] - old_summary["risk_recall_exact_label"]),
        "accuracy_delta": float(new_summary["overall_accuracy_including_pass"] - old_summary["overall_accuracy_including_pass"]),
    }

    confusion_old = pd.crosstab(df[args.human_col], df[args.old_col], rownames=["human_label"], colnames=["old_label"])
    confusion_new = pd.crosstab(df[args.human_col], df[args.new_col], rownames=["human_label"], colnames=["new_label"])
    transition = (
        df.groupby([args.old_col, args.new_col, args.human_col])
        .size()
        .reset_index(name="count")
        .rename(columns={args.old_col: "old_label", args.new_col: "new_label", args.human_col: "human_label"})
        .sort_values("count", ascending=False)
    )
    old_new_transition = (
        df.groupby([args.old_col, args.new_col])
        .size()
        .reset_index(name="count")
        .rename(columns={args.old_col: "old_label", args.new_col: "new_label"})
        .sort_values("count", ascending=False)
    )

    old_correct = df[args.old_col] == df[args.human_col]
    new_correct = df[args.new_col] == df[args.human_col]
    regression_samples = df[old_correct & ~new_correct].copy()
    improvement_samples = df[~old_correct & new_correct].copy()
    changed_samples = df[df[args.old_col] != df[args.new_col]].copy()

    write_csv(summaries, output_dir / "summary_metrics.csv")
    write_csv(all_label_metrics, output_dir / "label_metrics.csv")
    write_csv(label_delta, output_dir / "label_delta.csv")
    write_csv(confusion_old, output_dir / "confusion_old.csv")
    write_csv(confusion_new, output_dir / "confusion_new.csv")
    write_csv(transition, output_dir / "old_new_human_transition.csv")
    write_csv(old_new_transition, output_dir / "old_new_transition.csv")
    write_csv(regression_samples, output_dir / "regression_samples.csv")
    write_csv(improvement_samples, output_dir / "improvement_samples.csv")
    write_csv(changed_samples, output_dir / "changed_samples.csv")

    old_has_negative_predictions = bool((df[args.old_col] == args.negative_label).any())
    recommendation, blockers = decision_text(summary_delta, label_delta, args, old_has_negative_predictions)
    report = build_report(
        output_dir,
        input_path.name,
        df,
        labels,
        summaries,
        label_delta,
        transition,
        args,
        recommendation,
        blockers,
    )
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote report: {output_dir / 'model_report.md'}")
    print(f"Recommendation: {recommendation}")
    print(
        "Overall exact-label precision delta: "
        f"{signed_pct(summary_delta['precision_delta'])}; "
        f"recall delta: {signed_pct(summary_delta['risk_recall_delta'])}"
    )


if __name__ == "__main__":
    main()
