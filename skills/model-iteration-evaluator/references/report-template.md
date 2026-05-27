# Report Template

Use this structure for `model_report.md` or an equivalent user-facing Markdown report.

## 1. 数据与口径

- 数据源
- 样本量
- 人工风险样本数和通过样本数
- 风险标签列表
- 负类/通过标签
- 指标口径说明

## 2. 核心结论

Start with a direct rollout decision:

- `可以进入灰度/替换`
- `不建议直接全量替换`
- `仅建议部分标签替换`

Then list the main blockers or supporting evidence.

## 3. 整体指标

Include at least:

- 风险命中量
- 精确标签精准率
- 精确标签召回率
- 含通过整体准确率
- 二分类风险召回

Show old value, new value, and delta.

## 4. 分标签表现

For each risk label, show:

- 人审量
- old/new precision and delta
- old/new recall and delta
- 少误杀
- 漏召回
- 新增正确
- 新增误杀

Order labels by human support unless a high-risk priority list is provided.

## 5. 主要迁移路径

Show the largest `old_label -> new_label -> human_label` paths. Use this section to make the business tradeoff concrete, such as "new model changed old risk hits to pass and removed many false positives, but also lost true positives."

## 6. 建议

Use decisive, operational recommendations:

1. Whether to replace fully, gray-release, or keep old-model fallback.
2. Which labels can be considered for rollout.
3. Which labels need fallback, human review, or retraining.
4. What extra data is required if the sample is biased.

## 7. 输出文件

List generated report and CSV artifacts. Keep paths relative to the output directory inside the report; use absolute Markdown links in the final chat response.
