<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisProcessorConfig -->
# unary_analysis 配置说明

## 1. 配置概述

一元（无量化）敏感度分析处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `UnaryAnalysisProcessorConfig` |
| 源码 | [processor.py](../../../../../msmodelslim/processor/analysis/unary_operator/processor.py) |

## 2. 参数列表

<h3 id="2-1-unary-analysis">2.1 UnaryAnalysisProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `unary_analysis` | `unary_analysis` | 处理器类型，固定为 `unary_analysis`。 | 无 |
| `metrics` | `string` | 可选 | `kurtosis` | — | 分析指标：`quantile`（分位数）、`std`（标准差）、`kurtosis`（峰度）、`ra_compress`（RA Compress 长序列压缩头筛选） | 无 |
| `patterns` | `list[string]` | 可选 | `['*']` | — | 待分析的层名模式列表，默认 `*` 匹配全部 | 无 |
| `metric_params` | `object` | 可选 | `{}` | — | 指标专属超参，仅由 `metrics` 指定的分析方法解析。当前仅 `metrics=ra_compress` 支持：`induction_head_ratio`（默认 0.14）、`echo_head_ratio`（默认 0.01），取值范围均为 [0, 1]。 | 无 |

**配置约束**

- 校验指标专属超参，非法使用直接报错。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: unary_analysis
    metrics: kurtosis
    patterns:
    - '*'
    metric_params: {}
```

## 4. 注意事项

- `metric_params` 仅对 `metrics=ra_compress` 生效，其它指标传入会直接报错；该字段只在配置中设置，CLI 不暴露对应参数，`msmodelslim analyze attn_head` 按默认值（`induction_head_ratio=0.14`、`echo_head_ratio=0.01`）执行。
- 比例越大，入选（保留）的 KV head 越多，KV Cache 压缩强度越低、精度风险越小、显存收益越小；比例越小则相反，建议按筛选头数与下游压缩比实测评估。
- 命令行流程读取随包安装的模板 `msmodelslim/core/analysis_service/pipeline_analysis/pipeline_template/ra_compress.yaml`；修改该 YAML 后，需在仓库根目录重新执行 `bash install.sh` 才会生效。

`ra_compress` 覆盖默认比例：

```yaml
process:
- type: unary_analysis
  metrics: ra_compress
  metric_params:
    induction_head_ratio: 0.14
    echo_head_ratio: 0.01
```
