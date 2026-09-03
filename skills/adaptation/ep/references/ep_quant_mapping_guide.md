# EP 量化映射适配指南

EP 改造完成后，必须检查 ModelSlim adapter 中所有遍历 routed experts 的代码，
确保只访问本地专家。以下按模块逐一说明。

## 1. IterSmooth（get_adapter_config_for_subgraph）

修改前（遍历全部专家）：

```python
for expert in range(num_experts):
    ...
```

修改后（使用 `_get_expert_range` 限制范围）：

```python
from msmodelslim.model.common.utils import _get_expert_range
expert_start, expert_end = _get_expert_range(config)

for expert in range(expert_start, expert_end):
    # 只构造本地专家的 up-down AdapterConfig
    ...
```

参考：`kimi_k3/model_adapter.py:_ffn_subgraph_configs()` 第 556行。

## 2. QuaRot（get_rotate_map）

修改前：

```python
for i in range(num_experts):
    right_rot[f"model.layers.{idx}.mlp.experts.{i}.gate_proj"] = rot
```

修改后：

```python
expert_start, expert_end = _get_expert_range(config)
for i in range(expert_start, expert_end):
    right_rot[f"model.layers.{idx}.mlp.experts.{i}.gate_proj"] = rot
```

参考：`glm_5/quarot.py:get_rotate_map()` 第 119行。

## 3. LN Fuse（get_ln_fuse_map）

修改前：

```python
for i in range(num_experts):
    ln_linear_map["post_attention_layernorm"].append(
        f"model.layers.{idx}.mlp.experts.{i}.gate_proj"
    )
```

修改后：

```python
expert_start, expert_end = _get_expert_range(config)
for i in range(expert_start, expert_end):
    ln_linear_map["post_attention_layernorm"].append(
        f"model.layers.{idx}.mlp.experts.{i}.gate_proj"
    )
```

参考：`glm_5/quarot.py:get_ln_fuse_map()` 第 55行。

## 4. 权重保存（state_dict）

EP 权重保存需确保只写出**本 rank 的 expert 权重**。若 expert 槽位为 `None`，
PyTorch 的 `state_dict()` 不会包含对应 key，天然满足要求。

但若 adapter 有自定义 `_get_state_dict`（如通过 `get_weight_map` 从 index.json 读取），
则需按 `module.named_parameters()` 过滤：

```python
state_dict = {}
for name, param in module.named_parameters():
    full_name = f"{prefix}.{name}" if prefix else name
    if full_name in weight_map:
        # 只读取本地 expert 的权重（非本地 expert 不会出现在 named_parameters 中）
        state_dict[name] = load_tensor(weight_map[full_name])
```

## 5. 低精度反量化/转换

若模型有自定义的 fp8→bf16、mxfp4→bf16 转换函数，需确保
转换函数只遍历 `module.named_parameters()` 中存在的 expert 参数
（即本地专家），而非遍历 index.json 中的全部 expert key。

## 6. 搜索指引

全局搜索以下关键词确认所有需要修改的遍历点：

```text
range(num_experts)
range(n_routed_experts)
range(self.n_routed_experts)
experts.{i}
experts[i]
experts[str(i)]
```

每处都需要确认：

- 是否在遍历 `routed experts`（而非 shared experts）？
- 是否在访问真实 expert module（而非仅作为 id 计算）？
- 若是，则必须改为 `range(expert_start, expert_end)` 或用 `local_expert_ids` 过滤。

## 7. 不建议的改造方式

- ❌ 在 mapping 中保留 `range(num_experts)` 但运行时跳过：不易维护，易遗漏。
- ❌ 在 mapping 中做 `if expert is None: continue`：正确但效率低（大量无效遍历）。
- ✅ 推荐：直接在 `range(expert_start, expert_end)` 内构造，注册时天然只覆盖本地专家。
