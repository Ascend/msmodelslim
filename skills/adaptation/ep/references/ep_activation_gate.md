# EP 激活值一致性门禁（EP Check 7：单卡量化 vs 多卡 EP 量化）

EP Check 1~6 全部是**结构性**检查：分片是否连续、本地专家数对不对、非本地专家是否驻留、
mapping 是否只访问本地专家、日志有没有 `[EP_CHECK]`。它们能证明「EP 改造的表面形态正确」，
但无法证明「EP 并行下的量化结果与单卡量化一致」。

本门禁补上这一步，对比对象是：

- **基准**：单卡量化模型（先跑单卡量化，目标量化配置逐层激活作为锚点存档）；
- **候选**：多卡 EP 量化模型（相同量化配置，EP 并行）。

用**同一份输入**分别过两个模型，逐层收集 decoder 输出 hidden_states，
以**余弦相似度**为主指标（幅度比为辅助护栏），判定 EP 量化路径（量化 mapping 本地化、
collective 执行、权重按 rank 加载）是否引入与单卡量化不一致的数值偏差。

---

## 1. 为什么「单卡量化」做基准

- 单卡量化与多卡 EP 量化使用**同一份量化配置**（同一 YAML，如 W8A8），

  唯一差异是并行方式：单卡全量专家 vs 多卡专家分片 + collective 合并。

- 若 EP 量化与单卡量化逐层激活高度一致，说明 EP 重组（router → 本地专家 →

  all_reduce → 切回 token）与量化映射本地化均未改变量化结果语义，
  后续调优的精度结论在 EP 并行下可信、可与单卡结论互相印证。

- 若不一致，优先怀疑不是量化算法，而是 EP 路径缺陷：

  量化 mapping 未按 rank 本地化（仍在全量专家上注册）、
  `all_reduce` 翻倍/漏做/未平均、权重按 rank 加载错位等。

### 为什么保留幅度护栏

余弦相似度只度量方向，不度量尺度。EP 缺陷（`all_reduce` 多做/少做一次、SUM 未转 MEAN、
DP token 切回重复累加）会把输出整体缩放或翻倍，**余弦相似度仍为 1.0**。
因此本门禁同时输出 `cosine`（主指标）与 `norm_ratio`（幅度护栏，`‖多卡激活‖/‖单卡激活‖`），缺一不可。

| 指标 | 含义 | 缺失风险 |
|---|---|---|
| `cosine`（方向） | 主指标，检测专家选错/路由错乱/量化 mapping 注册错位 | 中等尺度漂移仍可能漏检 |
| `norm_ratio`（幅度） | `‖多卡激活‖ / ‖单卡激活‖`，应接近 1 | 整体缩放/翻倍无法由 cosine 检出 |

---

## 2. 流程与对比对象

```text
第 1 步：单卡量化
        用目标量化配置（如 W8A8）在单卡上量化，保存模型 + 采集逐层激活作为基准锚点
第 2 步：多卡 EP 量化
        相同量化配置，EP 并行（world_size=N）量化，每个 rank 采集同一批锚点激活
第 3 步：对比判定
        同一份固定输入（batch=1、seq_len=1、相同 attention_mask/position_ids/dtype/seed）
        逐层计算「单卡量化激活 vs 多卡 EP 量化激活」余弦相似度 → 出判定
```

- **输入**：同一份固定 token 序列，保证两个模型走完全一样的计算路径，只有并行方式不同。
- **锚点**：每个 decoder layer 的输出 hidden_states（`model.layers.<i>` 输出），

  必要时加 `block_sparse_moe` 输出以把误差定位到 MoE 层。

### 为什么默认 `seq_len=1`

EP forward 在 MoE 末端会按 rank 的 token 区间 `y[:, start_pos:end_pos, :]` 切回本地 token。
`seq_len>1` 时各 rank 持有**不同的 token 切片**，逐层激活无法直接与单卡全序列对齐，
必须在每层额外 `all_gather` 回全序列才能比较。

`seq_len=1` 时该切片退化：rank0 持有唯一 token，其余 rank 经 `gather_variable_shapes` 后
**全部拿到同一条完整 token**，各 rank 逐层 hidden_states 形状一致、数值在 `all_reduce` 后一致。
于是每个 rank 都能直接与单卡基准逐层比较，无需额外通信重排。
`seq_len=1` 仍完整覆盖 EP 数值核心：router 产生全局 expert id → 本地专家只算自己那一份 →
`all_reduce` 合并 partial sum，正好验证「多 rank 部分和之和 == 单卡全量专家之和」。

若需覆盖多 token 的 DP 切分路径，再做一次 `seq_len = world_size` 的**可选增强项**，
此时每层锚点需先 `all_gather` 恢复全序列再比较（见第 6 节）。

---

## 3. 参考实现（Agent 按目标模型适配）

### 3.1 激活收集器

```python
import torch
from torch import nn
from msmodelslim.utils.logging import get_logger

class ActivationCollector:
    """前向 hook 收集指定模块的输出激活，detach 后落 CPU。"""
    def __init__(self, anchor_names):
        self.anchor_names = list(anchor_names)
        self.activations = {}
        self._handles = []

    def _hook(self, name):
        def fn(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().to("cpu")
        return fn

    def attach(self, model):
        for name, module in model.named_modules():
            if name in self.anchor_names:
                self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
```

### 3.2 指标

```python
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    na, nb = a.norm(), b.norm()
    if na == 0.0 and nb == 0.0:
        return 1.0        # 两个全零向量视为方向一致
    if na == 0.0 or nb == 0.0:
        return 0.0
    return (a @ b / (na * nb)).item()

def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    na, nb = a.norm(), b.norm()
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return float("inf")
    return (nb / na).item()
```

### 3.3 比较与判定

```python

# ref: 单卡量化基准 {name: tensor}；rank_acts[i]: 多卡 rank i 的 {name: tensor}
def evaluate_gate(ref, rank_acts, cos_threshold=0.999, norm_tolerance=1e-3):
    failures = []
    min_cos, worst_norm_dev = 1.0, 0.0
    for name, ref_t in ref.items():
        for rank, acts in enumerate(rank_acts):
            c = cosine(ref_t, acts[name])
            r = norm_ratio(ref_t, acts[name])
            min_cos = min(min_cos, c)
            worst_norm_dev = max(worst_norm_dev, abs(r - 1.0))
            if c < cos_threshold or abs(r - 1.0) > norm_tolerance:
                failures.append((name, rank, c, r))
    return min_cos, worst_norm_dev, failures
```

---

## 4. 判定与阈值

| 档位 | cosine | norm_ratio | 判定 |
|---|---|---|---|
| 通过 | `min_cos >= 0.999` | `≤ 1e-3` 偏差 | PASS |
| 复核 | `0.99 <= min_cos < 0.999` | `≤ 1e-3` 偏差 | WARN，需定位首个下降的 layer 并人工确认（见第 6 节） |
| 失败 | `min_cos < 0.99` | 或 `> 1e-3` 偏差 | FAIL |

阈值说明：

- 单卡量化 vs 多卡 EP 量化是**同一量化配置、同一数学表达式**（同权重、同路由、仅求和次序不同），

  理论余弦 ≈ 1.0；`0.999` 给 bf16 累加与 `all_reduce` 重排序留出余量，同时足够敏锐。

- 深模型（62层）的 bf16 误差会逐层累积，晚层 cosine 可自然略降；

  若只有最后几层低于 `0.999` 但高于 `0.99`，结合 `norm_ratio` 仍稳定，进入 WARN 而非直接失败。

- **`norm_ratio` 偏差 > 1e-3 直接 FAIL**，无论 cosine 多高——这是翻倍/减半类缺陷的兜底。
- 两个阈值均为默认值，可在调用参数中覆盖；覆盖时必须记录在交付结果里。

### 输出格式

```text
[EP_ACT_GATE] quant_ref=single_card ep_size=4 anchors=62 min_cos=0.99982 worst_norm_dev=3.2e-6 verdict=PASS
[EP_ACT_GATE] first_diverged_layer=model.layers.17 cos=0.9912
```

失败时逐条输出 `(anchor_name, rank, cos, norm_ratio)`，且 `first_diverged_layer` 标出最早低于阈值的层。

---

## 5. 脚本自验证（金标用例）

LLM 生成的 `cosine` / `norm_ratio` / `evaluate_gate` 脚本本身可能有 bug，
不能靠「我感觉写对了」。唯一硬手段：**让脚本对一批手算可得预期结果的构造输入，给出预期的判定**。
脚本必须先通过这套金标，才允许去测真实模型。

| 用例 | 构造 | 期望判定 | 验证什么 |
|---|---|---|---|
| 全等 | `b = a`（同一向量） | cosine=1.0、norm_dev=0 → **PASS** | 主流程正确 |
| 翻倍 | `b = 2 * a` | cosine=1.0 但 norm_dev=1.0 → **FAIL** | 幅度护栏真能抓翻倍 |
| 错位/噪声 | `b = a` 的随机置换或加大噪声 | cosine 明显 < 0.99 → **FAIL** | 主指标真能抓错位 |
| 双零 | `a = 0, b = 0` | cosine 定义为 1.0 → **PASS**，不 NaN 不崩 | 零向量边界 |
| 一零一非零 | `a = 0, b ≠ 0` | 返回明确值（0 或 inf），不崩 | 零向量边界 |
| 形状不一致 | `a` 与 `b` 维度不同 | 报错或显式对齐，不得静默算错 | reshape 前置校验 |

要点：

- 每个用例的期望结论是手算 / 显然的，不依赖模型自己的判断 → 构成**独立裁判**。
- 金标用例是纯 tensor 运算，不碰 GPU / 模型，秒级跑完，写完脚本立即执行。
- 金标任一用例未达期望 → **脚本本身不可信**，禁止进入真实模型门禁；去修脚本，而非放宽阈值。
- 特别地，「翻倍」用例逼着脚本把 `norm_ratio` 护栏做对：若脚本漏了幅度护栏，

  该用例（方向不变）会静默通过，翻倍缺陷就无法被检出。

- 金标全过后，先在一个**极小真实模型**（如 2 专家 × 2卡）端到端跑一遍，

  此时 local expert 循环行为可直接观察，与 `[EP_CHECK]` 结构日志交叉印证，再放大到大规模模型。

---

## 6. 误差定位与增强项

- **逐层定位**：把 `first_diverged_layer` 作为诊断出口——若某层余弦骤降，优先检查该层

  `block_sparse_moe` 的 local expert 循环 / `all_reduce` / DP token 区间 / 量化 mapping 是否按 rank 本地化。

- **MoE 输出锚点**：在每层 MoE block 输出再加一个 hook，可把「attention 误差」与「EP 专家重组误差」分开。
- **多 token 增强项**（可选，默认关）：`seq_len = world_size` 时每层锚点为各 rank 的本地切片，

  需先 `all_gather`（`DistHelper.gather_variable_shapes`）恢复全序列再与单卡基准比较，
  用于覆盖 DP 切分路径。

- **幅度护栏算法对数值差异更稳**：`norm_ratio` 用全序列 L2 范数比，比逐元素相对误差更抗零值噪声。

---

## 7. 与 EP Check 1~6 的关系

- Check 1~6：结构门禁（分片形态正确），全部通过才允许进入本数值门禁。
- Check 7（本门禁）：数值门禁（单卡量化 vs 多卡 EP 量化激活一致），是 `EP_ADAPT_RESULT=PASS` 的**必要条件**。
- 任一结构检查失败 → 不跑数值门禁，直接 `EP_ADAPT_RESULT=FAIL`。
- 结构全过但数值门禁失败 → `EP_ADAPT_RESULT=FAIL`，并回传 `first_diverged_layer` 供修复。
