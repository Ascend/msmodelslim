# msModelSlim 快速入门

## 1. 概述

msModelSlim 是面向昇腾生态的模型压缩工具，覆盖稠密 LLM、MoE 及多模态模型的量化与压缩。本文以 Qwen3.6-27B 为例，带你体验如何通过一键量化将模型权重压缩为 W8A8 格式，并基于vLLM-Ascend完成推理部署验证。

**体验地图（核心操作约 10 分钟，不含镜像与模型下载等网络传输时间）**

| 步骤 | 环节 | 核心工具           |       操作耗时       | 原理学习 |
|:--:|:-------|:---------------|:----------------:|:-----:|
| 1 | 容器环境准备 | vLLM-Ascend 容器 | 约 1 分钟（不含镜像下载时间） | 5 分钟 |
| 2 | 模型文件准备 | modelscope     | 约 1 分钟（不含模型下载时间） | 2 分钟 |
| 3 | 模型量化 | msModelSlim    |      约 3 分钟      | 5 分钟 |
| 4 | 量化结果验证 | vLLM-Ascend    |      约 5 分钟      | 10 分钟 |

## 2. 操作步骤

### 2.1 环境准备（必做）

🛑 **本节为强制前置步骤！跳过本节可能导致后续多项操作失败。**

本教程**仅支持**在标准化 vLLM-Ascend 容器中执行，不支持直接在裸机、虚拟机或其他非标准容器环境中运行。

#### 2.1.1 前置条件

开始前，请确认服务器满足以下要求：

| 项目       | 要求                                           | 验证方法                              |
|----------|----------------------------------------------|-----------------------------------|
| **硬件算力** | Linux 服务器配备至少 2 张 NPU 卡（A2 或 A3 系列），驱动与固件已安装 | 执行 `npu-smi info`，确认 NPU 卡状态正常    |
| **容器运行** | 已安装并运行 Docker（建议版本 ≥ 18.0）                   | 执行 `docker ps`，无报错即表示服务正常启动       |
| **脚本执行** | 宿主机已安装 Python 3（任意版本）                        | 在宿主机执行 `python3 -V`，有版本信息输出即表示已安装 |
| **网络通信** | 已安装 curl（任意版本）                               | 执行 `curl -V`，有版本信息输出即表示已安装        |
| **磁盘空间** | 至少 100GB 空闲磁盘空间（用于模型权重下载）                    | 执行 `df -h`，查看磁盘空间使用情况                     |

> 👉 确认前置条件满足后，若环境具备公网访问能力，则本章所有命令可直接 **Copy/Paste** 执行，无需手动输入或拼接。

#### 2.1.2 宿主机：自动识别并配置镜像环境变量

在宿主机执行以下命令（该命令依次完成：读取 NPU PCI ID，匹配镜像版本，写入环境变量供后续流程使用）：

```bash
dev_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2)
source /dev/stdin <<< "$(
  case "$dev_id" in
    'd802' )
      echo 'export MY_STUDY_VAR_VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.18.0"'
      echo 'echo -e "\e[32m[PASS] Successfully auto-selected image: $MY_STUDY_VAR_VLLM_IMAGE\e[0m"'
      ;;
    'd803' )
      echo 'export MY_STUDY_VAR_VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.18.0-a3"'
      echo 'echo -e "\e[32m[PASS] Successfully auto-selected image: $MY_STUDY_VAR_VLLM_IMAGE\e[0m"'
      ;;
    * )
      echo 'unset MY_STUDY_VAR_VLLM_IMAGE'
      echo 'echo -e "\033[31m[FAIL] Get device ID: '"$dev_id"'. Learning is not supported in the current environment.\033[0m" >&2'
      ;;
  esac
)"
```

> [!NOTE]说明
>
> **命令原理**
>
> 通过 `lspci` 获取 NPU 的 PCI ID，自动匹配 vLLM-Ascend 官方镜像，并将镜像地址赋给环境变量 `MY_STUDY_VAR_VLLM_IMAGE`，供后续使用。  
> 所有镜像均来自 Quay.io 发布的 vLLM-Ascend 官方仓库，镜像详情参阅 [vLLM-Ascend 官方镜像仓库](https://quay.io/repository/ascend/vllm-ascend?tab=tags)。

若输出 `[PASS]`，表示识别成功，继续下一步；若输出 `[FAIL]`，可能原因如下：

1. 硬件不在支持范围内：本教程仅支持昇腾 A2 和 A3 系列，请切换至兼容硬件后重试；
2. 底层环境异常：未安装 `lspci`，或当前用户无权执行 `lspci -n -D`，请联系环境管理员确认。

#### 2.1.3 宿主机：拉取镜像

在宿主机执行：

```bash
docker pull ${MY_STUDY_VAR_VLLM_IMAGE}
```

若因企业内网限制导致拉取失败，请参考 [第 3.1 节](#31-docker-镜像在隔离内网的获取方法)。

#### 2.1.4 宿主机：下载容器启动脚本

在宿主机执行：

```bash
cd ~ && curl -fLO --retry 3 https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py && chmod +x ctr_in.py
```

若因网络限制无法下载，请参考 [第 3.2 节](#32-传输容器启动脚本)。

#### 2.1.5 宿主机：启动容器

在宿主机执行以下命令，终端将显示容器创建信息并等待确认，直接回车即可完成创建：

```bash
~/ctr_in.py ${MY_STUDY_VAR_VLLM_IMAGE}
```

**预期结果**：

等待约 10 秒，终端显示如下 root Shell 提示符，表示容器已成功启动：

```text
[root@xxxxxx ~]#
```

若出现报错或容器选择界面，请返回 [第 2.1.2 节](#212-宿主机自动识别并配置镜像环境变量) 确认输出 `[PASS]` 后重试。

#### 2.1.6 容器内：安装 msModelSlim

进入容器后，安装 msModelSlim 及所需的 transformers 版本：

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ \
    transformers==5.2.0 \
    https://gitcode.com/Ascend/msmodelslim/releases/download/tag_MindStudio_26.1.0.B100_002/msmodelslim-26.1.0-py3-none-any.whl
```

> [!NOTE]说明
>
> **关于 transformers 的版本选择说明**
>
> transformers 版本取决于所量化的模型。本例中使用的 Qwen3.6-27B 模型需在 transformers==5.2.0 环境下运行。

若因企业内网限制导致安装失败，请参考 [第 3.3 节](#33-离线安装-python-依赖)。

#### 2.1.7 容器内：检查环境安装正确性

安装完成后，执行一键验证：

```bash
python3 -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU is unavailable"; print("PyTorch:", torch.__version__)' && msmodelslim --help >/dev/null && echo -e "\e[32m[PASS] NPU environment and msmodelslim check passed.\e[0m"
```

若输出 `[PASS]`，表示 NPU 驱动、PyTorch 及 msModelSlim 均已就绪，环境准备完成，进入量化阶段。

### 2.2 执行量化

#### 2.2.1 容器内：准备模型文件

> [!NOTE]说明
>
> **高效操作小技巧**
>
> 模型文件较大（约 50GB），即使千兆带宽满速下载也需10分钟左右。建议执行下载命令后先阅读后续章节，学习量化原理与部署流程，待下载完成后即可更高效地执行后续操作。

执行如下命令，从 ModelScope 下载 Qwen3.6-27B 原始权重：

```bash
modelscope download --model Qwen/Qwen3.6-27B --local_dir ~/qwen36_27b_base
```

#### 2.2.2 容器内：准备 NPU 卡

量化过程需要 NPU 算力加速，需确保至少有一张空闲 NPU 卡。执行以下命令自动选择一张空闲卡：

```bash
free_npu=$(npu-smi info | grep -oE "No running processes found in NPU\s+[0-9]+" | head -n 1 | awk '{print $NF}')
if [ -n "$free_npu" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=$free_npu
    echo -e "\e[32m[PASS] Successfully exported ASCEND_RT_VISIBLE_DEVICES=$free_npu\e[0m"
else
    echo -e "\e[31m[FAIL] All NPUs are busy. Please release NPUs and try again.\e[0m" >&2
fi
```

若输出 `[PASS]`，表示已成功指定空闲 NPU 卡，可进入下一步；若输出 `[FAIL]`，请先释放 NPU 资源后重试上述命令。

> [!NOTE]说明
>
> **关于 NPU 卡选择机制**
>
> **功能**：环境变量 `ASCEND_RT_VISIBLE_DEVICES` 指定当前进程可见的 NPU ID（支持单个或多个），无需修改代码即可切换设备。
>
> **索引映射规则**：
> 
> 设置该变量后，可见设备的**逻辑索引将从 0 开始重新编号**，后续操作需使用新索引而非原始 NPU ID。
> 
> - `=1`：仅 NPU 1 可见，其新索引为 **0**。
> - `=1,2,3`：NPU 1/2/3 可见，新索引依次为 **0、1、2**。
>
> ⚠️ **注意**：该环境变量为试用特性，后续版本可能变更，请勿用于生产环境。

#### 2.2.3 容器内：执行模型量化

执行以下命令，使用一键量化功能。系统将自动匹配该模型的最佳实践配置，以 W8A8（将模型权重和激活均量化为 8-bit）模式完成量化：

```bash
msmodelslim quant --model_path ~/qwen36_27b_base --save_path ~/qwen36_27b_w8a8 --device npu --model_type Qwen3.6-27B --quant_type w8a8 --trust_remote_code true
```

量化耗时约 4 分钟，出现如下输出即表示完成：

```text
msmodelslim.app.naive_quantization - INFO - ===========SUCCESS===========
```

若量化中止或报错，未出现上述成功标志，请按以下步骤排查：

1. 确认 NPU 卡状态正常：执行 `npu-smi info`，检查目标卡的 Health 状态是否为 `OK`、AI Core 占用率是否异常，若异常需先释放资源或更换卡。
2. 检查环境变量设置：执行 `echo ${ASCEND_RT_VISIBLE_DEVICES}`，确认该变量指向的卡 ID 真实存在且未被占用，不可包含空格或无效 ID。
3. 排查显存不足（OOM）：查看终端错误日志中是否包含类似 `Out of memory` 的关键字。若出现，说明当前卡显存不足，可切换至空闲 NPU 卡重试。

#### 2.2.4 容器内：查看量化输出结果

**1. 查看量化结果文件**

```bash
ls -al ~/qwen36_27b_w8a8
```

输出目录结构类似如下，其中标记【量化】的为量化产出文件，标记【原始】的为从原始模型复制的推理配置文件（仅列出主要文件，非完整列表）：

```text
~/qwen36_27b_w8a8/
├── Qwen3.6-27B_best_practice.yaml                  # 【量化】量化配置协议文件（记录本次量化的完整配置信息，可用于方案复现）
├── quant_model_description.json                    # 【量化】量化权重描述文件（记录每个权重张量的量化类型与元数据，是推理框架加载量化模型的重要依据）
├── quant_model_weights-00001-of-00009.safetensors  # 【量化】量化权重分片 1/9（INT8 量化后的模型权重数据，共 9 个分片）
├── ...                                             # 【量化】其余权重分片（00002 ~ 00008）
├── quant_model_weights-00009-of-00009.safetensors  # 【量化】量化权重分片 9/9（INT8 量化后的模型权重数据）
├── config.json                                     # 【原始】模型配置文件（定义网络架构、层数、隐藏层维度等结构超参数）
├── tokenizer_config.json                           # 【原始】分词器配置文件（定义特殊 token、词表大小及文本预处理逻辑）
├── tokenizer.json                                  # 【原始】分词器词汇表（定义 token 与 ID 的映射关系）
├── chat_template.jinja                             # 【原始】对话模板（定义多轮对话的提示词拼接格式）
└── generation_config.json                          # 【原始】生成配置文件（定义温度、Top-P、最大生成长度等采样策略）
```

**2. 验证量化压缩效果**

执行如下命令，对比量化前后的目录大小：

```bash
du -sh ~/qwen36_27b_base
du -sh ~/qwen36_27b_w8a8
```

预期结果：原始权重约50+GB，量化后约30+GB，体积缩减约40%，表明量化大幅压缩了模型体积。

### 2.3 量化模型功能验证

本节使用 vLLM-Ascend 部署量化模型并完成一次推理验证。

#### 2.3.1 容器内：恢复 vLLM 运行环境

msModelSlim 量化阶段需要 transformers 5.x，而 vLLM 运行时需要 transformers 4.x，因此推理前需降级恢复为镜像中原来的版本：

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ transformers==4.57.6
```

#### 2.3.2 容器内：准备 NPU 卡

推理服务需要使用 2 张卡做张量并行，执行如下命令，自动选择 2 张空闲卡：

```bash
# 1. 获取最多 2 张空闲卡的编号
free_npus_raw=$(npu-smi info | grep -oE "No running processes found in NPU\s+[0-9]+" | head -n 2 | awk '{print $NF}')
npu_count=$(echo "$free_npus_raw" | wc -w)
# 2. 判断是否满足 2 张卡的需求并设置控制环境变量；如果只有 1 张卡，或者完全没有空闲卡，都会报错
if [ "$npu_count" -eq 2 ]; then
    export_val=$(echo "$free_npus_raw" | paste -s -d ',')
    export ASCEND_RT_VISIBLE_DEVICES=$export_val
    echo -e "\e[32m[PASS] Successfully exported ASCEND_RT_VISIBLE_DEVICES=$export_val\e[0m"
else
    echo -e "\e[31m[FAIL] Insufficient free NPUs (Found $npu_count, Need 2). Please release NPUs and try again.\e[0m" >&2
fi
```

若输出 `[PASS]`，表示已自动选择空闲 NPU 卡，可继续执行下一步；若输出 `[FAIL]`，请先释放 NPU 资源后重试上述命令。

#### 2.3.3 容器内：启动服务

启动 vLLM-Ascend 在线推理服务（命令会持续占用当前终端）：

```bash
vllm serve ~/qwen36_27b_w8a8 \
    --port 5678 \
    --served-model-name Qwen3.6-27B-W8A8 \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

> [!NOTE]说明
>
> **知识点（可选阅读）：vLLM 主要启动参数说明**
>
> - `--quantization ascend`：指定使用 Ascend 量化推理后端，加载 msModelSlim 生成的 W8A8 权重。
> - `--served-model-name`：对外暴露的模型名称，需与客户端请求中的 `model` 字段一致。
> - `--tensor-parallel-size 2`：张量并行度，将模型切分到 2 张 NPU 卡上。
> - `--max-model-len 8192`：最大序列长度（Token 数），超出此长度的请求将被拒绝。
> - `--compilation-config`：启用 FULL_DECODE_ONLY 图模式，将 Decode 阶段编译为静态图加速推理。
> - `--additional-config`：启用 CPU-NPU NUMA 亲和绑定，降低 Host-Device 通信延迟。

启动约需 4~5 分钟，输出日志较多，主要耗时阶段如下（耗时为某次实测值，仅供参考）：

| 序号 | 阶段 | 耗时     | 说明 |
|:--:|:--|:-------|:--|
| 1  | 配置解析与插件激活 | 约 10 秒 | 加载 vLLM-Ascend 平台插件，解析模型架构与调度参数 |
| 2  | Worker 启动与 HCCL 握手 | 约 50 秒 | 拉起多卡 Worker 进程，建立通信链路，分配 TP Rank |
| 3  | CPU-NPU 亲和绑定 | 约 10 秒 | 按 NUMA 拓扑将 Worker 绑定至 NPU 近端 CPU 核心及中断 |
| 4  | 加载模型权重 | 约 30 秒 | 将 9 个 safetensors 分片（每卡约 16.7GB）加载至全局内存 |
| 5  | 图编译与算子融合 | 约 80 秒 | Dynamo 字节码转换（20s）+ CANN 算子编译（48s）+ 融合预热 |
| 6  | NPU Graph 捕获 | 约 30 秒 | 预编译 22 种 Batch 大小（1~152）的静态执行路径 |

当出现类似如下日志时，表示服务启动成功：

```text
(APIServer pid=6036) INFO:     Started server process [6036]
(APIServer pid=6036) INFO:     Waiting for application startup.
(APIServer pid=6036) INFO:     Application startup complete.
```

> 启动过程中可能会有一些 Warning 日志，只要出现上述成功日志即可忽略，具体原因参考 [FAQ 4.3](#43-vllm-启动时出现-warning-日志是否正常)。

若服务启动中止，未出现上述成功日志，请根据终端输出的错误信息定位问题。常见错误及处理方式：

1. 端口被占用：错误信息包含 `Address already in use` 或 `bind: address already in use`。请通过 `--port` 更换端口，或中止使用此端口的进程。
2. 显存不足（OOM）：错误信息包含类似 `Out of memory`。可通过 `npu-smi info` 确认目标卡显存占用情况，通过 `ASCEND_RT_VISIBLE_DEVICES` 切换至空闲卡。

#### 2.3.4 推理验证

服务启动成功后，因 vLLM 服务会持续占用当前终端，请新开一个宿主机终端（是否进入容器内执行均可，网络都是通的）：

**第一步：发送预热请求**

```bash
curl -s http://localhost:5678/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-27B-W8A8", "prompt": "This is a warm-up request.", "max_tokens": 256}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

首次响应可能较慢或返回乱码，等待返回信息即可，返回内容可忽略。

> [!NOTE]说明
>
> **知识点（可选阅读）：首次请求耗时较长或返回乱码的原因**
>
> 服务启动后首次推理会触发若干一次性初始化操作，导致首次推理耗时较长或返回乱码：
>
> 1. NPU Graph 首次执行：静态图路径在首次实际推理时才真正走通完整数据流，日志中可见 `Replaying aclgraph` 提示；
> 2. Triton 算子 JIT 编译：FlashAttention 等算子首次执行时进行动态编译与自动调优（Autotuning），产生数秒延迟；
> 3. KV Cache 脏数据：全局内存预分配后未逐字节清零，首次注意力计算可能读取到无效数据，导致输出乱码。
>
> 业界通常采用”**预热（Warmup）**”机制应对此类问题，即服务启动后先发送一条测试请求并丢弃其结果，待系统完成初始化后再接入正式流量。

**第二步：发送正式推理请求**：

```bash
curl -s http://localhost:5678/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-W8A8", "prompt":"Write a Python function to calculate the Fibonacci sequence.", "max_tokens":256}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

等待推理完成，若返回的信息中包括写代码的思考过程或有代码输出即表明推理成功

### 2.4 清理资源

#### 2.4.1 停止推理服务，释放 NPU 卡资源

在 vLLM 启动终端中按 `Ctrl+C` 停止服务，释放占用的 NPU 卡资源。执行 `npu-smi info` 确认 NPU 卡已无异常占用，必要时手动 kill 残留进程。

#### 2.4.2 删除容器，释放磁盘空间（可选）

若不再需要本教程的容器环境，在**宿主机**执行以下命令，选择目标容器进行删除以释放磁盘空间：

```bash
~/ctr_in.py -d
```

🎉 至此，快速入门体验已全部完成。已完成 msModelSlim 一键量化与 vLLM-Ascend 推理部署的完整流程，如需了解更多功能，请阅读 《[使用指南](../user_guide/README.md)》 等进阶文档。

<br>

## 3. 附录：内网环境无公网访问权限的应对方案

### 3.1 Docker 镜像在隔离内网的获取方法

**方案一：配置 Docker 代理直接拉取**

适用于 Docker 版本 ≥ 18.0 的大多数 Linux 发行版（不保证所有场景兼容，若遇异常请结合实际调整）。

编辑 Docker 服务代理配置文件 `/etc/systemd/system/docker.service.d/http-proxy.conf`（请根据实际环境替换用户名、密码、代理地址及端口）：

```text
[Service]
Environment="HTTP_PROXY=http://username:password@proxy.example.com:8080"
Environment="HTTPS_PROXY=http://username:password@proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

保存后重载并重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

随后即可正常执行 `docker pull`。

**方案二：离线导入镜像**

若代理方案不可行，请先在内网 NPU 服务器上执行 [第 2.1.2 节](#212-宿主机自动识别并配置镜像环境变量)，记录 `MY_STUDY_VAR_VLLM_IMAGE` 的完整值。然后在一台具备公网访问能力且 CPU 架构相同的中转机上执行：

```bash
VLLM_IMAGE='完整镜像地址'   # 替换为 MY_STUDY_VAR_VLLM_IMAGE 的值
docker pull "${VLLM_IMAGE}"
docker save -o vllm-ascend.tar "${VLLM_IMAGE}"
```

将 `vllm-ascend.tar` 通过 U 盘等方式传输至内网服务器后加载：

```bash
docker load -i vllm-ascend.tar
docker images | grep vllm-ascend
```

加载完成后，继续 [第 3.2 节](#32-传输容器启动脚本) 传输启动脚本，再返回 [第 2.1.5 节](#215-宿主机启动容器) 启动容器。若已切换宿主机 Shell 会话，需重新执行第 2.1.2 节恢复环境变量。

### 3.2 传输容器启动脚本

在可访问公网的浏览器中打开以下链接，下载 `ctr_in.py` 脚本，并将其传输至内网服务器的 `~/` 目录：

```text
https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py
```

在内网服务器宿主机上执行：

```bash
cd ~
chmod +x ctr_in.py
ls -l ctr_in.py
```

确认文件存在且具有执行权限后，返回 [第 2.1.5 节](#215-宿主机启动容器) 启动容器。

### 3.3 离线安装 Python 依赖

优先使用内网 pip 源安装依赖。若没有可用的内网软件源，请在具备公网访问能力、与内网 NPU 服务器的 CPU 架构和 Python 版本均相同的中转环境中，按以下方式下载所需安装包：

```bash
mkdir -p offline_wheels
python3 -m pip download <package_name> --dest offline_wheels
```

将 `offline_wheels` 目录传输到内网服务器并复制到容器的用户主目录，然后在容器内执行：

```bash
pip3 install --no-index --find-links="${HOME}/offline_wheels" <package_name>
```

安装完成后，返回 [第 2.1.7 节](#217-容器内检查环境安装正确性) 执行验证命令，无需再次执行联网安装命令。

## 4. 常见问题（FAQ）

### 4.1 退出容器后如何重新进入？

在宿主机上选择以下任一方法：

**方法一（推荐）：使用容器启动脚本**

```bash
~/ctr_in.py
```

根据提示选择目标容器；若仅有一个运行中的容器，脚本会自动进入。

**方法二：使用 Docker 原生命令**

```bash
docker exec -it alice_YYMMDD_HHMMSS bash
```

将 `alice_YYMMDD_HHMMSS` 替换为实际容器名称，可先执行 `docker ps` 查看。

### 4.2 执行 Docker 命令时提示 permission denied 如何处理？

当前用户可能未加入 Docker 用户组。在宿主机以 root 权限执行：

```bash
sudo usermod -aG docker "${USER}"
```

执行后需退出当前会话并重新登录，或执行 `newgrp docker` 使用户组变更立即生效。完成后执行 `docker ps` 验证。

> 注意：Docker 用户组具有较高系统权限，请仅将可信用户加入该组，不建议日常以 root 身份操作。

### 4.3 vLLM 启动时出现 WARNING 日志是否正常？

若最终输出 `Application startup complete`，则启动过程中的 WARNING 日志均不影响功能，主要包含以下几类：

1. **GPU 专属参数重置**：`--disable-cascade-attn`、`--disable-flashinfer-prefill` 等参数仅适用于 NVIDIA GPU，vLLM 在 Ascend 环境下会自动将其重置为 `False` 并忽略。
2. **FULL_DECODE_ONLY 图模式风险提示**：该模式处于实验阶段，提示过多 Batch 捕获可能导致显存不足。若最终输出 `Application startup complete`，即表明图捕获成功，服务可正常使用。
3. **CUDA Graph 捕获限制**：提示 `Capping cudagraph capture sizes`，表示系统根据可用 Mamba 缓存块自动调整了最大捕获 Batch 大小，属于正常适配行为。
4. **Gloo 通信回退**：提示 `Unable to resolve hostname`，表示 Gloo 通信库无法解析主机名，已自动回退至 loopback 地址，不影响单机多卡推理。
