# 项目目录结构总览

msModelSlim项目见证了从小模型到大模型的AI发展历程。

早期支持了小模型剪枝、蒸馏、量化等压缩功能，当大模型初现时，逐步转向大模型量化功能。

msModelSlim支持了广泛使用的SmoothQuant、GPTQ、HQQ等大模型量化算法。

但大模型结构愈发复杂，量化位宽也进入到4比特甚至极低比特时代，单一算法已经无法同时满足性能和精度的双重需求，复合策略成为常态。

另一方面，大模型数量也呈井喷之势，亟需提升工具易用性，提高量化调优效率，减少单个模型的量化耗时。

为了迎接大模型量化的浪潮，msModelSlim总结发展出了分层解耦的大模型量化框架，称为V1框架，而将过去的大模型量化手段归为V0框架。

我们推荐您使用V1框架，并期待您参与V1框架的共建。

msModelSlim V1框架仍在成长过程中，未完全覆盖V0框架能力和小模型压缩功能，仍需要历史代码填补功能缺失。

目前 V0 与 V1 代码共存于同一代码仓。V0 源码集中存放在根目录 `v0/` 下，与 V1 源码目录隔离；安装时由 `setup.py` 展开到原有的安装目录和导入路径。

---

# 仓库顶层目录结构

```text
msmodelslim/
├── config/                # 全局配置目录（V1框架）
├── docs/                  # 项目文档，如安装指南、快速入门、算法说明、功能指南、案例集等
├── example/               # 各模型的量化示例（主要为V0框架量化脚本，但V1量化过程FAQ在此统一管理）
├── lab_calib/             # 量化校准数据集示例，如json、jsonl、图片等（V1框架）
├── lab_practice/          # 各模型最佳实践量化配置（V1框架）
├── msmodelslim/           # V1框架源代码
├── pre-commit/            # Git 预提交检查配置
├── skills/                # Agent Skills（资料管理、量化安装/适配/量化/策略/调优/测评等）
├── test/                  # 测试用例与测试脚本
├── tools/                 # 开发与文档生成工具
├── v0/                    # V0框架和小模型压缩源代码，安装时展开到原有安装目录
│   ├── ascend_utils/      # ascend_utils 包
│   ├── modelslim/         # modelslim 兼容包
│   ├── msmodelslim/       # 安装后展开为 msmodelslim 的 V0 包
│   │   ├── common/
│   │   ├── mindspore/
│   │   ├── onnx/
│   │   ├── pytorch/
│   │   └── quant/
│   ├── precision_tool/    # 伪量化精度评估工具
│   └── security/          # 安全检查
├── install.sh             # 安装脚本
├── requirements.txt       # 第三方依赖列表
├── pytest.ini             # 测试配置
├── README.md              # 项目总览与使用说明
├── LICENSE                # 开源许可证
├── OWNERS                 # 代码维护审批配置
└── setup.py               # 打包与安装配置脚本
```

V1框架涉及目录如下：

```text
msmodelslim/
├── config/                # 全局配置目录
├── docs/                  # 项目文档，如安装指南、快速入门、算法说明、功能指南、案例集等
├── example/               # 各模型的量化示例（主要为V0框架量化脚本，但V1量化过程FAQ在此统一管理）
├── lab_calib/             # 量化校准数据集示例，如json、jsonl、图片等
├── lab_practice/          # 量化最佳实践仓库，管理各模型已验证的量化配置
├── msmodelslim/           # 源代码
├── skills/                # Agent Skills（含量化安装/适配/量化/策略/调优/测评）
├── test/                  # 测试用例与测试脚本
├── install.sh             # 安装脚本
├── requirements.txt       # 第三方依赖列表
├── pytest.ini             # 测试配置
├── README.md              # 项目总览与使用说明
├── LICENSE                # 开源许可证
├── OWNERS                 # 代码维护审批配置
└── setup.py               # 打包与安装配置脚本
```

> 说明：以上为最外层项目结构，V1 源代码主要集中在下文的 `msmodelslim/` 子目录中，V0 源代码集中在 `v0/` 中。

---

# `msmodelslim/` 结构（一级目录）

```text
msmodelslim/
├── app/              # 应用层，对外提供“一键量化、自动调优、模型分析”等功能（V1框架）
├── cli/              # 接口层，命令行接口及各功能子命令（V1框架）
├── core/             # 领域层，量化算法/模式/格式/DevKit等业务知识库（V1框架）
├── infra/            # 基础设施层，各类外部依赖适配知识库（V1框架）
├── model/            # 基础设施层，模型适配知识库，如DeepSeek、Qwen等（V1框架）
├── utils/            # 通用模块，如日志、配置、校验、安全等（V1框架）
├── Third_Party_Open_Source_Software_Notice
└── __init__.py
```

V1框架涉及目录如下：

```text
msmodelslim/
├── app/              # 应用层，对外提供“一键量化、自动调优、量化分析”等功能
├── cli/              # 接口层，命令行接口及各功能子命令
├── core/             # 领域层，量化算法/模式/格式/DevKit等业务知识库
├── infra/            # 基础设施层，各类外部依赖适配知识库
├── model/            # 基础设施层，模型适配知识库，如DeepSeek、Qwen等
├── utils/            # 通用模块，如日志、配置、校验、安全等
├── Third_Party_Open_Source_Software_Notice
└── __init__.py
```

> 说明：下面重点展开V1框架目录。

---

# `app/` 目录结构

```text
app/                          # 应用层，对外提供“一键量化、自动调优、量化分析”等功能
├── __init__.py
├── analysis/                    # 量化分析应用
│   ├── __init__.py
│   ├── application.py               # 量化分析的业务流程
│   └── result_displayer_infra.py    # 量化分析提出“结果展示”基础设施需求
├── auto_tuning/                 # 自动调优应用
│   ├── __init__.py
│   ├── application.py               # 自动调优的业务流程
│   ├── evaluation_service_infra.py  # 自动调优提出“测评”基础设施需求
│   ├── model_info_interface.py      # 自动调优提出“模型信息获取”模型适配需求
│   ├── plan_manager_infra.py        # 自动调优提出“调优计划管理”基础设施需求
│   ├── practice_accuracy_infra.py   # 自动调优提出“量化配置精度缓存”基础设施需求
│   ├── practice_history_infra.py    # 自动调优提出“调优历史管理”基础设施需求
│   └── practice_manager_infra.py    # 自动调优
└── naive_quantization/          # 一键量化应用
    ├── __init__.py
    ├── application.py               # 一键量化的业务流程
    ├── model_info_interface.py      # 一键量化提出“模型信息获取”模型适配需求
    └── practice_manager_infra.py    # 一键量化提出“最佳实践管理”基础设施需求
```

---

# `cli/` 目录结构

```text
cli/                          # 接口层，命令行接口及各功能子命令
├── __init__.py
├── __main__.py                   # msmodelslim命令总入口
├── analysis/                     # 量化分析子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim analyze命令入口
├── auto_tuning/                  # 自动调优子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim tune命令入口
├── naive_quantization/           # 一键量化相关 CLI 子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim quant命令入口
└── utils.py                      # 通用工具函数
```

---

# `core/` 目录结构

```text
core/                     # 领域层，按领域积累和管理量化知识
├── __init__.py
├── analysis_service/         # 量化分析服务知识库
├── base/                     # 基础协议（待移除，现有实现需移入对应的知识库）
├── context/                  # 上下文知识库
├── convert/                  # 离线权重转换（含 processors/）
├── devkit/                   # DevKit：group / load / save（非算法本体）
│   ├── group.py
│   ├── load.py
│   └── save.py
├── dtype_ops/                # 各数据格式的量化、反量化算子
├── graph/                    # 子图模式知识库
├── infer_engine/             # 伪量化推理引擎
├── observer/                 # 特征统计知识库
├── practice/                 # 最佳实践知识库
├── quant_service/            # 量化服务知识库
├── quantization_algorithm/   # 量化算法知识库（原顶层 processor/）
│   ├── base.py
│   ├── analysis/                 # 量化分析类指标算法
│   ├── anti_outlier/             # SmoothQuant 类离群值抑制算法
│   ├── quant/                    # 结构量化类算法
│   ├── quarot/                   # QuaRot 相关处理器
│   └── ...
├── quantization_format/      # 量化格式知识库（原顶层 format/）
├── quantization_schema/      # 量化模式 / QIR 知识库（原顶层 ir/）
├── quantizer/                # 权重量化和激活值量化知识库（待梳理）
├── runner/                   # 调度知识库
├── train/                    # 训练任务、训练钩子与启动器
└── tune_strategy/            # 调优策略知识库
```

资料侧对应入口：

- 算法：《[量化算法](../../knowledge_base/quantization_algorithms/README.md)》
- DevKit：《[DevKit](../../knowledge_base/devkit/README.md)》
- 模式：《[量化模式](../../knowledge_base/quantization_schema/README.md)》
- 格式：《[量化格式](../../knowledge_base/quantization_format/README.md)》

---

# `infra/` 目录结构

```text
infra/                                     # 基础设施，满足其他组件对外部依赖的需求，管理适配知识
├── __init__.py
├── dataset_loader/                            # 数据集加载类基础设施
├── evaluation/                                # 评测工具类基础设施
├── analysis_pipeline_loader.py                # 基于YAML的流水线模板加载基础设施
├── debug_info_persistence.py                  # 基于JSON和Safetensors的调试信息持久化基础设施
├── file_dataset_loader.py                     # 基于文件的LLM数据集加载基础设施
├── logging_analysis_result_displayer.py       # 基于日志的分析结果展示基础设施
├── plugin_practice_dirs.py
├── service_oriented_evaluate_service.py       # 基于服务的模型测评服务基础设施
├── vllm_ascend_server.py                      # 基于vLLM-Ascend的服务化基础设施
├── yaml_plan_manager.py                       # 基于YAML的调优计划管理基础设施
├── yaml_practice_accuracy_manager.py          # 基于YAML的调优精度缓存管理基础设施
├── yaml_practice_history_manager.py           # 基于YAML的调优历史管理基础设施
├── yaml_practice_manager.py                   # 基于YAML的最佳实践管理基础设施
└── yaml_quant_config_exporter.py              # 基于YAML的量化配置导出基础设施
```

---

# `core/quantization_schema/` 目录结构（原 `ir/`）

```text
quantization_schema/          # 领域层，量化模式知识库，以最精简方式描述用于替换浮点结构的量化结构
├── __init__.py
├── qal/                          # 数据类型定义
├── api/                          # 汇总各数据类型的量化、反量化算法
├── const.py                      # 常见量化组合
├── w8a8_static.py                # W8A8静态量化模式
├── w8a8_dynamic.py               # W8A8动态量化模式
└── ...                           # 其它量化模式
```

---

# `model/` 目录结构

```text
model/                        # 基础设施层，模型适配知识库，满足其他组件的模型适配需求
├── __init__.py
├── base.py                       # 模型适配基类
├── interface.py                  # 模型适配基础协议
├── interface_hub.py              # 模型适配协议路由
├── common/                       # 通用模型逻辑
├── default/                      # 默认模型适配（兜底实现）
├── deepseek_v3/                  # DeepSeek V3系列适配
├── deepseek_v3_2/                # DeepSeek V3.2系列适配
└── ...                           # 其它模型适配
```

---

# `core/quantization_algorithm/` 目录结构（原 `processor/`）

```text
quantization_algorithm/   # 领域层，量化算法知识库，集成量化分析、离群值抑制、结构量化算法等
├── __init__.py
├── base.py                   # 算法基类
├── analysis/                 # 量化分析类指标算法
├── anti_outlier/             # SmoothQuant 类离群值抑制算法
├── quant/                    # 结构量化类算法
├── quarot/                   # QuaRot 相关处理器
└── ...                       # 其他算法
```

> 说明：`group` / `load` / `saver` 已抽至 `core/devkit/`，资料见《[DevKit](../../knowledge_base/devkit/README.md)》；离线 convert processors 位于 `core/quantization_algorithm/convert/`。

---

# `utils/` 目录结构

```text
utils/                   # 通用模块
├── __init__.py
├── patch/                    # 外部库补丁
├── plugin/                   # 插件机制
├── security/                 # 安全
├── validation/               # 类型和数值校验
├── config.py                 # 通用配置加载/管理
├── config_map.py             # 配置映射工具
├── exception.py              # 自定义异常类型
├── exception_decorator.py    # 统一异常装饰器
├── logging.py                # 日志系统封装
├── timeout.py                # 超时控制工具
└── ...                       # 其他通用模块
```
