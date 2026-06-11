# 项目目录结构总览

msModelSlim项目见证了从小模型到大模型的AI发展历程。

早期支持了小模型剪枝、蒸馏、量化等压缩功能，当大模型初现时，逐步转向大模型量化功能。

msModelSlim支持了广泛使用的SmoothQuant、GPTQ、HQQ等大模型量化算法。

但大模型结构愈发复杂，量化位宽也进入到4比特甚至极低比特时代，单一算法已经无法同时满足性能和精度的双重需求，复合策略成为常态。

另一方面，大模型数量也呈井喷之势，亟需提升工具易用性，提高量化调优效率，减少单个模型的量化耗时。

为了迎接大模型量化的浪潮，msModelSlim总结发展出了分层解耦的大模型量化框架，称为V1框架，而将过去的大模型量化手段归为V0框架。

我们推荐您使用V1框架，并期待您参与V1框架的共建。

msModelSlim V1框架仍在成长过程中，未完全覆盖V0框架能力和小模型压缩功能，仍需要历史代码填补功能缺失。

目前三套代码共存于msModelSlim代码仓，尽量以目录隔离，后续将逐步整理归一。

---

# 仓库顶层目录结构

```text
msmodelslim/
├── ascend_utils/          # 小模型剪枝、量化、蒸馏等压缩功能的基础依赖（小模型）
├── config/                # 全局配置目录（V1框架）
├── docs/                  # 项目文档，如安装指南、快速入门、算法说明、功能指南、案例集等
├── example/               # 各模型的量化示例（主要为V0框架量化脚本，但V1量化过程FAQ在此统一管理）
├── lab_calib/             # 量化校准数据集示例，如json、jsonl、图片等（V1框架）
├── lab_practice/          # 各模型最佳实践量化配置（V1框架）
├── modelslim/             # 兼容modelslim别名（V0框架）
├── msmodelslim/           # 源代码
├── precision_tool/        # 伪量化精度评估工具（V0框架）
├── security/              # 安全检查基础模块（V0框架）
├── test/                  # 测试用例与测试脚本
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
├── test/                  # 测试用例与测试脚本
├── install.sh             # 安装脚本
├── requirements.txt       # 第三方依赖列表
├── pytest.ini             # 测试配置
├── README.md              # 项目总览与使用说明
├── LICENSE                # 开源许可证
├── OWNERS                 # 代码维护审批配置
└── setup.py               # 打包与安装配置脚本
```

> 说明：以上为最外层项目结构，源代码主要集中在 `msmodelslim/` 子目录中，下面重点展开。

---

# `msmodelslim/` 结构（一级目录）

```text
msmodelslim/
├── app/              # 应用层，对外提供“一键量化、自动调优、模型分析”等功能（V1框架）
├── cli/              # 接口层，命令行接口及各功能子命令（V1框架）
├── common/           # 与AI框架无关的压缩逻辑，如知识蒸馏、低秩分解、剪枝等（小模型）
├── core/             # 领域层，各类量化业务知识库（V1框架）
├── infra/            # 基础设施层，各类外部依赖适配知识库（V1框架）
├── ir/               # 领域层，量化模式知识库，最基础的知识，描述何为量化，如W8A8量化等（V1框架）
├── mindspore/        # MindSpore 框架的适配与功能实现（V0框架和小模型）
├── model/            # 基础设施层，模型适配知识库，如DeepSeek、Qwen等（V1框架）
├── onnx/             # ONNX 框架的适配和功能实现（V0框架和小模型）
├── processor/        # 领域层，量化算法知识库，描述如何做量化，如GPTQ算法等（V1框架）
├── pytorch/          # PyTorch 框架的适配与功能实现（V0框架和小模型）
├── quant/            # 多模态量化（V0框架）
├── utils/            # 通用模块，如日志、配置、校验、安全等（V1框架）
├── Third_Party_Open_Source_Software_Notice
└── __init__.py
```

V1框架涉及目录如下：

```text
msmodelslim/
├── app/              # 应用层，对外提供“一键量化、自动调优、量化分析”等功能
├── cli/              # 接口层，命令行接口及各功能子命令
├── core/             # 领域层，各类量化业务知识库
├── infra/            # 基础设施层，各类外部依赖适配知识库
├── ir/               # 领域层，量化模式知识库，最基础的知识，描述何为量化，如W8A8量化等
├── model/            # 基础设施层，模型适配知识库，如DeepSeek、Qwen等
├── processor/        # 领域层，量化算法知识库，描述如何做量化，如GPTQ算法等
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
├── analysis_service/         # 量化分析服务知识库，量化分析服务使用特定指标分析特定模型以辅助设计该模型的量化方案
│   ├── __init__.py
│   ├── interface.py              # 量化分析服务定义和协议
│   └── pipeline_analysis/        # 基于msModelSlim V1框架流水线的量化分析服务
├── base/                     # 基础协议（待移除，现有实现需移入对应的知识库）
│   ├── __init__.py
│   ├── processor.py
│   └── protocol.py
├── const.py                  # 常量（待移除，现有实现需移入对应的知识库）
├── context/                  # 上下文知识库，上下文统一管理和共享处理流程中的状态，用户可以将上下文当作无限容量的存储空间
│   ├── __init__.py
│   ├── base.py                   # 上下文基类
│   ├── context_factory.py        # 上下文工厂
│   ├── interface.py              # 上下文定义和协议
│   ├── local_dict_context/       # 基于本地字典的上下文
│   └── shared_dict_context/      # 基于多进程字典的上下文
├── graph/                    # 子图模式知识库，量化算法均是基于特定模式的模型结构子图，在模型中匹配到并描述清楚子图是应用量化算法的前提
│   ├── __init__.py
│   └── adapter_types.py          # 离群值抑制子图模式（待展开，每个子图模式一个文件）
├── observer/                 # 特征统计知识库
│   ├── __init__.py
│   ├── histogram.py              # 直方图统计
│   ├── minmax.py                 # 最大最小统计
│   └── recall_window.py          # 滑动窗统计
├── practice/                 # 最佳实践知识库（待优化，目前仅支持一种最佳实践格式）
│   ├── __init__.py
│   └── interface.py              # 最佳实践定义
├── quant_service/            # 量化服务知识库，量化服务将浮点模型转化为量化模型并生成量化权重
│   ├── __init__.py
│   ├── dataset_loader_infra.py   # 量化服务提出“数据集加载”基础设施要求
│   ├── interface.py              # 量化服务定义和协议
│   ├── key_info_persistence_infra.py
│   ├── modelslim_v0/             # 基于msModelSlim V0框架的LLM量化服务
│   ├── modelslim_v1/             # 基于msModelSlim V1框架的LLM量化服务
│   ├── multimodal_sd_v1/         # 基于msModelSlim V1框架的多模态生成模型量化服务
│   ├── multimodal_vlm_v1/        # 基于msModelSlim V1框架的多模态理解模型量化服务
│   └── proxy/                    # 基于上述量化服务的综合性量化服务
├── quantizer/                # 权重量化和激活值量化知识库（待梳理）
│   ├── __init__.py
│   ├── attention.py
│   ├── base.py
│   ├── impl/
│   └── linear.py
├── runner/                   # 调度知识库（待梳理）
│   ├── __init__.py
│   ├── base.py
│   ├── dp_layer_wise_runner.py
│   ├── generated_runner.py
│   ├── layer_wise_runner.py
│   ├── model_hook_interface.py
│   ├── pipeline_interface.py
│   └── pipeline_parallel_runner.py
└── tune_strategy/            # 调优策略知识库，调优策略根据精度反馈调整量化配置
    ├── __init__.py
    ├── base.py                   # 调优策略基类
    ├── common/                   # 调优策略通用模块
    ├── dataset_loader_infra.py   # 调优策略提出“数据集加载”需求
    ├── interface.py              # 调优策略定义和协议
    ├── plugin_factory.py         # 基于插件的调优策略工厂
    ├── standing_high/            # 摸高调优策略
    ├── standing_high_with_experience/  # 基于专家经验的摸高调优策略
    └── binary_fallback/          # 二分回退调优策略
```

---

# `infra/` 目录结构

```text
infra/                                     # 基础设施，满足其他组件对外部依赖的需求，管理适配知识
├── __init__.py
├── dataset_loader/                            # 数据集加载类基础设施
├── evaluation/                                # 测评工具类基础设施
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

# `ir/`目录结构

```text
ir/                           # 领域层，量化模式知识库，量化模式以最精简的方式描述用于替换浮点结构的量化结构
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

# `processor/` 目录结构

```text
processor/                # 领域层，量化算法知识库，集成量化分析、离群值抑制、结构量化算法等
├── __init__.py
├── base.py                   # 算法基类
├── common/                   # 通用模块
├── analysis/                 # 量化分析类指标算法
├── anti_outlier/             # SmoothQuant类离群值抑制算法
├── quant/                    # 结构量化类算法
├── quarot/                   # QuaRot 相关处理器
└── ...                       # 其他算法
```

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
