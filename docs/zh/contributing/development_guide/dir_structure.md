# 项目目录结构总览

msModelSlim项目见证了从小模型到大模型的AI发展历程。为了迎接大模型量化的浪潮，msModelSlim总结发展出了分层解耦的大模型量化框架，称为V1框架，而将过去的小模型剪枝、蒸馏、量化等压缩功能归为V0框架。我们推荐您使用V1框架，并期待您参与V1框架的共建。msModelSlim V1框架仍在成长过程中，未完全覆盖V0框架能力和小模型压缩功能，仍需要历史代码填补功能缺失。目前 V0 与 V1 代码共存于同一代码仓。V0 源码集中存放在根目录 v0/ 下，与 V1 源码目录隔离。

## 仓库顶层目录结构

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
├── v0/                    # V0 框架与小模型压缩源码，安装时由 setup.py 展开到原有安装目录
├── build.py               # 构建脚本：依赖拉取、编译、安装与测试
├── install.sh             # 安装脚本
├── mkdocs.yml             # 文档站点配置
├── requirements.txt       # 第三方依赖列表
├── pytest.ini             # 测试配置
├── README.md              # 项目总览与使用说明
├── README_EN.md           # 英文项目总览与使用说明
├── LICENSE                # 开源许可证
├── OWNERS                 # 代码维护审批配置
└── setup.py               # 打包与安装配置脚本
```

## `msmodelslim/` 目录结构

```text
msmodelslim/
├── app/              # 应用层，对外提供“一键量化、自动调优、模型敏感层分析”等功能
├── cli/              # 接口层，命令行接口及各功能子命令
├── core/             # 领域层，量化算法、模式、格式、DevKit 等业务知识库
├── infra/            # 基础设施层，各类外部依赖适配
├── model/            # 模型适配，如 DeepSeek、Qwen 等
├── utils/            # 通用模块，如日志、配置、校验、安全等
├── Third_Party_Open_Source_Software_Notice    # 第三方开源软件声明
└── __init__.py
```

### `app/` 目录结构

```text
app/                          # 应用层，对外提供“一键量化、自动调优、模型敏感层分析”等功能
├── __init__.py
├── analysis/                    # 模型敏感层分析
│   ├── __init__.py
│   ├── application.py               # 模型敏感层分析的业务流程
│   └── result_displayer_infra.py    # 模型敏感层分析提出“结果展示”基础设施需求
├── auto_tuning/                 # 自动调优
│   ├── __init__.py
│   ├── application.py               # 自动调优的业务流程
│   ├── evaluation_service_infra.py  # 自动调优提出“测评”基础设施需求
│   ├── model_info_interface.py      # 自动调优提出“模型信息获取”模型适配需求
│   ├── plan_manager_infra.py        # 自动调优提出“调优计划管理”基础设施需求
│   ├── practice_accuracy_infra.py   # 自动调优提出“量化配置精度缓存”基础设施需求
│   ├── practice_history_infra.py    # 自动调优提出“调优历史管理”基础设施需求
│   └── practice_manager_infra.py    # 自动调优提出“最佳实践管理”基础设施需求
├── eval/                        # 伪量化评测
│   ├── __init__.py
│   ├── application.py               # 伪量化评测的业务流程
│   └── result_displayer_infra.py    # 伪量化评测提出“结果展示”基础设施需求
├── naive_quantization/          # 一键量化
│   ├── __init__.py
│   ├── application.py               # 一键量化的业务流程
│   ├── model_info_interface.py      # 一键量化提出“模型信息获取”模型适配需求
│   └── practice_manager_infra.py    # 一键量化提出“最佳实践管理”基础设施需求
└── train/                       # 训练
    ├── __init__.py
    ├── proxy/                       # 训练任务代理
    └── qat/                         # 量化感知训练
```

### `cli/` 目录结构

```text
cli/                          # 接口层，命令行接口及各功能子命令
├── __init__.py
├── __main__.py                   # msmodelslim 命令总入口
├── logo.py                       # 命令行 Logo 与版本信息
├── utils.py                      # 命令行通用工具
├── analysis/                     # 模型敏感层分析子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim analyze 命令入口
├── auto_tuning/                  # 自动调优子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim tune 命令入口
├── eval/                         # 伪量化评测子命令
│   ├── __init__.py
│   └── __main__.py                   # msmodelslim eval 命令入口
└── naive_quantization/           # 一键量化子命令
    ├── __init__.py
    └── __main__.py                   # msmodelslim quant 命令入口
```

### `core/` 目录结构

```text
core/                     # 领域层，按领域积累和管理量化知识
├── __init__.py
├── analysis_service/         # 模型敏感层分析服务
├── base/                     # 处理器与调度请求的基础协议
├── const.py                  # 设备类型、量化类型等公共枚举
├── context/                  # 上下文，在流程间传递和汇总中间结果
├── convert/                  # 离线权重转换的类型、配置与协议
├── devkit/                   # 量化流程中的分组、加载与保存
│   ├── __init__.py
│   ├── group.py                  # 将多个处理器合并，减少校准次数
│   ├── load.py                   # 在流水线前后加载或卸载模块
│   └── save.py                   # 按指定格式保存量化结果
├── dtype_ops/                # 各数据格式的量化、反量化算子
├── graph/                    # 子图构建与适配
├── infer_engine/             # 伪量化推理引擎
├── observer/                 # 激活与权重的特征统计
├── practice/                 # 最佳实践配置
├── quant_service/            # 量化服务，组织校准、量化和保存流程
├── quantization_algorithm/   # 量化算法知识库
├── quantization_format/      # 量化格式知识库
├── quantization_schema/      # 量化模式知识库
├── quantizer/                # 权重量化与激活值量化
├── runner/                   # 量化流程调度
├── train/                    # 训练任务、训练钩子与启动器
└── tune_strategy/            # 调优策略
```

资料侧对应入口：

- 算法：《[量化算法](../../knowledge_base/quantization_algorithm/README.md)》
- DevKit：《[DevKit](../../knowledge_base/devkit/README.md)》
- 模式：《[量化模式](../../knowledge_base/quantization_schema/README.md)》
- 格式：《[量化格式](../../knowledge_base/quantization_format/README.md)》

#### `quantization_algorithm/` 目录结构

```text
quantization_algorithm/   # 量化算法知识库，集成模型敏感层分析、离群值抑制、结构量化等算法
├── __init__.py
├── base.py                   # 算法基类与处理器配置
├── analysis/                 # 模型敏感层分析算法
├── anti_outlier/             # 离群值抑制算法，如 SmoothQuant
├── quant/                    # 结构量化算法
├── convert/                  # 离线权重转换算法
└── ...                       # 其他算法，如 QuaRot、稀疏、FlatQuant
```

#### `quantization_schema/` 目录结构

```text
quantization_schema/          # 量化模式知识库，描述用于替换浮点结构的量化结构
├── __init__.py
├── qal/                          # 量化抽象层，定义量化数据类型
├── api/                          # 各数据类型的量化与反量化接口
├── kernels/                      # 量化计算核
├── const.py                      # 常见量化组合
├── w8a8_static.py                # W8A8 静态量化模式
├── w8a8_dynamic.py               # W8A8 动态量化模式
└── ...                           # 其他量化模式
```

#### `quantization_format/` 目录结构

```text
quantization_format/          # 量化格式知识库，定义量化结果的保存格式
├── __init__.py
├── base.py                       # 格式基类
├── interface.py                  # 格式导出协议
├── registry.py                   # 内置格式注册
├── ascendV1_format/              # AscendV1 保存格式
├── compressed_tensors_format/    # compressed-tensors 保存格式
├── mindie_format/                # MindIE 保存格式
└── ...                           # 其他格式与布局转换
```

### `infra/` 目录结构

```text
infra/                                     # 基础设施，满足其他组件对外部依赖的需求
├── __init__.py
├── dataset_loader/                            # 数据集加载
├── evaluation/                                # 评测工具适配，如 AISBench
├── io/                                        # JSON 与 Safetensors 读写
├── logging_analysis_result_displayer.py       # 基于日志的模型敏感层分析结果展示
├── vllm_ascend_server.py                      # 基于 vLLM-Ascend 的服务化推理
├── yaml_practice_manager.py                   # 基于 YAML 的最佳实践管理
└── ...                                        # 其他外部依赖适配，如调优计划、精度缓存、量化配置导出
```

### `model/` 目录结构

```text
model/                        # 模型适配，满足其他组件的模型适配需求
├── __init__.py
├── base.py                       # 模型适配基类
├── interface.py                  # 模型适配基础协议
├── interface_hub.py              # 模型适配协议路由
├── common/                       # 多模型共用逻辑
├── default/                      # 默认模型适配
├── plugin_factory/               # 插件方式加载模型适配
├── transformers/                 # Transformers 模型的加载与适配
├── deepseek_v3/                  # 具体模型适配示例
└── ...                           # 其他模型适配
```

### `utils/` 目录结构

```text
utils/                   # 通用模块
├── __init__.py
├── patch/                    # 外部库补丁
├── plugin/                   # 插件机制
├── security/                 # 路径、权限等安全检查
├── validation/               # 类型和数值校验
├── config.py                 # 通用配置加载与管理
├── config_map.py             # 配置映射
├── exception.py              # 自定义异常类型
├── exception_decorator.py    # 统一异常装饰器
├── logging.py                # 日志
├── timeout.py                # 超时控制
└── ...                       # 其他通用模块，如设备、分布式、缓存
```
