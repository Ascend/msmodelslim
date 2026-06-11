# Project Directory Structure Overview

The msModelSlim project tracks the evolutionary trajectory of AI from small models to foundation models.

Initial development introduced compression capabilities including pruning, distillation, and quantization for small models. Following the emergence of foundation models, development progressively shifted toward foundation model quantization techniques.

msModelSlim supports industry-standard foundation model quantization algorithms, including Smooth Quant, GPTQ, and HQQ.

As foundation model architectures grow more complex and quantization bit-widths scale down to a 4-bit configuration or even ultra-low bit-widths, standalone algorithms can no longer simultaneously satisfy performance and accuracy requirements. Consequently, composite strategies have become the standard practice.

Furthermore, the number of foundation models is growing exponentially, posing strict requirements for better tool usability, higher quantization tuning efficiency, and reduced single-model processing duration.

To address these challenges, msModelSlim has developed a hierarchically decoupled framework for foundation model quantization, referred to as the V1 framework. Legacy quantization approaches are categorized under the V0 framework.

You are advised to use the V1 framework, and your contributions to its development are highly welcome.

As the msModelSlim V1 framework is continuously evolving, it does not yet fully encapsulate all capabilities of the V0 framework or previous small model compression features. Legacy code is still required to bridge these functional gaps.

Currently, three codebases coexist within the msModelSlim repository. They are organized into separate directories where possible and will be progressively consolidated in future updates.

---

# Top-Level Directory Structure of the Repository

```text
msmodelslim/
├── ascend_utils/          # Base dependencies for small model compression including pruning, quantization, and distillation
├── config/                # Global configuration directory (V1 framework)
├── docs/                  # Project documentation including installation guides, quickstart guides, algorithm descriptions, feature guides, and case studies
├── example/               # Model quantization examples (primarily V0 scripts, alongside centralized V1 process FAQs)
├── lab_calib/             # Calibration dataset examples containing formats such as JSON, JSONL, and images (V1 framework)
├── lab_practice/          # Best-practice quantization configurations for various models (V1 framework)
├── modelslim/             # Compatibility layer for the modelslim alias (V0 framework)
├── msmodelslim/           # Project source code
├── precision_tool/        # Fake-quantization accuracy evaluation tool (V0 framework)
├── security/              # Core security verification modules (V0 framework)
├── test/                  # Test cases and test scripts
├── install.sh             # Installation script
├── requirements.txt       # Third-party dependency list
├── pytest.ini             # Test configuration
├── README.md              # Project overview and usage guidelines
├── LICENSE                # Open source license
├── OWNERS                 # Code ownership and approval configuration
└── setup.py               # Packaging and installation configuration script
```

The V1 framework involves the following directories:

```text
msmodelslim/
├── config/                # Global configuration directory
├── docs/                  # Project documentation including installation guides, quickstart guides, algorithm descriptions, feature guides, and case studies
├── example/               # Model quantization examples (primarily V0 scripts, alongside centralized V1 process FAQs)
├── lab_calib/             # Calibration dataset examples containing formats such as JSON, JSONL, and images
├── lab_practice/          # Repository for verified best-practice model quantization configurations
├── msmodelslim/           # Project source code
├── test/                  # Test cases and test scripts
├── install.sh             # Installation script
├── requirements.txt       # Third-party dependency list
├── pytest.ini             # Test configuration
├── README.md              # Project overview and usage guidelines
├── LICENSE                # Open source license
├── OWNERS                 # Code ownership and approval configuration
└── setup.py               # Packaging and installation configuration script
```

> Note: This is the top-level project structure. The source code is primarily located in the `msmodelslim/` subdirectory and is described in detail below.

---

# `msmodelslim/` Structure (Level-1 Directory)

```text
msmodelslim/
├── app/              # Application layer for features such as quick quantization, automatic tuning, and model analysis (V1 framework)
├── cli/              # Interface layer containing the command-line interface and functional subcommands (V1 framework)
├── common/           # Framework-agnostic compression logic including knowledge distillation, low-rank factorization, and pruning (small models)
├── core/             # Domain layer containing business logic and functional modules for quantization (V1 framework)
├── infra/            # Infrastructure layer containing adapters for external dependencies (V1 framework)
├── ir/               # Domain layer defining core intermediate representations and low-bit modes such as W8A8 quantization (V1 framework)
├── mindspore/        # Adapter components and functional implementations for the MindSpore framework (V0 and small models)
├── model/            # Infrastructure layer containing model-specific structure adapters for models such as DeepSeek and Qwen
├── onnx/             # Adapter components and functional implementations for the ONNX framework (V0 and small models)
├── processor/        # Domain layer implementing quantization execution processors such as the GPTQ algorithm (V1 framework)
├── pytorch/          # Adapter components and functional implementations for the PyTorch framework (V0 and small models)
├── quant/            # Multimodal quantization capabilities (V0 framework)
├── utils/            # Common modules, such as logging, configuration, validation, and security (V1 framework)
├── Third_Party_Open_Source_Software_Notice
└── __init__.py
```

The V1 framework involves the following directories:

```text
msmodelslim/
├── app/              # Application layer for features such as quick quantization, automatic tuning, and quantization analysis
├── cli/              # Interface layer containing the command-line interface and functional subcommands
├── core/             # Domain layer containing business logic and functional modules for quantization
├── infra/            # Infrastructure layer containing adapters for external dependencies
├── ir/               # Domain layer defining core intermediate representations and low-bit modes such as W8A8 quantization
├── model/            # Infrastructure layer containing model-specific structure adapters for models such as DeepSeek and Qwen
├── processor/        # Domain layer implementing quantization execution processors such as the GPTQ algorithm
├── utils/            # Common modules, such as logging, configuration, validation, and security
├── Third_Party_Open_Source_Software_Notice
└── __init__.py
```

> Note: The following section describes the V1 framework directory components.

---

# `app/` Directory Structure

```text
app/                          # Application layer for features such as quick quantization, automatic tuning, and quantization analysis
├── __init__.py
├── analysis/                    # Quantization analysis application module
│   ├── __init__.py
│   ├── application.py               # Workflow and orchestration for quantization analysis
│   └── result_displayer_infra.py    # Infrastructure interface for result visualization
├── auto_tuning/                 # Automatic tuning application module
│   ├── __init__.py
│   ├── application.py               # Workflow and orchestration for automatic tuning
│   ├── evaluation_service_infra.py  # Infrastructure interface for evaluation services
│   ├── model_info_interface.py      # Model adaptation interface for retrieving model metadata
│   ├── plan_manager_infra.py        # Infrastructure interface for tuning plan management
│   ├── practice_accuracy_infra.py   # Infrastructure interface for quantization configuration accuracy caching
│   ├── practice_history_infra.py    # Infrastructure interface for tuning history management
│   └── practice_manager_infra.py    # Infrastructure interface for practice management
└── naive_quantization/          # Quick quantization application module
    ├── __init__.py
    ├── application.py               # Workflow and orchestration for quick quantization
    ├── model_info_interface.py      # Model adaptation interface for retrieving model metadata
    └── practice_manager_infra.py    # Infrastructure interface for best-practice configuration management
```

---

# `cli/` Directory Structure

```text
cli/                          # Interface layer containing the command-line interface and functional subcommands
├── __init__.py
├── __main__.py                   # Primary entry point for the msmodelslim CLI application
├── analysis/                     # Quantization analysis subcommand module
│   ├── __init__.py
│   └── __main__.py                   # Entry point for the msmodelslim analyze command
├── auto_tuning/                  # Automatic tuning subcommand module
│   ├── __init__.py
│   └── __main__.py                   # Entry point for the msmodelslim tune command
├── naive_quantization/           # Quick quantization subcommand module
│   ├── __init__.py
│   └── __main__.py                   # Entry point for the msmodelslim quant command
└── utils.py                      # General-purpose utility functions for the CLI module
```

---

# `core/` Directory Structure

```text
core/                     # Domain layer organizing and managing quantization capabilities by domain
├── __init__.py
├── analysis_service/         # Quantization analysis module that analyzes specific models using defined metrics to guide quantization scheme design
│   ├── __init__.py
│   ├── interface.py              # Definition and protocol for the quantization analysis service
│   └── pipeline_analysis/        # msModelSlim V1 pipeline-based quantization analysis service
├── base/                     # Base protocols (to be deprecated; existing implementations will be relocated to respective modules
│   ├── __init__.py
│   ├── processor.py
│   └── protocol.py
├── const.py                  # Constants (to be deprecated; existing implementations will be relocated to respective modules)
├── context/                  # Context management module for the unified management and sharing of states across processing workflows
│   ├── __init__.py
│   ├── base.py                   # Base class for context
│   ├── context_factory.py        # Context factory
│   ├── interface.py              # Definition and protocol for the context
│   ├── local_dict_context/       # Local dictionary-based context
│   └── shared_dict_context/      # Multi-process dictionary-based context
├── graph/                    # Subgraph pattern module. Quantization algorithms rely on matching specific structural subgraphs in models, which is a prerequisite for applying quantization methods.
│   ├── __init__.py
│   └── adapter_types.py          # Outlier suppression subgraph pattern (to be expanded into separate files for each pattern)
├── observer/                 # Feature statistics module
│   ├── __init__.py
│   ├── histogram.py              # Histogram statistics
│   ├── minmax.py                 # Min-max statistics
│   └── recall_window.py          # Sliding-window statistics
├── practice/                 # Best practice module (to be optimized; currently only one schema is supported)
│   ├── __init__.py
│   └── interface.py              # Best practice definition
├── quant_service/            # Quantization service module that converts floating-point models into quantized models and generates quantized weights
│   ├── __init__.py
│   ├── dataset_loader_infra.py   # Infrastructure interface for dataset loading
│   ├── interface.py              # Definition and protocol for the quantization service
│   ├── key_info_persistence_infra.py
│   ├── modelslim_v0/             # msModelSlim V0-based LLM quantization service
│   ├── modelslim_v1/             # msModelSlim V1-based LLM quantization service
│   ├── multimodal_sd_v1/         # msModelSlim V1-based multimodal generative model quantization service
│   ├── multimodal_vlm_v1/        # msModelSlim V1-based multimodal understanding model quantization service
│   └── proxy/                    # Comprehensive quantization service based on the preceding quantization services
├── quantizer/                # Weight and activation quantization module (to be organized)
│   ├── __init__.py
│   ├── attention.py
│   ├── base.py
│   ├── impl/
│   └── linear.py
├── runner/                   # Scheduling module (to be organized)
│   ├── __init__.py
│   ├── base.py
│   ├── dp_layer_wise_runner.py
│   ├── generated_runner.py
│   ├── layer_wise_runner.py
│   ├── model_hook_interface.py
│   ├── pipeline_interface.py
│   └── pipeline_parallel_runner.py
└── tune_strategy/            # Tuning strategy module that adjusts quantization configurations based on accuracy feedback
    ├── __init__.py
    ├── base.py                   # Base class for tuning strategies
    ├── common/                   # Common module for tuning strategies.
    ├── dataset_loader_infra.py   # Infrastructure interface for dataset loading
    ├── interface.py              # Definition and protocol for tuning strategies
    ├── plugin_factory.py         # Plugin-based factory for tuning strategies
    ├── standing_high/            # Standing-high tuning strategy
    └── standing_high_with_experience/  # Standing-high tuning strategy based on expert experience
```

---

# `infra/` Directory Structure

```text
infra/                                     # Infrastructure module for external dependency adaptation and cross-component support
├── __init__.py
├── dataset_loader/                            # Dataset loading infrastructure
├── evaluation/                                # Evaluation tool infrastructure
├── analysis_pipeline_loader.py                # YAML-based pipeline template loader
├── debug_info_persistence.py                  # JSON and Safetensors-based debug information persistence component
├── file_dataset_loader.py                     # File-based LLM dataset loader
├── logging_analysis_result_displayer.py       # Log-based analysis result displayer
├── plugin_practice_dirs.py
├── service_oriented_evaluate_service.py       # Service-based model evaluation service
├── vllm_ascend_server.py                      # vLLM-Ascend-based service serving infrastructure
├── yaml_plan_manager.py                       # YAML-based tuning plan manager
├── yaml_practice_accuracy_manager.py          # YAML-based tuning accuracy cache manager
├── yaml_practice_history_manager.py           # YAML-based tuning history manager
├── yaml_practice_manager.py                   # YAML-based best practice manager
└── yaml_quant_config_exporter.py              # YAML-based quantization configuration exporter
```

---

# `ir/` Directory Structure

```text
ir/                           # Domain layer module for quantization representations that concisely describe structures replacing floating-point operations
├── __init__.py
├── qal/                          # Data type definitions
├── api/                          # Unified quantization and dequantization algorithms for all data types
├── const.py                      # Common quantization configuration combinations
├── w8a8_static.py                # W8A8 static quantization mode
├── w8a8_dynamic.py               # W8A8 dynamic quantization mode
└── ...                           # Other quantization modes
```

---

# `model/` Directory Structure

```text
model/                        # Infrastructure layer module for model adaptation and cross-component support
├── __init__.py
├── base.py                       # Base class for adapters
├── interface.py                  # Base model adaptation protocol
├── interface_hub.py              # Model adaptation protocol routing
├── common/                       # Common model logic
├── default/                      # Default model adapter (fallback implementation)
├── deepseek_v3/                  # DeepSeek V3 series adapter
├── deepseek_v3_2/                # DeepSeek V3.2 series adapter
└── ...                           # Other model adapters
```

---

# `processor/` Directory Structure

```text
processor/                # Domain layer algorithm module integrating quantization analysis, outlier suppression, and structured quantization
├── __init__.py
├── base.py                   # Base algorithm class
├── common/                   # Common module
├── analysis/                 # Quantization analysis metric algorithms
├── anti_outlier/             # SmoothQuant outlier suppression algorithms
├── quant/                    # Structured quantization algorithms
├── quarot/                   # QuaRot-related processors
└── ...                       # Other algorithms
```

---

# `utils/` Directory Structure

```text
utils/                   # General-purpose utilities module
├── __init__.py
├── patch/                    # External library patches
├── plugin/                   # Plugins
├── security/                 # Security utilities
├── validation/               # Type and value validation utilities
├── config.py                 # General configuration loader and manager
├── config_map.py             #Configuration mapping utility
├── exception.py              # Custom exception types
├── exception_decorator.py    # Unified exception decorator
├── logging.py                # Log system encapsulation
├── timeout.py                # Timeout control utility
└── ...                       # Other general-purpose modules
```
