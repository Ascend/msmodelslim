# **msModelSlim Automatic Optimization and Acceleration Feature Design Specifications**

|                                           |            |
| ----------------------------------------- | ---------- |
| SIG group:                                | msit       |
| Incorporated into the following versions: | 26.0.0     |
| Designer:                                 | joejoezhou |
| Date:                                     | 20260122   |

**Copyright © 2026 msModelSlim Community**

Your reproduction, use, modification and distribution of this document is subject to the Creative Commons Attribution-ShareAlike 4.0 International Public License ("CC BY-SA 4.0"). For ease of understanding, you can visit thehttps://creativecommons.org/licenses/by-sa/4.0/Understand the overview (but not the replacement) of CC BY-SA 4.0. You can obtain the complete CC BY-SA 4.0 agreement from the following website:https://creativecommons.org/licenses/by-sa/4.0/legalcode.

**Revision records**

| Date     | Revised version | Revision Description | Authors    | Audited   |
| -------- | --------------- | -------------------- | ---------- | --------- |
| 20260122 | 1.0.0           | Document Creation    | joejoezhou | panyj1993 |

**The Table of Contents**

1. Feature Overview

     1.1 Scope

     1.2 Feature Requirement List

2. Requirement Scenario Analysis

     2.1 Feature Requirement Source and Value Overview

     2.2 Feature Scenario Analysis

     2.3 Feature Impact Analysis

     2.3.1 Hardware Limitations

     2.3.2 Technical Limitations

     2.3.3 Impact Analysis on the License

     2.3.4 Analysis of Impact on System Performance Specifications

     2.3.5 Analysis of Impact on System Reliability Specifications

     2.3.6 Impact on System Compatibility

     2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

     2.4 Analysis on the Implementation Solution of Similar Community/Commercial Software

3. Feature/Function implementation principles (multiple use cases can be broken down)

     3.1 Objectives

     3.2 Overall Solution

4. Use Case 1 Implementation

     4.1 Use Case Description

     4.2 Feature Design Ideas

     4.3 Constraints

     4.4 Detailed implementation (module-level or process-level message sequence diagram from user entry)

     4.5 Interfaces Between Subsystems (Mainly Covering the Interface Definition of Modules)

     4.6 Subsystem LLD

     4.6.1 Garbled Character Detection Check Item Design

     4.6.2 Pre-check Process Design

     4.7 DFX Attribute Design

     4.7.1 Performance Design

     4.7.2 Upgrade and Capacity Expansion Design

     4.7.3 Exception Handling Design

     4.7.4 Resource Management Design

     4.7.5 Miniaturized Design

     4.7.6 Testability Design

     4.7.7 Security Design

     4.8 External Interfaces

     4.9 Self-Test Case Design

5. Use Case 2 Implementation

     5.1 Use Case Description

     5.2 Feature Design Roadmap

     5.3 Constraints

     5.4 Detailed implementation (module level or process level message sequence diagram from user entry)

     5.5 Interfaces Between Subsystems (Mainly Covering the Interface Definition of Modules)

     5.6 Subsystem LLD

     5.6.1 Precision Cache Design

     5.6.2 Historical Index Design

     5.6.3 Cache reuse mechanism design

     5.6.4 Design of Breakpoint Resume Process

     5.7 DFX Attribute Design

     5.7.1 Performance Design

     5.7.2 Upgrade and Capacity Expansion Design

     5.7.3 Design of Exception Handling

     5.7.4 Resource Management Design

     5.7.5 Miniaturized Design

     5.7.6 Testability Design

     5.7.7 Security Design

     5.8 External Interfaces

     5.9 Self-Test Case Design

6. Use Case 3 Implementation

     6.1 Use Case Description

     6.2 Feature Design Roadmap

     6.3 Constraints

     6.4 Detailed implementation (module level or process level message sequence diagram from user entry)

     6.5 Interfaces Between Subsystems (Mainly Covering the Definition of Module Interfaces)

     6.6 Detailed Design of Subsystems

     6.6.1 New Policy Module Design

     6.6.2 Design for identification of model structure type

     6.6.3 Design of Expert Experience Table

     6.6.4 Design of the automatic table lookup mechanism

     6.6.5 Policy Implementation Mode

     6.7 DFX Attribute Design

     6.7.1 Performance Design

     6.7.2 Upgrade and Capacity Expansion Design

     6.7.3 Design of Exception Handling

     6.7.4 Resource Management Design

     6.7.5 Miniaturized Design

     6.7.6 Testability Design

     6.7.7 Security Design

     6.8 External Interfaces of the System

     6.9 Self-Test Case Design

7. Reliability and availability design

     7.1 Redundancy Design

     7.2 Fault Management

     7.3 Overload control design

     7.4 Upgrade Without Service Interruption

     7.5 Design for human error

     7.6 Fault Prediction and Prevention Design

8. Design for features and non-functional quality attributes

     8.1 Testability

     8.2 Serviceability

     8.3 Evolvability

     8.4 Openness

     8.5 Compatibility

     8.6 Scalability/Scalability

     8.7 Maintainability

     8.8 Documentation

9. Data Structure Design (Optional)

10. List of references

**Table Catalogue**

Table 1 List of feature requirements

Table 2: Safety Design Qualification Form

Table 3: List of modified documents

**Figure Catalogue**

Figure 1: Overall implementation principle

**List of abbreviations:**

| Abbreviations Abbreviations | Full spelling English full name | Chinese explanation Chinese explanation |
| --------------------------- | ------------------------------- | --------------------------------------- |
| MHA                         | Multi-Head Attention            | multiheaded attention mechanism         |
| MLA                         | Multi-Head Latent Attention     | Potential attention mechanism of bulls  |
| DSA                         | Distributed Sparse Attention    | distributed sparse attention mechanism  |
| SWA                         | Sliding Window Attention        | Sliding window attention mechanism      |
| NPU                         | Neural Processing Unit          | neural network processing unit          |
| YAML                        | YAML Ain't Markup Language      | YAML Markup Language                    |
| MD5                         | Message Digest Algorithm 5      | Message Digest Algorithm 5              |

## 1. Feature Overview

Automatic precision feedback optimization is an existing core feature of msModelSlim. The automatic process reduces the manual workload of model quantization precision optimization. Based on the precision optimization experience accumulated in mature quantization modes, this function automatically generates quantization configurations, evaluates model precision, and adjusts policies based on precision feedback until a quantization solution that meets precision requirements is found.

**Automatic optimization acceleration is an acceleration optimization based on the existing precision feedback automatic optimization function, aiming to further improve the efficiency and reliability of automatic optimization. This feature uses three key optimization points to accelerate the automatic optimization process: (1) Dataset evaluation is skipped when garbled characters are displayed, avoiding the waste of computing resources. 2. Resumable scheduling is supported to avoid repeated evaluation caused by unexpected interruption. 3. Optimization policies based on expert experience simplify user configuration and improve optimization efficiency.**

This feature provides the following benefits to customers: (1) Saves computing resources and reduces unnecessary computing overheads by intelligently skipping invalidity evaluation. 2. Improves optimization reliability and supports resumable adjustment at breakpoints to avoid work loss. (3) Simplified user operations and automatic selection of optimal policies based on expert experience, reducing configuration complexity.

This document describes the design and implementation of the automatic optimization acceleration feature, including three main use cases: skipping dataset test when garbled characters are displayed, automatic optimization supports resumable adjustment at breakpoints, and optimization policies based on expert experience. This document is intended for the development, test, and maintenance personnel of the automatic optimization function of the msModelSlim tool.

### 1.1 Scope

This feature is an acceleration optimization based on the existing automatic precision feedback optimization function. It provides the following functions:

1. **Garbled character detection and skipping mechanism: Before formal evaluation, the system checks whether garbled characters are displayed in the model output. If garbled characters are detected, the system skips the dataset evaluation, saving computing resources. This is an optimization of the existing automatic optimization function, preventing invalid evaluation from wasting computing resources.**
2. **Resumable adjustment by breakpoint: Restores the evaluated quantitative configuration results from the historical precision cache, avoiding repeated evaluation and implementing resumable adjustment at breakpoints during optimization. This is an optimization of the existing automatic optimization function, improving the reliability of the optimization process.**
3. **Optimization policy based on expert experience: Create an independent expert_experience policy module to automatically query tables based on the model structure type (such as MHA/MLA/DSA/SWA/GatedDeltaNet) and obtain the algorithm search space without manual input. This is an optimization of the existing automatic optimization function, simplifying user configuration and improving optimization efficiency.**

**Note: The automatic precision feedback optimization process is an existing function and is not covered by this feature. This feature involves only the preceding three acceleration optimization points.**

### 1.2 Feature Requirement List

Table 1 List of feature requirements

| Requirement No. | Requirement name                                                     | Feature Description                                                                                                                                                                                                                                                                                                          | Remarks |
| --------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 1               | Skipping Dataset Assessment When Conversation Garbled Characters     | Before evaluating the model precision, the precheck mechanism is used to check whether garbled characters are displayed in the model output. If garbled characters are detected, the system skips the dataset evaluation and directly returns the evaluation result with the precision of 0. This saves computing resources. | -       |
| 2               | Automatic optimization supports resumable adjustment at breakpoints. | Restores the evaluated quantization configuration result from the historical precision cache. When the tuning process is interrupted unexpectedly, the historical precision cache is automatically detected and reused during restart, avoiding repeated evaluation of the same quantization configuration.                  | -       |
| 3               | Optimization policies based on expert experience                     | Create an independent expert_experience policy module. The module supports automatic table query based on the model structure type (such as MHA/MLA/DSA/SWA/GatedDeltaNet) to obtain the search space of the algorithm. Users do not need to manually enter the search space configuration.                                  | -       |

## 2. Requirement Scenario Analysis

### 2.1 Feature Requirement Source and Value Overview

The msModelSlim tool has implemented the automatic precision tuning function, which reduces the manual workload of model quantization precision tuning through the automatic process. However, the automatic optimization function has the following problems:

1. **Waste of resources in invalid evaluation: When the quantized model output is garbled characters, complete dataset evaluation is still performed, wasting a lot of computing resources.**
2. **If the optimization process is interrupted due to an unexpected interruption (for example, a system fault or manual shutdown), the system restarts from the beginning. In this case, the evaluated configuration results cannot be reused.**
3. **High configuration complexity: The standing_high policy requires users to manually enter the search space of the algorithm. For users who are not familiar with quantitative optimization, the configuration complexity is high.**

**The automatic optimization acceleration feature is designed to solve the preceding problems and accelerates the automatic optimization process through three key optimization points. This feature brings the following benefits to users:**

1. **Computing resource saving: Intelligently skips invalid evaluation (such as garbled character detection), saving computing resources and reducing optimization costs. Expect to save 10 - 30% of the evaluation time.**
2. **Improving optimization reliability: Resumable adjustment is supported, avoiding work loss caused by unexpected interruptions and improving optimization reliability. Even if the tuning process is interrupted, it can be recovered from the historical precision cache, avoiding repeated evaluation.**
3. **Simplified user operations: The system automatically selects the optimal policy based on expert experience and does not need to manually enter the search space configuration, reducing configuration complexity and improving optimization efficiency.**

Without this feature, the automatic optimization function can work properly, but the preceding problem occurs, affecting optimization efficiency and user experience.

### 2.2 Feature Scenario Analysis

#### Scenario Trigger Conditions and Objects

1. **Trigger conditions:**
    
     * Users need to quantify large language models or multimodal models.
     * Users want to find a quantization solution that meets precision requirements through automatic optimization.
     * An automatic optimization plan (YAML configuration file) has been configured.
2. **Intended:**
    
     * Model quantification engineer: Have certain knowledge of model quantification and be familiar with the use of the msModelSlim tool.
     * AI application developers: Deploy models on NPUs and have requirements on quantization precision.
3. **Use the following interface:**
    
     * Command line interface:`msmodelslim tune`commanded
     * Configuration file: configuration file of the optimization plan in YAML format.

#### Main Application Scenarios

1. **New model quantization scenario:**
    
     * When a user quantizes a model for the first time, the user needs to find a quantization solution that meets the precision requirements.
     * Sub-scenario: The model structure is known, but the quantization parameters are unknown.
     * Key operations: Configure the optimization plan, start the automatic optimization, wait for the optimization to complete, and obtain the final quantitative configuration.
2. **Model precision optimization scenario:**
    
     * The user has a quantization solution, but the precision does not meet the requirements. The precision needs to be optimized automatically.
     * Sub-scenario: fine-tuning based on the existing quantification solution
     * Key operations: Start optimization, iterative optimization, and verification precision improvement based on the existing solution.
3. **Batch model quantification scenario:**
    
     * Users need to quantify multiple models and want automatic processing.
     * Sub-scenario: Same Model Series but Different Parameters
     * Key operations: Configure optimization plans in batches, perform optimization in parallel or serial mode, and summarize results.

### 2.3 Feature Impact Analysis

The automatic optimization acceleration feature is integrated into the core optimization process of the existing precision feedback automatic optimization function and interacts with the following modules:

1. **Quantification service module: invokes the quantization service to quantify the model.**
2. **Evaluation service module: invokes the evaluation service to evaluate the precision of the quantized model.**
3. **Tuning policy module: Use different tuning policies to generate quantitative configurations.**
4. **History management module: manages the tuning history and precision cache.**
5. **Model adapter module: adapts to interfaces of different model series.**

#### Interaction Analysis with Other Requirements and Features

1. **Interaction with the quantification feature: Automatic optimization depends on the quantification function and requires the quantification service to support multiple quantification configurations.**
2. **Interaction with the evaluation feature: Automatic optimization depends on the evaluation function. The evaluation service must support precision evaluation and pre-check.**
3. **Interaction with the best practice library: After successful tuning, the final quantified configuration can be saved to the best practice library.**
4. **Interaction with the model adapter: The model adapter needs to implement the StandingHighInterface interface.**

#### Platform Difference Analysis

1. **Hardware platform: supports NPU devices (such as the Ascend series). NPU devices must support model quantization and inference.**
2. **Operating system: supports the Linux operating system and requires Python 3.8+.**

#### Compatibility Analysis

1. **Forward compatibility: The automatic optimization function of the new version is compatible with the quantization configuration format of the old version.**
2. **Configuration compatibility: Older tuning plan configuration file formats are supported, but the new format is recommended.**

#### Constraints and Limitations

1. **Model support limitations: Only model families with implemented model adapters are supported.**
2. **Precision evaluation restriction: The vLLM-Ascend must support the service-based startup of quantized models.**
3. **Resource limitation: The optimization process requires sufficient storage space for storing the quantitative model and evaluation results.**

#### 2.3.1 Hardware Limitations

1. **NPU device requirements: NPU devices that support model quantization and inference are required. At least one NPU card is required.**
2. **Memory requirements: The optimization process requires sufficient memory for loading models and performing quantitative calculation. It is recommended that at least 32 GB memory be used.**
3. **Storage requirements: Sufficient storage space is required for storing quantitative models, evaluation results, and historical records. It is recommended that the available space be at least 100 GB.**
4. **Network requirements: Requires a stable network connection if using the Remote Assessment Service**

**Workaround:**

 * If the memory is insufficient, reduce the batch size or use model parallelism to reduce the memory usage.
 * In the case of insufficient storage, you can periodically clean up your history or use external storage.

#### 2.3.2 Technical Limitations

**Operating system: Linux (Ubuntu 20.04+ or CentOS 7+ is recommended)**

**Programming language: Python 3.8+**

**Dependency framework:**

 * PyTorch: used for model loading and quantization
 * vLLM-Ascend: used for service-based model startup and inference.
 * AISbench: used for precision evaluation.

**Workaround:**

 * For unsupported Python versions, it is recommended to use conda or virtualenv to create a virtual environment.
 * If the dependency framework version is incompatible, you are advised to use the specified version by referring to the installation guide.

#### 2.3.3 Impact Analysis on the License

This feature uses the following open source software and technologies:

1. **PyTorch: BSD license, allowing commercial use**
2. **vLLM-Ascend: Apache 2.0 license, allowing commercial use**
3. **AISbench: Apache 2.0 license for commercial use**
4. **Pydantic: MIT license, allowing commercial use**

All third-party open-source software introduced meets the license requirements of the msModelSlim project and does not affect the license compliance of the project.

#### 2.3.4 Impact Analysis on System Performance Specifications

Conditions for running resources based on features:

1. **Memory requirements: 32 GB or higher (64 GB or higher recommended) memory is used for:**
    
     * Model loading: 10 to 50 GB memory may be required depending on model size
     * Quantization Computing: Requires an additional 10-20 GB of memory for the quantization process
     * Assessment Service: 5 to 10 GB of memory required for the assessment service to run
2. **Storage requirements: At least 100 GB available storage space is required. 200 GB or more is recommended. The storage space is used for:**
    
     * Quantization model storage: 10 GB to 50 GB for each quantization configuration model
     * Storage of evaluation results: 10 GB to 50 GB is required for the history and precision cache.
     * Temporary files: Temporary files during tuning may need 20-50 GB
3. **NPU requirements: At least one NPU card is required. Two or more NPU cards are recommended. The NPU cards are used for:**
    
     * Model quantization: The quantization process requires the support of the NPU.
     * Model inference: NPU inference is required during evaluation.

#### 2.3.5 Analysis of Impact on System Reliability Specifications

Assumptions and Constraints on Reliability Counters:

1. **Optimization success rate: Under normal conditions, the optimization success rate must be higher than 80% for the supported model series.**
2. **Resuming reliability at breakpoints: The recovery success rate of the historical precision cache must be higher than 99%.**
3. **Troubleshooting: For common exceptions (such as network interruption and insufficient storage), the system can gracefully degrade the fault or provide clear error prompts.**

#### 2.3.6 Impact on System Compatibility

This feature does not affect the forward compatibility of the system.

1. **Configuration compatibility: The automatic optimization function of the new version is compatible with the quantitative configuration format and optimization plan format of the old version.**
2. **Interface compatibility: Backward compatibility is considered in the design of the automatically optimized interface. The calling mode of the earlier version is still valid.**
3. **Data compatibility: The format design of the historical precision cache considers version compatibility and supports cross-version use.**

#### 2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

1. **Interaction with the quantization feature:**
    
     * Automatic optimization depends on the quantization function. The quantization service needs to support multiple quantitative configurations.
     * Automatic optimization does not affect the manual quantization function. Both functions can coexist.
2. **Interaction with the evaluation feature:**
    
     * Automatic optimization depends on the evaluation function and requires the evaluation service to support precision evaluation and pre-check.
     * Automatic optimization does not affect the manual evaluation function. The two functions can coexist.
3. **Interaction with the Best Practices Library:**
    
     * After the tuning is successful, you can save the final quantification configuration to the best practice library.
     * Configurations in the best practice library can be referenced by automatic tuning policies.
4. **Interaction with Model Adapter:**
    
     * Automatic tuning requires the model adapter to implement the StandingHighInterface interface.
     * For models that do not support automatic tuning, you can still use the manual quantization function.

### 2.4 Analysis on the Implementation Solution of Similar Community/Commercial Software

Currently, the main implementation solutions in the field of automatic model quantification optimization are as follows:

1. **Neural Network Intelligence (NNI) is an open-source automatic machine learning tool developed by Microsoft. It supports model compression and quantitative automatic optimization. It supports multiple optimization algorithms and distributed optimization. However, it is mainly oriented to the PyTorch and TensorFlow frameworks and has limited support for NPU devices.**
2. **msModelSlim automatic optimization: This feature has the following advantages:**
    
     * **High integration: In-depth integration with the msModelSlim tool to support a complete quantification, evaluation, and optimization process.**
     * **NPU optimization: In-depth optimization is performed on Ascend NPU devices to support efficient quantitative inference.**
     * **Intelligent skipping: Uses garbled character detection mechanisms to intelligently skip invalid evaluation, saving computing resources.**
     * **Resumable tuning at breakpoints: Supports historical precision cache restoration, improving the reliability of tuning.**
     * **Expert experience: The optimal policy is automatically selected based on historical experience, improving the optimization success rate.**

Compared with similar solutions, this feature provides in-depth optimization and intelligent optimization policies for NPUs, enabling you to efficiently find a quantization solution that meets precision requirements.

## 3. Feature/Function implementation principles (multiple use cases can be broken down)

### 3.1 Objectives

The automatic optimization acceleration feature aims to accelerate the automatic optimization process by using three key optimization points based on the existing precision feedback automatic optimization function. Specific objectives include:

1. **Resource optimization: Intelligently skips invalid evaluation (such as garbled character detection), saving computing resources and reducing optimization costs. The goal is to save more than 90% of the evaluation time when garbled characters are detected.**
2. **Reliability assurance: Resumable adjustment is supported, avoiding work loss caused by unexpected interruptions and improving reliability of the optimization process. The objective is to achieve the recovery success rate of the historical precision cache over 99%.**
3. **Efficiency improvement: The system automatically selects the optimal policy based on expert experience, simplifying user configuration and improving optimization efficiency. The objective is to reduce the configuration time and improve the optimization success rate.**
4. **Compatibility assurance: All optimization points are compatible with the existing automatic optimization functions and do not affect the use of the existing functions.**

### 3.2 Overall Solution

The automatic optimization acceleration feature is an optimization based on the existing automatic optimization function with precision feedback. The design idea is as follows:

1. **Pre-check optimization: A pre-check mechanism is added to the existing evaluation process. Before formal evaluation, the system checks whether the model output contains garbled characters. If garbled characters are detected, the system skips the dataset evaluation.**
2. **Historical cache optimization: The historical precision cache mechanism is added to the existing optimization process. The evaluated configuration results can be restored from the historical cache, implementing resumable commissioning.**
3. **Policy optimization: The expert_experience policy module is created to automatically obtain the search space of the algorithm based on the expert experience, simplifying user configuration.**

All optimization points are integrated into the existing automatic optimization process, which does not affect the normal use of existing functions.

#### Hardware Selection

 * **NPU: Use Ascend NPUs for model quantization and inference, and make full use of the NPU's quantization acceleration capability.**
 * **Storage device: Use local storage or network storage to store quantitative models and evaluation results.**

#### Algorithm selection

 * **Optimization policy: The standing_high policy is used as the basic optimization policy. In addition, an independent expert_experience policy module is provided to support automatic table query based on expert experience based on the model structure type.**
 * **Precision evaluation: The AISbench is used to evaluate the precision. Multiple evaluation data sets are supported.**
 * **Pre-check mechanism: Uses pre-check mechanisms such as garbled character detection and expected answer check to intelligently skip invalid evaluation.**

#### Architecture Layout

The existing precision feedback automatic optimization function adopts the hierarchical architecture design.

1. **Application layer: AutoTuningApplication, which coordinates the entire optimization process.**
2. **Policy layer: ITuningStrategy, which is responsible for generating quantitative configurations and adjusting policies.**
3. **Service layer: quantification service and evaluation service, responsible for specific quantification and evaluation operations.**
4. **Data layer: history management module, which manages the optimization history and precision cache.**

**The automatic optimization acceleration feature is integrated into the existing architecture in the following ways:**

1. **Pre-check optimization: Add the pre-check mechanism at the evaluation service layer to detect garbled characters before formal evaluation.**
2. **Historical cache optimization: A precision cache mechanism is added to the data layer to support resumable scheduling.**
3. **Policy optimization: The expert_experience policy module is added to the policy layer to automatically obtain search space based on expert experience.**

#### Use Case Breakdown

Based on the scenario analysis and system breakdown, three key use cases are identified. Each use case has a specific impact on the automatic optimization function and needs to implement the following features:

1. **Use case 1: During automatic optimization, if garbled characters are displayed in the model output, the user wants to skip invalid evaluation.**
    
     * **User scenario: During automatic optimization, users find garbled characters in the quantized model output. They want the system to intelligently identify and skip invalid datasets, saving computing resources.**
     * **Impact on the automatic optimization function: Before the evaluation, check whether the model output contains garbled characters. If garbled characters are detected, skip the dataset evaluation.**
     * **Implemented feature: Skipping dataset assessment when dialog garbled characters**
2. **Use case 2: During automatic optimization, if the optimization is interrupted abnormally, you want to restart the system to continue the optimization.**
    
     * **User scenario: If the automatic optimization is interrupted due to unexpected interruptions (such as system faults or manual shutdown), users want to reuse historical records and continue the precision optimization process after the automatic optimization is restarted, avoiding repeated evaluation.**
     * **Impact on the automatic tuning function: Resumable tuning is supported, and the evaluated quantization configuration result can be restored from the historical precision cache.**
     * **Implemented feature: Automatic optimization supports resumable adjustment at breakpoints.**
3. **Use Case 3: When configuring automatic optimization, you want to obtain the search space automatically based on the model structure type.**
    
     * **User scenario: When configuring automatic optimization, users are not familiar with the search space configuration for quantitative optimization and want the system to automatically obtain the search space based on the model structure type (such as MHA/MLA/DSA/SWA/GatedDeltaNet), simplifying configuration operations.**
     * **Impact on the automatic optimization function: Optimization policies based on expert experience are required. Search space can be obtained automatically based on the model structure type.**
     * **Implemented feature: optimization policy based on expert experience**

#### Interconnection Principles

1. **Standardized interfaces: All module interfaces are defined using standardized interfaces, which facilitates expansion and maintenance.**
2. **Unified data format: The configuration and results are saved in the uniform YAML format, facilitating parsing and storage.**
3. **Error processing specifications: unified error processing and log recording mechanism, facilitating fault locating and debugging.**

#### Overall solution architecture

```text
┌─────────────────────────────────────────────────────────────┐
│              User Command Line Interface                    │
│                  msmodelslim tune                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AutoTuningApplication                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Load the tuning plan                             │   │
│  │  2. Initialize the tuning policy                     │   │
│  │  3. Check the historical accuracy cache              │   │
│  │  4. Perform iterative tuning                         │   │
│  │     - Generate quantization configuration            │   │
│  │     - Attempt to restore the history                 │   │
│  │     - Quantize the model                             │   │
│  │     - Evaluate the model accuracy (including pre-check)   │   │
│  │     - Save the tuning history                        │   │
│  │     - Determine whether to continue                  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Optimization Strategy Layer    │ │ Quantization Service Layer    │ │ Evaluation Service Layer    │
│ITuningStrategy│ │IQuantService │ │EvaluateService│
│              │ │              │ │              │
│- standing_high│ │- Model quantization     │ │- Accuracy evaluation     │
│- Expert experience policy  │ │- Configuration generation     │ │- Pre-check       │
└──────────────┘ └──────────────┘ └──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        Historical Management Module (Data Layer)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - Accuracy cache management (accuracy.yaml)         │   │
│  │  - History management (history.yaml)                 │   │
│  │  - Configuration file management (practice configs)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Figure 1: Overall implementation principle

## 4. Use Case 1 Implementation

### 4.1 Use Case Description

**Use Case Name: Garbled Characters Skipping Dataset Evaluation**

**Use case scenario:**

 * Garbled characters are displayed in the quantized model output during automatic optimization.
 * Users want the system to intelligently identify garbled characters and skip invalid dataset evaluation, saving computing resources.
 * The system checks whether the model output contains garbled characters before the formal evaluation.
 * If garbled characters are detected, the system skips the data set evaluation and directly returns the evaluation result with the precision of 0.

**Impact on the automatic optimization function:**

 * A pre-check mechanism needs to be added to the assessment process.
 * Garbled character detection must be implemented.
 * The logic for skipping invalid evaluation must be supported.

**Implemented feature: Skipping dataset assessment when dialog garbled characters**

### 4.2 Feature Design Ideas

If the quantized model output is garbled characters during model precision evaluation, a large amount of computing resources will be wasted if the complete dataset evaluation is continued. Therefore, this feature checks whether the model output contains garbled characters before formal evaluation. If garbled characters are detected, the data set evaluation is skipped and the evaluation result with the precision of 0 is returned.

The design idea includes:

1. **Pre-check mechanism: Before formal evaluation, test messages are sent to check whether the model output meets the expectation.**
2. **Garbled character detection: Use multiple check items. (empty text, repeated characters, normal character ratio, control characters, repetition pattern, etc.) Check whether the model output contains garbled characters.**
3. **Intelligent skip: If garbled characters are detected, the dataset evaluation is skipped and the evaluation result with the precision of 0 is returned, saving computing resources.**

### 4.3 Constraints

1. **Model servitization requirements: Models must be started in servitized mode through vLLM-Ascend and support API invoking.**
2. **Precheck configuration requirements: The precheck field must be configured in the optimization plan configuration file to specify the garbled character detection test cases.**
3. **Network requirements: Requires a stable network connection if the assessment service is running remotely**

### 4.4 Detailed implementation (module-level or process-level message sequence diagram from user entry)

#### Handling Procedure

```text
用户启动自动调优
   │
   ▼
AutoTuningApplication.tune()
   │
   ▼
评估服务启动模型服务化
   │
   ▼
EvaluateService.evaluate()
   │
   ├─→ 检查是否配置了precheck
   │   │
   │   ├─→ 是：执行预检查
   │   │   │
   │   │   ├─→ GarbledTextRule.check()
   │   │   │   │
   │   │   │   ├─→ 遍历测试用例
   │   │   │   │   │
   │   │   │   │   ├─→ test_chat_via_api() 发送测试消息
   │   │   │   │   │
   │   │   │   │   ├─→ is_garbled_text() 检测乱码
   │   │   │   │   │   │
   │   │   │   │   │   ├─→ 空文本检查 (EmptyTextCheckItem)
   │   │   │   │   │   ├─→ 重复字符检查 (RepeatedCharCheckItem)
   │   │   │   │   │   ├─→ 正常字符比例检查 (NormalCharRatioCheckItem)
   │   │   │   │   │   ├─→ 控制字符检查 (ControlCharCheckItem)
   │   │   │   │   │   └─→ 重复模式检查 (RepeatedPatternCheckItem)
   │   │   │   │   │
   │   │   │   │   └─→ 如果检测到乱码：返回精度为0的评估结果
   │   │   │   │
   │   │   │   └─→ 如果所有测试用例通过：继续执行正式评估
   │   │   │
   │   │   └─→ 否：直接执行正式评估
   │   │
   │   └─→ 执行正式数据集测评
   │
   └─→ 返回评估结果
```

#### Module Interaction Description

1. **AutoTuningApplication: coordinates the entire optimization process and invokes the evaluation service to evaluate the precision.**
2. **EvaluateService: evaluates model precision and checks whether precheck is configured before performing formal evaluation.**
3. **GarbledTextRule: implements garbled character detection pre-check rules and uses multiple check items to check whether the model output is garbled characters.**
4. **GarbledTextCheckItem: implements various garbled character check items, including empty text, repeated characters, normal character ratio, control character, and repetition mode.**

### 4.5 Interfaces Between Subsystems (Mainly Covering the Definition of Module Interfaces)

#### New Interface

1. **GarbledTextPrecheckConfig (**`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
    
     * Type: Pydantic BaseModel
     * Run the following command to configure garbled character detection precheck, including the test case list:
     * Field:
        
         * `type`\: Literal\["garbled_text"\], fixed value
         * `test_cases`\: Optional\[List\[TestCaseConfig\]\], test case list.
2. **GarbledTextRule (**`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
    
     * Type: BasePrecheckRule subclass
     * Function: Implements garbled character detection pre-check rules.
     * Method:
        
         * `is_garbled_text(text: str, check_items: List[str]) -> bool`\: Check whether the text is garbled.
         * `check(host: str, port: int, served_model_name: str, datasets: List[str]) -> Optional[List[EvaluateAccuracy]]`\: Perform garbled character detection precheck.
3. **GarbledTextCheckItem (**`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
    
     * Type: ABC abstract base class
     * Function: Garbled character check item interface
     * Subcategory:
        
         * `EmptyTextCheckItem`\: Empty Text Check
         * `RepeatedCharCheckItem`\: Check for duplicate characters.
         * `NormalCharRatioCheckItem`\: Check the normal character ratio.
         * `ControlCharCheckItem`\: Control character check
         * `RepeatedPatternCheckItem`\: Check the duplicate pattern.

#### Modifying an Interface

1. **BasePrecheckRule.check() (**`msmodelslim/infra/evaluation/precheck/base.py`)
    
     * Function extension: The precision evaluation result can be returned. If the pre-check fails, the precision 0 is returned.

### 4.6 Subsystem LLD

#### 4.6.1 Garbled Character Detection Check Item Design

Garbled character detection uses the responsibility chain mode and supports the combination of multiple check items.

1. **EmptyTextCheckItem: checks whether the text is empty.**
    
     * Implementation: Check whether the text is empty after whitespace is removed.
     * Threshold: None
2. **RepeatedCharCheckItem: Check whether a large number of consecutive repeated characters exist.**
    
     * Implementation: Collect statistics on the maximum length of consecutive repeated characters in a text. If the length exceeds 30% of the text length, garbled characters are identified.
     * Threshold: 0.3 (configurable)
3. **NormalCharRatioCheckItem: Check whether the proportion of normal characters is too low.**
    
     * Implementation: Collect statistics on the proportion of Chinese and English characters, digits, and common punctuations. If the proportion is lower than 50%, garbled characters are regarded.
     * Threshold: 0.5 (configurable)
4. **ControlCharCheckItem: Checks whether a large number of control characters are contained.**
    
     * Implementation: Calculate the proportion of control characters (excluding line feed, carriage return, and tab characters). If the proportion exceeds 10%, garbled characters are identified.
     * Threshold: 0.1 (configurable)
5. **RepeatedPatternCheckItem: Checks for obvious duplicate patterns.**
    
     * Implementation: Check whether the pattern at the beginning of the text appears repeatedly in the text. If the number of repetitions reaches the threshold and the proportion of the repeated part to the total length reaches the threshold, the text is regarded as garbled characters.
     * Threshold: min_pattern_count=3, min_pattern_ratio=0.5 (configurable)

#### 4.6.2 Pre-check Process Design

1. **Configuration parsing: Parse the precheck configuration from the optimization plan configuration file and create the GarbledTextPrecheckConfig object.**
2. **Test case execution: Traverse the configured test cases and perform the following operations for each test case:**
    
     * Send test messages to the model service through the API.
     * Response for obtaining a model
     * Use the configured check items to check whether the response contains garbled characters.
     * If garbled characters are detected, the system records a warning log and returns the evaluation result with the precision of 0.
3. **Continue the assessment: If all test cases pass, proceed with the formal assessment of the dataset.**

### 4.7 DFX Attribute Design

#### 4.7.1 Performance Design

1. **Pre-check overhead: Only a few test messages (usually one to three messages) are sent. The overhead is low. Compared with the complete data set assessment, the pre-check can save more than 90% of the time.**
2. **Check item performance: The implementation of various check items is O(n) time complexity, where n indicates the text length. The performance overhead is negligible.**
3. **Impact on existing features: The precheck is optional. If the precheck is not configured, the existing evaluation process is not affected.**

#### 4.7.2 Upgrade and Capacity Expansion Design

1. **Configuration compatibility: The precheck function of the new version is compatible with the configuration file format of the old version. If the precheck field is not configured, the precheck is skipped.**
2. **Extensibility of check items: New check items are supported through the registration mechanism, which does not affect the use of existing check items.**

#### 4.7.3 Exception Handling Design

1. **API invoking exception: If the test message fails to be sent, the system records a warning log and proceeds to the next test case without interrupting the evaluation process.**
2. **Check item exception: If a check item fails, the system records a warning log and proceeds to the next check item.**
3. **Configuration exception: If the precheck configuration format is incorrect, the system records the error log, skips the precheck, and continues the formal evaluation.**

#### 4.7.4 Resource Management Design

1. **Memory usage: Only a small amount of memory is required to store test messages and responses. The memory usage is negligible.**
2. **Network I/O: A small number of HTTP requests need to be sent during the pre-check, and the network I/O overhead is small.**
3. **Computing resources: The computing overhead of the precheck is small and does not affect the system performance.**

#### 4.7.5 Miniaturized Design

This feature does not affect the specifications of the miniaturized version. The pre-check function is lightweight and consumes little memory and CPU resources.

#### 4.7.6 Testability Design

The test should cover the following areas:

1. **Function test:**
    
     * Normal text passes all check items
     * Empty text is detected as garbled characters.
     * Duplicate character text is detected as garbled characters.
     * The control character text is detected as garbled characters.
     * Duplicate mode text detected as garbled
     * Mixed garbled text is detected as garbled characters.
2. **Boundary Value Test:**
    
     * The text length is 0.
     * The text length is 1.
     * Very large text length (> 10000 characters)
     * Threshold value of the check item
3. **Abnormal scenario test:**
    
     * Failed to invoke the API.
     * Check Item Execution Failure
     * Incorrect configuration format.
     * Network interruption
4. **Performance test:**
    
     * Pre-check time-consuming test
     * Performance test with a large number of test cases

#### 4.7.7 Security Design

##### 4.7.7.1 Safety Design Qualification

| Security attributes                | Check Item                                                                                                                                                   | Check Item Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Involved or Not | Satisfied or not |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------- |
| Access channel control             | Whether to add a listening port                                                                                                                              | The communication matrix needs to be updated for new listening ports.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | No.             | Not involved.    |
| Access channel control             | Whether to add new processes or communication between components                                                                                             | Added the communication matrix between new processes or components.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Yes             | Satisfied        |
| Access channel control             | Whether to add an authentication mode                                                                                                                        | The communication matrix and product documentation must be updated for the new authentication mode.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | No.             | Not involved.    |
| Permission control                 | Whether files or directories need to be created                                                                                                              | To create a file or directory, you must explicitly specify the access permission for the file or directory.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | No.             | Not involved.    |
| Permission control                 | Check whether the account permission meets the "minimum permission principle".                                                                               | All accounts in the system must be assigned with the least permission.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Yes             | Satisfied        |
| Permission control                 | Whether user privilege escalation exists                                                                                                                     | Illegal user privilege escalation is prohibited.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | No.             | Not involved.    |
| Undisclosed Interface              | Whether to add GUC parameters                                                                                                                                | The product documentation needs to be updated when GUC parameters are added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | No.             | Not involved.    |
| Undisclosed Interface              | Add or modify functions, views, and system tables.                                                                                                           | When adding or modifying functions, views, and system tables, the product documentation must be updated and permission control must be considered.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | No.             | Not involved.    |
| Undisclosed Interface              | Add SQL Syntax                                                                                                                                               | The new SQL syntax needs to be updated in the product documentation to support recording audit logs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | No.             | Not involved.    |
| Undisclosed Interface              | Whether to add internal tools                                                                                                                                | Product documentation must be updated for new internal tools.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No.             | Not involved.    |
| Undisclosed Interface              | Check whether the script contains comment code.                                                                                                              | Do not comment out code in explanatory languages such as Shell and Python. The comment out code must be deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | No.             | Not involved.    |
| Undisclosed Interface              | Check whether there are access modes such as hidden commands, parameters, and ports.                                                                         | Access modes, such as commands, parameters, and ports, that are not used during maintenance on the live network (including but not limited to product production, commissioning, and maintenance purposes), must be deleted (e.g. by compiling macros)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | No.             | Not involved.    |
| Undisclosed Interface              | Check whether the system has hidden backdoors.                                                                                                               | Do not reserve any undisclosed accounts in the system. All accounts must be managed by the system and must be described in the documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No.             | Not involved.    |
| Undisclosed Interface              | It is prohibited to provide cracking and network sniffing tools in the software (including software packages and patch packages) released to external users. | 1. It is prohibited to provide the software (including software packages and patch packages) released to external users that can change the password of any user or have the "password cracking capability". (Virtual-force password cracking and maliciously cracking passwords by exploiting system/algorithm vulnerabilities) 2. A function or tool used to decrypt files containing sensitive data (such as configuration files and databases containing keys). 2. Do not retain third-party network sniffing tools, such as tcpdump, gdb, strace, readelf, and process debugging tools, in the system. CPP, GCC, dexdump, mirror, JDK development/compilation tools, and self-developed debugging tools/scripts used only in the commissioning phase (for example, encryption and decryption scripts, commissioning functions, and commands that can be used only in the commissioning phase), which must be retained due to service requirements, and strict access control is required. In addition, describe the reason, application scenario, and risk for the retention. | No.             | Not involved.    |
| Sensitive data protection          | Authentication credentials cannot be stored in the system in plaintext and must be encrypted.                                                                | Authentication credentials (such as passwords and private keys) must be encrypted and cannot be stored in the system in plaintext.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | No.             | Not involved.    |
| Sensitive data protection          | The key used for encrypting sensitive data transmission cannot be hard-coded.                                                                                | Hard coding of passwords and keys is prohibited.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | No.             | Not involved.    |
| Sensitive data protection          | Check whether sensitive information, such as passwords and keys, is printed in plaintext.                                                                    | Do not display sensitive information (passwords, private keys, and pre-shared keys) in plaintext in logs, debugging information, error messages, and ps commands stored in the system.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | No.             | Not involved.    |
| Sensitive data protection          | Specifies whether to display the password in plaintext.                                                                                                      | Do not display passwords in plaintext.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | No.             | Not involved.    |
| Sensitive data protection          | Whether the default passwords of third-party and open-source software are used                                                                               | Do not use the default passwords of third-party and open-source software. For details, see section 1.5 in the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | No.             | Not involved.    |
| Sensitive data protection          | Indicates whether to store passwords in plaintext in configuration files.                                                                                    | Plaintext passwords cannot be written into configuration files. (except the scenario where the password must be configured during the installation, deployment, and use of the command-line tool.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | No.             | Not involved.    |
| Sensitive data protection          | Whether to use insecure encryption algorithms                                                                                                                | Do not use proprietary or insecure encryption algorithms. Recommended Encryption Algorithm Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | No.             | Not involved.    |
| Sensitive data protection          | Check whether sensitive information, such as passwords, is transmitted over secure channels.                                                                 | Sensitive information must be transmitted between untrusted networks through secure transmission channels or encrypted transmission. For details, see chapter 10 of the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Yes             | Satisfied        |
| Sensitive data protection          | Check whether sensitive information such as passwords and keys in the memory is destroyed after being used.                                                  | The passwords or keys in the memory are cleared immediately after being used.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No.             | Not involved.    |
| Sensitive data protection          | The random number used in cryptographic algorithm must be the cryptographically defined secure random number.                                                | The random number used in the cryptographic algorithm must be the cryptographic secure random number. For details, see section 6.3 in the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | No.             | Not involved.    |
| Sensitive data protection          | Check whether there are insecure examples in the documentation.                                                                                              | The examples in the documentation must be secure and provide correct guidance for users. If the examples contain potential risks, describe the risks in the documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Yes             | Satisfied        |
| Certification                      | Provide authentication mechanism                                                                                                                             | The new system needs to provide the authentication mechanism and the authentication mechanism is enabled by default.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | No.             | Not involved.    |
| Certification                      | Indicates whether authentication is performed on the server.                                                                                                 | The authentication process needs to be performed on the server.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | No.             | Not involved.    |
| Certification                      | Indicates whether the server returns valid information after the authentication fails.                                                                       | After the authentication fails, the information returned by the server does not provide detailed information that can be used to locate the error cause.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | No.             | Not involved.    |
| External parameter validation      | Indicates whether to verify the validity of external input.                                                                                                  | 1. If external input data is used as the loop termination condition, array subscript, and memory allocation parameter, infinite loop, buffer overflow, memory overwriting, and DoS may occur. 2. The validity of external input, such as file paths, must be verified to prevent injection risks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Yes             | Satisfied        |
| Third-party component introduction | Third-party components are introduced.                                                                                                                       | 1. New third-party components must be scanned by using secure compilation options, viruses, vulnerabilities, open source fragment reference, license compliance, and open source components. For details, see the version release cyber security quality requirements. 2. The source of the new third-party components must be trusted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | No.             | Not involved.    |

##### 4.7.7.2 Sensitive Data Analysis

This use case does not involve sensitive data processing. It mainly detects garbled characters in the model output text and does not involve sensitive operations such as user authentication and key management.

##### 4.7.7.3 Design Implementation

The security design of this use case is as follows:

1. **External input verification: Validity of test messages and model responses is verified to prevent injection attacks.**
2. **Network communication security: If the assessment service is running remotely, use a secure transmission channel such as HTTPS.**
3. **Error information processing: The error information does not disclose sensitive information. Only necessary debugging information is recorded.**

### 4.8 External Interfaces

This use case does not affect the external interfaces of the system. It is mainly used for internal implementation optimization. You can enable or disable garbled character detection by setting the precheck field in the optimization plan configuration file.

### 4.9 Self-Test Case Design

The self-test case design is as follows:

1. **Normal text test:**
    
     * Input: Normal Chinese text "Hello, World"
     * Expected result: All check items are passed and the formal evaluation continues.
2. **Empty text test:**
    
     * Input: an empty string or a string containing only white space characters
     * Expected result: Garbled characters are detected by EmptyTextCheckItem and the evaluation result with the precision of 0 is returned.
3. **Duplicate character test:**
    
     * Input: text that contains a large number of consecutive repeated characters, such as "aaaaaaaaaa..."
     * Expected result: Garbled characters are detected by RepeatedCharCheckItem and the evaluation result with the precision of 0 is returned.
4. **Control character test:**
    
     * Input: Text with a large number of control characters
     * Expected result: Garbled characters are detected by ControlCharCheckItem and the evaluation result with the precision of 0 is returned.
5. **Repeat mode test:**
    
     * Input: Text that contains obvious repetition patterns, such as "abcabcabc..."
     * Expected result: Garbled characters are detected by RepeatedPatternCheckItem and the evaluation result with the precision of 0 is returned.
6. **Mixed garbled characters test:**
    
     * Input: text with multiple garbled characters
     * Expected result: Garbled characters are detected by at least one check item and the evaluation result with the precision of 0 is returned.
7. **API invoking failure test:**
    
     * Input: Simulate an API invoking failure.
     * Expected result: The warning log is recorded and the next test case continues without interrupting the evaluation process.
8. **Configuration error test:**
    
     * Input: Incorrect precheck configuration format.
     * Expected result: The error log is recorded, the precheck is skipped, and the formal evaluation continues.

## 5. Use Case 2 Implementation

### 5.1 Use Case Description

**Use Case Name: Resuming Precision Optimization After Abnormal Interruption**

**Use case scenario:**

 * During the automatic optimization, the optimization is interrupted due to unexpected faults, such as system faults and manual stop.
 * Users want to reuse historical records after automatic calibration is started again to continue the precision calibration process.
 * The system automatically detects the historical precision cache on restart.
 * If the historical precision cache exists, the system reuses the evaluated quantitative configuration result to avoid repeated evaluation.

**Impact on the automatic optimization function:**

 * The function of resuming the resumable call must be supported.
 * The historical precision cache mechanism needs to be implemented.
 * Recovering the evaluated configuration results from the historical cache is required.

**Implemented feature: Automatic optimization supports resumable adjustment at breakpoints.**

### 5.2 Feature Design Ideas

During automatic optimization, if the optimization is interrupted due to unexpected interruptions (such as system faults and manual shutdown), the evaluated quantitative configuration results must be restored from the historical precision cache during restart, avoiding repeated evaluation of the same configuration. Implements the breakpoint continuation of the optimization process.

The design idea is as follows:

1. **Precision cache mechanism: The quantization configuration and precision evaluation result of each iteration are saved to the precision cache, and the MD5 hash value is used as the unique ID of the configuration.**
2. **History detection mechanism: When optimization starts, the system checks whether the history precision cache exists. If yes, the system loads the cache to the memory.**
3. **Cache reuse mechanism: In each iteration, the system searches the precision cache for the matching quantization configuration. If the matching quantization configuration is found, the system uses the historical evaluation result and skips the quantization, servitization startup, precheck, and evaluation steps.**

### 5.3 Constraints

1. **Storage requirements: Sufficient storage space is required to store the precision cache. The precision cache file must be in YAML format.**
2. **Path consistency: The same save_path must be used for resumable callout. The system searches for the precision cache in the save_path/history directory.**
3. **Configuration consistency: The quantification configuration must be the same (matched by the MD5 hash value). If the configuration changes, it cannot be reused.**

### 5.4 Detailed implementation (module level or process level message sequence diagram from user entry)

#### Handling Procedure

```text
用户启动自动调优（可能为断点续调）
   │
   ▼
AutoTuningApplication.tune()
   │
   ▼
TuningHistoryManager.load_history()
   │
   ├─→ 检测save_path/history目录
   │   │
   │   ├─→ 如果存在accuracy.yaml：加载精度缓存到内存
   │   │
   │   └─→ 如果不存在：创建空的精度缓存
   │
   ▼
YamlTuningHistory._load_accuracy_database()
   │
   ├─→ 读取accuracy.yaml文件
   │
   ├─→ 解析为字典格式（key为MD5，value为评估结果）
   │
   └─→ 加载到内存中的_accuracy_cache
   │
   ▼
开始迭代调优循环
   │
   ├─→ 生成量化配置 (PracticeConfig)
   │
   ├─→ 计算配置的MD5哈希值
   │
   ├─→ 在精度缓存中查找匹配的配置
   │   │
   │   ├─→ 如果找到：直接使用历史评估结果，跳过量化、评估步骤
   │   │
   │   └─→ 如果未找到：执行量化、评估步骤
   │       │
   │       ├─→ 量化模型
   │       │
   │       ├─→ 评估模型精度
   │       │
   │       └─→ 保存到精度缓存
   │
   └─→ 继续下一次迭代
```

#### Module Interaction Description

1. **AutoTuningApplication: coordinates the entire tuning process, loads the historical precision cache at the beginning of tuning, and attempts to restore the evaluation result from the cache during each iteration.**
2. **TuningHistoryManager: manages the tuning history, loads and saves the precision cache.**
3. **YamlTuningHistory: implements YAML-based precision cache management, including loading, saving, and querying.**
4. **calculate_practice_md5: calculates the MD5 hash value of the quantization configuration, which is used to uniquely identify the configuration.**

### 5.5 Interfaces Between Subsystems (Mainly Covering the Interface Definition of Modules)

#### New Interface

1. **TuningHistoryInfra (**`msmodelslim/app/auto_tuning/practice_history_infra.py`)
    
     * Type: ABC abstract base class
     * Function: Interface for optimizing historical operations
     * Method:
        
         * `get_accuracy(practice: PracticeConfig) -> Optional[EvaluateResult]`\: Obtain the precision evaluation result from the history.
         * `append_history(practice: PracticeConfig, evaluation: EvaluateResult) -> None`\: Append History
         * `clear_records() -> None`\: Clear history (but retain precision cache)
         * `get_accuracy_count() -> int`\: Return the number of precision records.
2. **TuningHistoryManagerInfra (**`msmodelslim/app/auto_tuning/practice_history_infra.py`)
    
     * Type: ABC abstract base class
     * Function: Tuning History Manager Interface
     * Method:
        
         * `load_history(database: str) -> TuningHistoryInfra`\: Load the optimization history.
3. **YamlTuningHistory (**`msmodelslim/infra/yaml_practice_history_manager.py`)
    
     * Type: TuningHistoryInfra implementation class
     * Function: YAML-based historical optimization implementation
     * Data file:
        
         * `accuracy.yaml`\: precision cache. The key is MD5, and the value is the evaluation result.
         * `history.yaml`\: historical index, which records the configuration ID and evaluation result of each iteration.
4. **calculate_practice_md5 (**`msmodelslim/infra/yaml_practice_history_manager.py`)
    
     * Type: Function
     * Run the following command to calculate the MD5 hash value of the quantization configuration:
     * Implementation: Serialize the configuration into a JSON string and calculate the MD5 hash value.

#### Modifying an Interface

1. **AutoTuningApplication._tune() (**`msmodelslim/app/auto_tuning/application.py`)
    
     * Extended function: Load the historical precision cache at the start of optimization and try to restore the evaluation result from the cache during each iteration.

### 5.6 Detailed Design of Subsystems

#### 5.6.1 Precision Cache Design

The precision cache is stored in the YAML format. The structure is as follows:

```yaml
accuracy:
  <md5_hash_1>:
    accuracies:
      - dataset: dataset1
        accuracy: 0.85
      - dataset: dataset2
        accuracy: 0.90
    is_satisfied: true
  <md5_hash_2>:
    accuracies:
      - dataset: dataset1
        accuracy: 0.75
    is_satisfied: false
```

The key of the precision cache is the MD5 hash value of the quantization configuration, and the value is the evaluation result (the dictionary after the serialization of the EvaluateResult object).

#### 5.6.2 Historical Index Design

Historical indexes are stored in YAML format. The structure is as follows:

```yaml
records:
  - practice_id: standing_high_0
    evaluation:
      accuracies:
        - dataset: dataset1
          accuracy: 0.85
      is_satisfied: true
    md5: <md5_hash_1>
    time: "2026-01-22 10:00:00"
  - practice_id: standing_high_1
    evaluation:
      accuracies:
        - dataset: dataset1
          accuracy: 0.75
      is_satisfied: false
    md5: <md5_hash_2>
    time: "2026-01-22 10:05:00"
```

The history index records the configuration ID, evaluation result, MD5 hash value, and timestamp for each iteration.

#### 5.6.3 Cache reuse mechanism design

1. **Configuration matching: Use the MD5 hash value to match the quantized configuration to ensure that the configurations are the same.**
2. **Cache lookup: At each iteration, the MD5 hash value of the current configuration is calculated first, and then the matching configuration is found in the precision cache.**
3. **Result reuse: If a matching configuration is found, the historical evaluation result is used and the quantification, service-based startup, pre-check, and evaluation steps are skipped.**
4. **Cache update: If no matching configuration is found, the result is saved to the precision cache after the quantization and evaluation steps are performed.**

#### 5.6.4 Design of Breakpoint Resume Process

1. **Historical check: When the optimization starts, check whether the accuracy.yaml file exists in the save_path/history directory.**
2. **Cache loading: If the loading precision exists, the loading precision is cached in the memory. If it does not exist, create an empty precision cache.**
3. **Iterative recovery: In each iteration, the evaluation result is first recovered from the precision cache. If the evaluation result is found, the evaluation result is reused. If the evaluation result is not found, the complete quantization and evaluation process is executed.**
4. **Cache storage: After each iteration, the evaluation result is saved to the precision cache to ensure that the evaluation result can be reused during the next startup.**

### 5.7 DFX Attribute Design

#### 5.7.1 Performance Design

1. **Cache loading performance: The precision cache is loaded to the memory, and the search performance is O(1). The optimization performance is not affected.**
2. **MD5 calculation performance: The MD5 hash calculation overhead is small and does not affect the optimization performance.**
3. **Impact on the existing features: The resumable tuning function is optional. If the same save_path is not used, the existing tuning process is not affected.**

#### 5.7.2 Upgrade and Capacity Expansion Design

1. **Data format compatibility: The precision cache data format design considers version compatibility and supports cross-version use.**
2. **Storage scalability: The precision cache uses the YAML format, which is easy to expand and maintain.**

#### 5.7.3 Design for Exception Handling

1. **File reading exception: If the precision cache file fails to be read, an alarm log is recorded and an empty precision cache is created. The optimization process is not interrupted.**
2. **Data parsing exception: If the format of the precision cache data is incorrect, an error log is recorded and an empty precision cache is created. The optimization process is not interrupted.**
3. **Storage exception: If the precision cache fails to be saved, an error log is recorded but the optimization process is not interrupted. If the precision cache fails to be saved, the optimization process may not be interrupted.**

#### 5.7.4 Resource Management Design

1. **Memory usage: The precision cache is loaded to the memory. The memory usage depends on the cache size and ranges from several MB to dozens of MB.**
2. **Disk I/O: Read and write operations of precision cache are low, and disk I/O overhead is low.**
3. **Storage space: The size of the precision cache file depends on the number of evaluation results, typically a few KB to a few MB**

#### 5.7.5 Miniaturized Design

This feature does not affect the specifications of the miniaturized version. The precision cache function is lightweight and occupies little memory and storage space.

#### 5.7.6 Testability Design

The test should cover the following areas:

1. **Function test:**
    
     * First Tuning: Creating an Empty Precision Cache
     * Resumable: Restore the evaluation result from the historical precision cache.
     * Configuration matching: The same configuration can be correctly matched.
     * Configuration mismatch: Different configurations cannot be matched.
2. **Boundary Value Test:**
    
     * The precision cache is empty.
     * The precision cache contains a large number of records (> 1000)
     * MD5 hash conflict (not theoretically possible, but needs to be tested)
3. **Abnormal scenario test:**
    
     * The precision cache file does not exist.
     * Incorrect precision cache file format.
     * Failed to read the precision cache file.
     * Saving the precision cache file failed.
     * Insufficient storage space.
4. **Performance test:**
    
     * Precise cache loading time test
     * Precise Cache Search Duration Test
     * Cache performance test for a large number of records

#### 5.7.7 Security Design

##### 5.7.7.1 Safety Design Qualification

The security design of this use case is similar to that of use case 1. The use case focuses on the security of file operations and does not involve sensitive data processing.

##### 5.7.7.2 Sensitive Data Analysis

This use case does not involve sensitive data processing. It mainly involves storage and query of quantitative configuration and evaluation results. It does not involve sensitive operations such as user authentication and key management.

##### 5.7.7.3 Design Implementation

The security design of this use case is as follows:

1. **File permission control: Secure file permissions are used for precision cache files to prevent unauthorized access.**
2. **Data integrity: Use MD5 hashes to ensure uniqueness and integrity of the configuration**
3. **Error handling: The error information does not disclose sensitive information and only necessary debugging information is recorded.**

### 5.8 External Interfaces

This use case does not affect the external interfaces of the system. It is mainly used for internal implementation optimization. You can use the same save_path to implement the resumable function.

### 5.9 Self-Test Case Design

The self-test case design is as follows:

1. **First optimization test:**
    
     * Input: new save_path. No historical precision cache exists.
     * Expected result: An empty precision cache is created and the optimization process is executed normally.
2. **Resuming test at breakpoints:**
    
     * Input: The same save_path and historical precision cache exist.
     * Expected result: Load the historical precision cache and reuse the evaluated configuration result during iteration.
3. **Test the configuration matching:**
    
     * Input: Same quantization configuration
     * Expected result: The matching is correct and historical evaluation results can be reused.
4. **Test the configuration mismatch.**
    
     * Input: Different quantization configurations
     * Expected: No matching. Complete quantification and evaluation processes are performed.
5. **Cache file abnormality test:**
    
     * Input: The precision cache file does not exist or the format is incorrect.
     * Expected result: Warning logs are recorded and an empty precision cache is created. The optimization process is not interrupted.
6. **Massive Record Test:**
    
     * Input: The precision cache contains a large number of records (> 1000).
     * Expected result: The loading and search can be performed properly, and the performance meets the requirements.

## 6. Use Case 3 Implementation

### 6.1 Use Case Description

**Use Case Name: Built-in Expert Experience in Optimization Policies**

**Use case scenario:**

 * Users are not familiar with the search space configuration for quantitative optimization when configuring automatic optimization.
 * Users expect the system to automatically query tables based on the model structure type (such as MHA/MLA/DSA/SWA/GatedDeltaNet) to obtain the search space of the algorithm.
 * The system automatically obtains the search space of the recommendation algorithm based on the model structure type, simplifying user configuration operations.

**Impact on the automatic optimization function:**

 * Optimization policies based on expert experience must be provided.
 * The model structure type identification function needs to be implemented.
 * The expert experience table mechanism needs to be implemented.
 * Automatic table query is supported to obtain the search space.

**Implemented feature: optimization policy based on expert experience**

### 6.2 Feature Design Ideas

Currently, the standing_high policy requires users to manually enter the search space (anti_outlier_strategies) of the algorithm. For users who are not familiar with quantitative optimization, the configuration complexity is high. This feature creates an independent optimization policy module.`expert_experience`In, the system automatically queries tables based on the model structure type (such as MHA/MLA/DSA/SWA/GatedDeltaNet) to obtain the search space of the algorithm. You do not need to manually enter the search space configuration.

The new policy module can reuse the core logic of the standing_high policy (such as the standing_high algorithm). However, the new policy module is implemented as a separate policy and has the following characteristics:

1. **standalone module:creating a new policy catalog**`msmodelslim/core/tune_strategy/expert_experience/`, including independent policy configuration and implementation.
2. **Model structure type recognition: supports the recognition of model attention mechanism types, such as Multi-Head Attention (MHA), Multi-Head Latent Attention (MLA), Distributed Sparse Attention (DSA), Sliding Window Attention (SWA), and GatedDeltaNet.**
3. **Expert experience table: Maintain an expert experience table, which records the search space of the recommendation algorithm corresponding to different model structure types.**
4. **Automatic table lookup: Automatically obtains the search space of the algorithm based on the model structure type. If no matching type is found, the default search space is used.**
5. **Policy reuse: The algorithm logic of the standing_high policy can be reused. However, the search space is automatically obtained through the expert experience table, simplifying user configuration.**

### 6.3 Constraints

1. **Model structure support: The model adapter must be able to identify and provide model structure type information.**
2. **Expert experience table maintenance: The expert experience table needs to be maintained to record the search space of the recommendation algorithm corresponding to different model structure types.**
3. **Policy registration: New policies need to be registered in setup.py, including policy configuration and entry points for policy implementation.**
4. **Backward compatibility: The new policy does not affect the existing standing_high policy. The two policies can coexist.**

### 6.4 Detailed implementation (module level or process level message sequence diagram from user entry)

#### Handling Procedure

```text
用户启动自动调优（使用expert_experience策略）
   │
   ▼
ExpertExperienceStrategy.__init__()
   │
   ├─→ 获取模型结构类型
   │   │
   │   └─→ ModelAdapter.get_attention_type()
   │
   ├─→ 在专家经验表中查找匹配的类型
   │   │
   │   └─→ ExpertExperienceTable.get_search_space(attention_type)
   │
   ├─→ 如果找到：使用查表结果作为anti_outlier_strategies
   │
   └─→ 如果未找到：使用默认搜索空间
   │
   ▼
创建StandingHighStrategy实例（复用摸高算法逻辑）
   │
   ├─→ 使用自动获取的anti_outlier_strategies
   │
   └─→ 执行standing_high策略的摸高算法
      │
      └─→ 使用自动获取的搜索空间进行调优
```

#### Module Interaction Description

1. **ExpertExperienceStrategy: new optimization strategy implementation, which automatically obtains the search space of the algorithm based on the model structure type.**
2. **ExpertExperienceTable: expert experience table, used to maintain the search space of the recommendation algorithm corresponding to different model structure types.**
3. **ModelAdapter: model adapter, which provides model structure type information.**
4. **StandingHighStrategy: Implemented by the standing_high policy, which can be reused by ExpertExperienceStrategy.**

### 6.5 Interfaces Between Subsystems (Mainly Covering the Definition of Module Interfaces)

#### New Interface

1. **ExpertExperienceStrategyConfig (**`msmodelslim/core/tune_strategy/expert_experience/strategy.py`)
    
     * Type: StrategyConfig subcategory
     * Function: configures the expert experience policy. The configuration item of StandingHighStrategyConfig is inherited, but the anti_outlier_strategies field is optional.
     * Field:
        
         * `type`\: Literal\["expert_experience"\], fixed value
         * `anti_outlier_strategies`\: Optional\[List\[AutoProcessorConfigList\]\], which is optional. If this parameter is not specified, the system automatically obtains the value.
         * `template`\: ModelslimV1ServiceConfig, quantization template configuration
         * `metadata`\: Metadata, which indicates metadata configuration.
2. **ExpertExperienceStrategy (**`msmodelslim/core/tune_strategy/expert_experience/strategy.py`)
    
     * Type: Subclass of BaseTuningStrategy, which is implemented by ITuningStrategy.
     * Function: Implement the expert experience policy. Automatically obtain the search space of the algorithm based on the model structure type, and then reuse the logic of the high touch algorithm in the standing_high policy.
     * Method:
        
         * `__init__(config: StrategyConfig, dataset_loader: DatasetLoaderInfra)`\: Initializes the policy and automatically obtains the search space.
         * `generate_practice(model: IModel, device: DeviceType) -> Generator[PracticeConfig, Optional[EvaluateResult], None]`\: Generate quantitative configurations and reuse the logic of the standing_high policy.
3. **ExpertExperienceTable (**`msmodelslim/core/tune_strategy/expert_experience/expert_experience_table.py`)
    
     * Type: Class
     * Function: expert experience table, which is used to maintain the search space of the recommendation algorithm corresponding to different model structure types.
     * Method:
        
         * `get_search_space(attention_type: str) -> Optional[List[AutoProcessorConfigList]]`\: Obtain the search space of the recommendation algorithm based on the model structure type.
         * `get_default_search_space() -> List[AutoProcessorConfigList]`\: Obtains the default algorithm search space.
4. **ModelAdapter.get_attention_type() (**`msmodelslim/model/base.py`)
    
     * Type: Method (Requires model adapter implementation)
     * Function: Obtains the attention mechanism type of the model.
     * Return value: str, for example, MHA, MLA, DSA, SWA, or GatedDeltaNet.

#### New Directory Structure

```text
msmodelslim/core/tune_strategy/expert_experience/
├── __init__.py
├── strategy.py              #ExpertExperienceStrategyConfig and ExpertExperienceStrategy
└── expert_experience_table.py  #ExpertExperienceTable
```

#### Policy registration

Register the new policy in setup.py:

```python
"msmodelslim.strategy_config.plugins": [
    "standing_high=msmodelslim.core.tune_strategy.standing_high.strategy:StandingHighStrategyConfig",
    "expert_experience=msmodelslim.core.tune_strategy.expert_experience.strategy:ExpertExperienceStrategyConfig",
],
"msmodelslim.strategy.plugins": [
    "standing_high=msmodelslim.core.tune_strategy.standing_high.strategy:StandingHighStrategy",
    "expert_experience=msmodelslim.core.tune_strategy.expert_experience.strategy:ExpertExperienceStrategy",
],
```

### 6.6 Subsystem LLD

#### 6.6.1 New Policy Module Design

ExpertExperienceStrategy, as an independent new strategy module, has the following design points:

1. **Policy configuration: ExpertExperienceStrategyConfig inherits StandingHighStrategyConfig, but the anti_outlier_strategies field is optional.**
2. **Strategy implementation: ExpertExperienceStrategy automatically obtains the search space during initialization, and then creates a StandingHighStrategy instance to reuse its core logic.**
3. **Policy reuse: Reuse the height sensing algorithm of the standing_high policy in combination mode instead of directly modifying the standing_high policy.**

#### 6.6.2 Design for identification of model structure type

Model structure type recognition is implemented by the model adapter. The model adapter needs to implement the get_attention_type() method to return the attention mechanism type of the model. Common model structure types include:

1. **MHA: standard multi-head attention mechanism**
2. **MLA: Multi-Head Latent Attention, a multi-head potential attention mechanism (such as DeepSeek-V3.2)**
3. **DSA: Distributed Sparse Attention**
4. **SWA: Sliding Window Attention**
5. **GatedDeltaNet: A Gated Delta Network Attention Mechanism**

#### 6.6.3 Expert Experience Table Design

The expert experience table uses the dictionary structure. Key indicates the model structure type, and value indicates the search space of the recommendation algorithm. The example structure is as follows:

```python
EXPERT_EXPERIENCE_TABLE = {
    "MHA": [
        [LinearProcessorConfig(...), ...],  #Policy 1
        [LinearProcessorConfig(...), ...],  #Policy 2
        ...
    ],
    "MLA": [
        [LinearProcessorConfig(...), ...],  #Policy 1
        [LinearProcessorConfig(...), ...],  #Policy 2
        ...
    ],
    "DSA": [
        [LinearProcessorConfig(...), ...],  #Policy 1
        ...
    ],
    ...
}
```

The expert experience table is maintained based on historical optimization experience and records the search space of the recommendation algorithm corresponding to different model structures.

#### 6.6.4 Design of the automatic table lookup mechanism

1. **Type identification: Obtain the model structure type through the model adapter.**
2. **Table lookup: Find the matching type in the expert experience table.**
3. **Result use: If the matching type is found, the table query result is used. If not found, use the default search space**
4. **Policy reuse: Transfer the automatically obtained search space to StandingHighStrategy and reuse the high algorithm logic.**

#### 6.6.5 Policy Implementation Mode

Implementation of ExpertExperienceStrategy:

1. **Initialization phase:**
    
     * Check whether anti_outlier_strategies is specified by the user.
     * If not specified, get the model structure type through the model adapter
     * Searches for the matching type in the expert experience table and obtains the search space of the recommendation algorithm.
     * If not found, use the default search space
2. **Policy execution phase:**
    
     * Create StandingHighStrategyConfig and use the automatically obtained anti_outlier_strategies.
     * Create a StandingHighStrategy instance and reuse the logic of the StandingHighStrategy algorithm.
     * Call StandingHighStrategy.generate_practice() to perform tuning.

### 6.7 DFX Attribute Design

#### 6.7.1 Performance Design

1. **Table query performance: The table query performance is O(1), which does not affect the optimization performance.**
2. **Type recognition performance: The model structure type recognition overhead is low and does not affect the optimization performance.**
3. **Impact on existing features: The automatic table lookup function is optional. If the search space is manually specified, the existing optimization process is not affected.**

#### 6.7.2 Upgrade and Capacity Expansion Design

1. **Extensibility of the expert experience table: The expert experience table adopts the dictionary structure, which is easy to expand and maintain. New model structure types can be added at any time.**
2. **Backward compatibility: If the user manually specifies the search space, the user-specified configuration is used preferentially to ensure backward compatibility.**

#### 6.7.3 Exception Handling Design

1. **Type recognition exception: Log a warning log and use the default search space if the model adapter does not recognize the model structure type**
2. **Table lookup exception: If no matching type is found in the expert experience table, log a warning and use the default search space.**
3. **Configuration error: If the table query result is in incorrect format, the error log is recorded and the default search space is used.**

#### 6.7.4 Resource management related design

1. **Memory usage: The expert experience table is loaded to the memory. The memory usage is small, usually several KB.**
2. **Computing resources: The computing overhead of table query operations is small and does not affect system performance.**

#### 6.7.5 Miniaturized Design

This feature does not affect the specifications of the miniaturized version. The expert experience table is lightweight and occupies a small amount of memory.

#### 6.7.6 Testability Design

The test should cover the following areas:

1. **Function test:**
    
     * User-specified search space: Use user-specified configuration
     * Automatic table query succeeded: The search space is obtained automatically based on the model structure type.
     * Automatic table lookup failed: Use the default search space.
     * Type recognition failed: using default search space
2. **Boundary Value Test:**
    
     * The expert experience table is empty.
     * Expert Experience Table contains a large number of types (> 100)
     * Model structure type unknown
3. **Abnormal scenario test:**
    
     * Model adapter does not support type recognition
     * The format of the expert experience table is incorrect.
     * The format of the query result is incorrect.
4. **Performance test:**
    
     * Time required for querying tables
     * Type Identification Time-consuming Test

#### 6.7.7 Security Design

##### 6.7.7.1 Safety Design Qualification

The security design of this use case is similar to that of use case 1. The use case focuses on configuration security and does not involve sensitive data processing.

##### 6.7.7.2 Sensitive Data Analysis

This use case does not involve sensitive data processing. It mainly involves querying and configuring the model structure type and algorithm search space, and does not involve sensitive operations such as user authentication and key management.

##### 6.7.7.3 Design Implementation

The security design of this use case is as follows:

1. **Configuration verification: Verify the configuration in the table query result to ensure that the configuration format is correct.**
2. **Error handling: The error information does not disclose sensitive information and only necessary debugging information is recorded.**

### 6.8 External Interfaces of the System

This use case affects the following external interfaces:

1. **New policy configuration interface: The anti_outlier_strategies field in ExpertExperienceStrategyConfig is optional. You do not need to specify this field. The system automatically obtains the field based on the model structure type.**
2. **Model adapter interface: The model adapter needs to implement the get_attention_type() method to provide model structure type information.**
3. **Policy selection interface: Users can select the "expert_experience" policy instead of the "standing_high" policy in the tuning plan configuration file.**
4. **Policy registration interface: New policies need to register entry points in setup.py, including policy configuration and policy implementation.**

### 6.9 Self-Test Case Design

The self-test case design is as follows:

1. **Policy selection test:**
    
     * Input: Select the "expert_experience" policy in the tuning plan configuration file.
     * Expected: The system creates an ExpertExperienceStrategy instance instead of a StandingHighStrategy instance.
2. **User-specified search space test:**
    
     * Input: The user specifies anti_outlier_strategies in the configuration file.
     * Expected result: The user-specified configuration is used, and automatic table query is not performed.
3. **Automatic table query test:**
    
     * Input: The model structure type is MHA, which exists in the expert experience table.
     * Expected result: Automatically obtain the corresponding search space, create a StandingHighStrategy instance, and reuse its logic.
4. **Automatic table query failure test:**
    
     * Input: The model structure type is Unknown, which does not exist in the expert experience table.
     * Expected result: Use the default search space, record warning logs, create a StandingHighStrategy instance, and reuse its logic.
5. **Type Identification Failure Test:**
    
     * Input: Model adapter does not support type recognition
     * Expected result: Use the default search space, record warning logs, create a StandingHighStrategy instance, and reuse its logic.
6. **Expert experience table abnormality test:**
    
     * Input: The format of the expert experience table is incorrect.
     * Expected result: Use the default search space, record error logs, create a StandingHighStrategy instance, and reuse its logic.
7. **Policy reuse test:**
    
     * Input: Use the expert_experience policy for optimization.
     * Expected result: The algorithm logic of the standing_high policy can be correctly reused, and the optimization result is the same as that of the standing_high policy.

## 7. Reliability and availability design

### 7.1 Redundancy Design

The automatic optimization acceleration feature uses the following redundancy design:

1. **Precise cache redundancy: The precision cache is stored in YAML persistently. Even if the optimization process is interrupted, the historical precision cache is retained. Resumable scheduling is supported.**
2. **Configuration backup: Quantified configurations of each iteration are saved in historical records, and configuration backup and restoration are supported.**
3. **Log recording: Detailed logs are recorded to support fault locating and rectification.**

### 7.2 Fault Management

#### Fault detection

1. **Precision evaluation failure detection: If the precision evaluation fails, the system records an error log and proceeds to the next iteration.**
2. **Quantization failure detection: If quantization fails, log the error and proceed to the next iteration.**
3. **Service-based startup failure detection: If the service-based startup fails, the system records an error log and proceeds to the next iteration.**

#### Fault isolation

1. **Iteration isolation: Each iteration is executed independently. If an iteration fails, other iterations are not affected.**
2. **Module isolation: The quantification, evaluation, and pre-check modules are executed independently. The failure of a single module does not affect other modules.**

#### Fault recovery

1. **Automatic recovery: Restores the evaluated configuration result from the historical precision cache through the resumable resuming mechanism.**
2. **Manual recovery: Users can restart the optimization with the same save_path to automatically recover the historical precision cache.**

### 7.3 Overload control design

1. **Limitation on the number of iterations: The maximum number of iterations can be set to prevent infinite iterations.**
2. **Timeout control: The maximum iteration time can be set to prevent the optimization process from running indefinitely.**
3. **Resource monitoring: Monitors the usage of memory and storage resources. If resources are insufficient, stop optimization.**

### 7.4 Upgrade Without Service Interruption

1. **Configuration compatibility: The automatic optimization function of the new version is compatible with the configuration file format of the earlier version.**
2. **Data compatibility: The format design of the historical precision cache considers version compatibility and supports cross-version use.**
3. **Interface compatibility: Backward compatibility is considered in the design of the automatically optimized interface. The calling mode of the earlier version is still valid.**

### 7.5 Human Error Design

1. **Configuration verification: Verify the configuration file of the optimization plan. If the configuration is incorrect, a clear error message is displayed.**
2. **Parameter check: Check the parameters in the command line. If the parameters are incorrect, a clear error message is displayed.**
3. **Log prompt: Detailed logs are recorded, helping users understand the optimization process and results.**

### 7.6 Fault Prediction and Prevention Design

1. **Resource monitoring: Monitors the usage of memory and storage resources and warns of resource shortage in advance.**
2. **Performance monitoring: Monitors performance indicators during optimization and warns performance problems in advance.**
3. **Exception detection: detects exceptions (such as precision evaluation failure and quantification failure) and warns potential problems in advance.**

## 8. Design for features and non-functional quality attributes

### 8.1 Testability

*This document describes the test direction and specifications of the feature, and describes the aspects that should be tested, boundary values, abnormal values, and abnormal scenarios that should be noted by the test personnel.*

### 8.2 Serviceability

*Provides various maintainable and serviceable measures for features, and provides complete documentation for feature usage, maintenance, and troubleshooting.*

### 8.3 Evolvability

*Focus on the evolvability of the feature architecture and functions.*

### 8.4 Openness

*Focus on the openness of external interfaces, including the standardization of interfaces, for example, compliance with the SQL 2011 standard.*

### 8.5 Compatibility

*Focus on whether the feature affects the forward compatibility of the system, that is, whether the old functions are available after the upgrade and whether the usage behavior is consistent with that of the old version.*

### 8.6 Scalability/Scalability

*This feature effectively meets the requirements for system capacity changes, including scaling of database nodes and database servers.*

### 8.7 Maintainability

*Focus on feature maintainability, such as diagnosis view and log printing.*

### 8.8 Documentation

*Refer to the following table to evaluate the modification points of various documents involved in the feature and describe the specific modification points.*

| Category                                                                                                                                                                  | Manual Name           | Involved or Not (Y/N)                                       | Description of the modified or added content |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------- | -------------------------------------------- |
| White Paper                                                                                                                                                               | Technical white paper | N                                                           | Added the XX technology in section XX.       |
| Product Documentation                                                                                                                                                     | Product Description   | Y                                                           | Updated the technical specifications to XX.  |
|-| Feature Description                                                                                                                                                       | Y                     | Added the XX feature.                                       |
|-| Compilation Guide                                                                                                                                                         | Y                     | XXX                                                         |
|-| Installation guide                                                                                                                                                        | Y                     | Updated the scenario in the "Installing a Cluster" section. |
|-| Administrator's Guide                                                                                                                                                     | N                     | XXX                                                         |
|-| Developer guide (including the development tutorial, SQL reference, system tables and system views, GUC parameter description, error code description, and API reference) | Y                     | Added the XXX function in section XX.                       |
|-| Tool Reference                                                                                                                                                            | Y                     | Added the XX tool.                                          |
|-| Glossary of terms                                                                                                                                                         | Y                     | New term XX                                                 |
| Getting Started                                                                                                                                                           | Easy tutorial         | N                                                           | XXX                                          |

## 9. (Optional) Data Structure Design

The automatic optimization acceleration feature stores data in YAML format, including:

1. **Accuracy cache (accuracy.yaml):**
    
     * Structure: dictionary format, key is the MD5 hash value, and value is the evaluation result.
     * Function: Stores the evaluated quantization configuration and precision results, and supports breakpoint continuation.
2. **History index (history.yaml):**
    
     * Structure: List format, with each element containing the configuration ID, evaluation result, MD5 hash value, and timestamp
     * Function: Records the configuration and evaluation results of each iteration. Historical query is supported.
3. **Quantified configuration (practice configs):**
    
     * Structure: PracticeConfig object in YAML format
     * Function: Stores quantitative configurations of each iteration and supports configuration backup and restoration.

## 10. List of references

1. **msModelSlim tool documentation:**
    
     * User Guide for the Automatic Optimization Function
     * Description of the file format of the optimization plan configuration file
     * API Reference Document
2. **Related code implementation:**
    
     * `msmodelslim/app/auto_tuning/application.py`\: Automatically tune the application layer implementation.
     * `msmodelslim/core/tune_strategy/standing_high/strategy.py`\: implementation of the standing_high policy
     * `msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`\: implement garbled character detection and pre-check.
     * `msmodelslim/infra/yaml_practice_history_manager.py`\: Implemented by the history management module
3. **Design principles:**
    
     * Interface Standardization Principles
     * Data format unification principle
     * Error Handling Specifications
