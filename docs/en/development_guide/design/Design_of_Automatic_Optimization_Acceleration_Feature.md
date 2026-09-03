# **msModelSlim Auto Tuning Acceleration Feature Design Specification**

<table>
    <tr>
        <td>SIG Group:</td>
        <td>msit</td>
    </tr>
    <tr>
        <td>Target Version:</td>
        <td>26.0.0</td>
    </tr>
    <tr>
        <td>Designers:</td>
        <td>joejoezhou</td>
    </tr>
    <tr>
        <td>Date:</td>
        <td>20260122</td>
    </tr>
</table>

**Copyright © 2026 msModelSlim Community**

Your reproduction, use, modification, and distribution of "this document" are subject to the Creative Commons Attribution-ShareAlike 4.0 International Public License (hereinafter referred to as "CC BY-SA 4.0").
For your convenience, you can access <https://creativecommons.org/licenses/by-sa/4.0/> to understand a summary of CC BY-SA 4.0 (but not as a substitute).
You can access the full text of CC BY-SA 4.0 at the following URL: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

**Revision History**

<table>
    <tr>
        <th>Date</th>
        <th>Revision</th>
        <th>Revision Description</th>
        <th>Author</th>
        <th>Review</th>
    </tr>
    <tr>
        <td>20260122</td>
        <td>1.0.0</td>
        <td>Document created</td>
        <td>joejoezhou</td>
        <td>panyj1993</td>
    </tr>
</table>

**Table of Contents**

1. Feature Overview

   1.1 Scope

   1.2 Feature Requirement List

2. Requirement Scenario Analysis

   2.1 Feature Requirement Source and Value Overview

   2.2 Feature Scenario Analysis

   2.3 Feature Impact Analysis

   2.3.1 Hardware Constraints

   2.3.2 Technical Constraints

   2.3.3 License Impact Analysis

   2.3.4 System Performance Specification Impact Analysis

   2.3.5 System Reliability Specification Impact Analysis

   2.3.6 System Compatibility Impact Analysis

   2.3.7 Interaction and Conflict Analysis with Other Major Features

   2.4 Similar Community and Commercial Software Implementation Analysis

3. Feature/Function Implementation Principles (Multiple Use Cases Can Be Decomposed)

   3.1 Objectives

   3.2 Overall Solution

4. Use Case 1 Implementation

   4.1 Use Case Description

   4.2 Feature Design Approach

   4.3 Constraints

   4.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

   4.5 Subsystem Interfaces (Module Interface Definitions)

   4.6 Subsystem Detailed Design

   4.6.1 Garbled Text Detection Check Item Design

   4.6.2 Pre-check Flow Design

   4.7 DFX Attribute Design

   4.7.1 Performance Design

   4.7.2 Upgrade and Expansion Design

   4.7.3 Exception Handling Design

   4.7.4 Resource Management Design

   4.7.5 Miniaturization Design

   4.7.6 Testability Design

   4.7.7 Security Design

   4.8 System External Interfaces

   4.9 Self-Test Case Design

5. Use Case 2 Implementation

   5.1 Use Case Description

   5.2 Feature Design Approach

   5.3 Constraints

   5.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

   5.5 Subsystem Interfaces (Module Interface Definitions)

   5.6 Subsystem Detailed Design

   5.6.1 Accuracy Cache Design

   5.6.2 History Index Design

   5.6.3 Cache Reuse Mechanism Design

   5.6.4 Checkpoint Resume Flow Design

   5.7 DFX Attribute Design

   5.7.1 Performance Design

   5.7.2 Upgrade and Expansion Design

   5.7.3 Exception Handling Design

   5.7.4 Resource Management Design

   5.7.5 Miniaturization Design

   5.7.6 Testability Design

   5.7.7 Security Design

   5.8 System External Interfaces

   5.9 Self-Test Case Design

6. Use Case 3 Implementation

   6.1 Use Case Description

   6.2 Feature Design Approach

   6.3 Constraints

   6.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

   6.5 Subsystem Interfaces (Module Interface Definitions)

   6.6 Subsystem Detailed Design

   6.6.1 New Strategy Module Design

   6.6.2 Model Structure Type Identification Design

   6.6.3 Expert Experience Table Design

   6.6.4 Automatic Lookup Mechanism Design

   6.6.5 Strategy Implementation Approach

   6.7 DFX Attribute Design

   6.7.1 Performance Design

   6.7.2 Upgrade and Expansion Design

   6.7.3 Exception Handling Design

   6.7.4 Resource Management Design

   6.7.5 Miniaturization Design

   6.7.6 Testability Design

   6.7.7 Security Design

   6.8 System External Interfaces

   6.9 Self-Test Case Design

7. Reliability and Availability Design

   7.1 Redundancy Design

   7.2 Fault Management

   7.3 Overload Control Design

   7.4 Service Continuity During Upgrade

   7.5 Human Error Design

   7.6 Fault Prediction and Prevention Design

8. Feature Non-functional Quality Attribute Design

   8.1 Testability

   8.2 Serviceability

   8.3 Evolvability

   8.4 Openness

   8.5 Compatibility

   8.6 Scalability and Extensibility

   8.7 Maintainability

   8.8 Documentation

9. Data Structure Design (Optional)

10. Reference List

**List of Tables**

Table 1: Feature Requirement List

Table 2: Security Design Confirmation Table

Table 3: Documentation Modification List

**List of Figures**

Figure 1: Overall Implementation Principle Diagram

**List of Abbreviations**:

<table>
    <tr>
        <th>Abbreviation</th>
        <th>Full Name</th>
        <th>Chinese Description</th>
    </tr>
    <tr>
        <td>MHA</td>
        <td>Multi-Head Attention</td>
        <td>Multi-Head Attention</td>
    </tr>
    <tr>
        <td>MLA</td>
        <td>Multi-Head Latent Attention</td>
        <td>Multi-Head Latent Attention</td>
    </tr>
    <tr>
        <td>DSA</td>
        <td>Distributed Sparse Attention</td>
        <td>Distributed Sparse Attention</td>
    </tr>
    <tr>
        <td>SWA</td>
        <td>Sliding Window Attention</td>
        <td>Sliding Window Attention</td>
    </tr>
    <tr>
        <td>NPU</td>
        <td>Neural Processing Unit</td>
        <td>Neural Processing Unit</td>
    </tr>
    <tr>
        <td>YAML</td>
        <td>YAML Ain't Markup Language</td>
        <td>YAML Markup Language</td>
    </tr>
    <tr>
        <td>MD5</td>
        <td>Message Digest Algorithm 5</td>
        <td>Message Digest Algorithm 5</td>
    </tr>
</table>

## 1. Feature Overview

Accuracy feedback auto tuning is a core feature of the msModelSlim tool. It reduces the manual workload of model quantization accuracy tuning through an automated process. This feature leverages the accuracy tuning experience accumulated from mature quantization patterns. It automatically iterates to generate quantization configurations, evaluate model accuracy, and adjust strategies based on accuracy feedback until a quantization solution that meets accuracy requirements is found.

The **Auto Tuning Acceleration Feature** is an acceleration optimization built on the existing accuracy feedback auto tuning function. It aims to further improve the efficiency and reliability of auto tuning. This feature accelerates the auto tuning process through three key optimization points: 1) skipping dataset evaluation when conversation output is garbled, avoiding wasted computational resources on invalid evaluations; 2) supporting checkpoint resume to avoid repeated evaluations caused by unexpected interruptions; 3) tuning strategies based on expert experience to simplify user configuration and improve tuning efficiency.

The value of this feature to customers is primarily reflected in: 1) saving computational resources by intelligently skipping invalid evaluations to reduce unnecessary computational overhead; 2) improving tuning reliability by supporting checkpoint resume to avoid work loss; 3) simplifying user operations by automatically selecting optimal strategies based on expert experience, reducing configuration complexity.

This document describes the design and implementation of the auto tuning acceleration feature, including three main Use Cases: skipping dataset evaluation when conversation output is garbled, supporting checkpoint resume for auto tuning, and tuning strategies based on expert experience. This document is intended for development, testing, and maintenance personnel of the auto tuning feature in the msModelSlim tool.

### 1.1 Scope

This feature is an acceleration optimization built on the existing accuracy feedback auto tuning function. It includes the following function points:

1. **Garbled Text Detection and Skipping Mechanism**: Perform a pre-check before the formal evaluation to detect whether the model output is garbled. If garbled text is detected, skip the dataset evaluation to save computational resources. This optimizes the existing auto tuning function by avoiding wasted computational resources on invalid evaluations.

2. **Checkpoint Resume Function**: Support recovering evaluated quantization configuration results from the historical accuracy cache to avoid repeated evaluations and enable checkpoint resume of the tuning process. This optimizes the existing auto tuning function to improve the reliability of the tuning process.

3. **Expert Experience-Based Tuning Strategy**: Create an independent expert_experience strategy module that supports automatic lookup of algorithm search spaces based on model structure types (such as MHA, MLA, DSA, SWA, and GatedDeltaNet), without requiring manual user input. This optimizes the existing auto tuning function to simplify user configuration and improve tuning efficiency.

**Note**: The accuracy feedback auto tuning process itself is an existing function and is not within the scope of this feature. This feature only includes the three acceleration optimization points described above.

### 1.2 Feature Requirement List

Table 1: Feature Requirement List

<table>
    <tr>
        <th>Requirement ID</th>
        <th>Requirement Name</th>
        <th>Feature Description</th>
        <th>Remarks</th>
    </tr>
    <tr>
        <td>1</td>
        <td>Skip dataset evaluation when conversation output is garbled</td>
        <td>Before model accuracy evaluation, detect whether the model output is garbled through a pre-check mechanism. If garbled text is detected, skip the dataset evaluation and directly return an evaluation result with an accuracy of 0 to save computational resources</td>
        <td>-</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Auto tuning supports checkpoint resume</td>
        <td>Support recovering evaluated quantization configuration results from the historical accuracy cache. When the tuning process is unexpectedly interrupted, the system automatically detects and reuses the historical accuracy cache upon restart, avoiding repeated evaluations of the same quantization configurations</td>
        <td>-</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Expert experience-based tuning strategy</td>
        <td>Create an independent expert_experience strategy module that supports automatic lookup of algorithm search spaces based on model structure types (such as MHA, MLA, DSA, SWA, and GatedDeltaNet), without requiring manual user input for search space configuration</td>
        <td>-</td>
    </tr>
</table>

## 2. Requirement Scenario Analysis

### 2.1 Feature Requirement Source and Value Overview

The msModelSlim tool has implemented the accuracy feedback auto tuning function, which reduces the manual workload of model quantization accuracy tuning through an automated process. However, during actual use, the following problems were identified in the auto tuning function:

1. **Wasted Resources on Invalid Evaluations**: When the quantized model output is garbled, the system still performs the full dataset evaluation, wasting significant computational resources.
2. **Inability to Recover After Interruption**: If the tuning process is unexpectedly interrupted (such as system failure or manual stop), the system must start from scratch upon restart, unable to reuse previously evaluated configuration results.
3. **High Configuration Complexity**: The standing_high strategy requires users to manually input the algorithm search space, which has high configuration complexity for users unfamiliar with quantization tuning.

The **Auto Tuning Acceleration Feature** aims to solve the above problems through three key optimization points to accelerate the auto tuning process. The specific value brought to users by this feature includes:

1. **Saving Computational Resources**: Save computational resources and reduce tuning costs by intelligently skipping invalid evaluations (such as garbled text detection). It is expected to save 10 to 30% of evaluation time.
2. **Improving Tuning Reliability**: Support checkpoint resume to avoid work loss caused by unexpected interruptions, improving the reliability of the tuning process. Even if the tuning process is interrupted, the system can recover from the historical accuracy cache, avoiding repeated evaluations.
3. **Simplifying User Operations**: Automatically select optimal strategies based on expert experience, without requiring users to manually input search space configuration, reducing configuration complexity and improving tuning efficiency.

Without this feature, the auto tuning function can still work normally, but the above problems would affect tuning efficiency and user experience.

### 2.2 Feature Scenario Analysis

#### Scenario Trigger Conditions and Objects

1. **Trigger Conditions**:
   - The user needs to quantize a large language model or multimodal model.
   - The user wants to find a quantization solution that meets accuracy requirements through auto tuning.
   - The user has configured an auto tuning plan (YAML configuration file).

2. **Target Users**:
   - Model quantization engineers: Have certain knowledge of model quantization and are familiar with the msModelSlim tool.
   - AI application developers: Need to deploy models on NPU devices and have accuracy requirements for quantization.

3. **User Interfaces**:
   - Command-line interface: `msmodelslim tune` command
   - Configuration file: YAML format tuning plan configuration file

#### Main Application Scenarios

1. **New Model Quantization Scenario**:
   - The user quantizes a model for the first time and needs to find a quantization solution that meets accuracy requirements.
   - Sub-scenario: The model structure is known, but the quantization parameters are unknown.
   - Key operations: Configure the tuning plan, start auto tuning, wait for tuning to complete, and obtain the final quantization configuration.

2. **Model Accuracy Optimization Scenario**:
   - The user has an existing quantization solution, but the accuracy does not meet the requirements. The user needs to optimize the accuracy through auto tuning.
   - Sub-scenario: Fine-tune based on the existing quantization solution.
   - Key operations: Start tuning based on the existing solution, iteratively optimize, and verify accuracy improvement.

3. **Batch Model Quantization Scenario**:
   - The user needs to quantize multiple models and wants automated processing.
   - Sub-scenario: The model series are the same, but the parameters are different.
   - Key operations: Batch configure tuning plans, execute tuning in parallel or serial, and summarize results.

### 2.3 Feature Impact Analysis

The auto tuning acceleration feature is integrated into the core tuning process of the existing accuracy feedback auto tuning function and interacts with the following modules:

1. **Quantization Service Module**: Calls the quantization service to quantize the model.
2. **Evaluation Service Module**: Calls the evaluation service to evaluate the accuracy of the quantized model.
3. **Tuning Strategy Module**: Uses different tuning strategies to generate quantization configurations.
4. **History Management Module**: Manages tuning history records and the accuracy cache.
5. **Model Adapter Module**: Adapts interfaces for different model series.

#### Interaction Analysis with Other Requirements and Features

1. **Interaction with Quantization Feature**: Auto tuning depends on the quantization function and requires the quantization service to support multiple quantization configurations.
2. **Interaction with Evaluation Feature**: Auto tuning depends on the evaluation function and requires the evaluation service to support accuracy evaluation and pre-checks.
3. **Interaction with Best Practice Library**: After successful tuning, the final quantization configuration can be saved to the best practice library.
4. **Interaction with Model Adapter**:
   - When the three tuning strategies require **automatic sensitive layer analysis**, the model adapter must implement **`ModelSlimPipelineInterfaceV1`** (that is, `PipelineInterface`, same as CLI `msmodelslim analyze`).
   - `standing_high`: Always performs automatic sensitive layer analysis.
   - `binary_fallback`: Skips sensitive layer analysis when non-empty `rollback_candidates` is configured.
   - `standing_high_with_experience`: Additionally requires **`StandingHighWithExperienceInterface`** (`load_model`, outlier suppression detection).
   - Sensitive layer analysis is called by `PipelineAnalysisService` / Runner through `init_model` and pipeline methods. The strategy side does not pre-load the model.

#### Platform Difference Analysis

1. **Hardware Platform**: Primarily supports NPU devices (such as the Ascend series), requiring NPU devices to support model quantization and inference.
2. **Operating System**: Supports Linux operating system, requiring Python 3.8+.

#### Compatibility Analysis

1. **Forward Compatibility**: The new version of the auto tuning function is compatible with the quantization configuration format of the old version.
2. **Configuration Compatibility**: Supports the tuning plan configuration file format of the old version, but the new format is recommended.

#### Constraints and Limitations

1. **Model Support Limitation**: Only supports model series with implemented model adapters.
2. **Accuracy Evaluation Limitation**: Requires vLLM-Ascend to support the serviced startup of the quantized model.
3. **Resource Limitation**: The tuning process requires sufficient storage space to save quantized models and evaluation results.

#### 2.3.1 Hardware Constraints

1. **NPU Device Requirements**: Requires an NPU device that supports model quantization and inference. At least one NPU card is required.
2. **Memory Requirements**: The tuning process requires sufficient memory to load the model and perform quantization computations. A minimum of 32GB of memory is recommended.
3. **Storage Requirements**: Requires sufficient storage space to save quantized models, evaluation results, and history records. A minimum of 100GB of available space is recommended.
4. **Network Requirements**: If a remote evaluation service is used, a stable network connection is required.

**Mitigation**:

- For insufficient memory, reduce the batch size or use model parallelism to reduce memory usage.
- For insufficient storage, periodically clean up history records or use external storage.

#### 2.3.2 Technical Constraints

**Operating System**: Linux (Ubuntu 20.04+ or CentOS 7+ recommended)

**Programming Language**: Python 3.8+

**Dependency Frameworks**:

- PyTorch: Used for model loading and quantization.
- vLLM-Ascend: Used for model serviced startup and inference.
- AISbench: Used for accuracy evaluation.

**Mitigation**:

- For unsupported Python versions, use conda or virtualenv to create a virtual environment.
- For incompatible dependency framework versions, refer to the installation guide to use the specified versions.

#### 2.3.3 License Impact Analysis

This feature primarily uses the following open-source software and technologies:

1. **PyTorch**: BSD license, allows commercial use.
2. **vLLM-Ascend**: Apache 2.0 license, allows commercial use.
3. **AISbench**: Apache 2.0 license, allows commercial use.
4. **Pydantic**: MIT license, allows commercial use.

All introduced third-party open-source software complies with the License requirements of the msModelSlim project and does not affect the License compliance of the project.

#### 2.3.4 System Performance Specification Impact Analysis

Based on the feature runtime resource conditions:

1. **Memory Requirements**: At least 32GB of memory is required, and 64GB or more is recommended. Primarily used for:
   - Model loading: Depending on the model size, 10 to 50GB of memory may be required.
   - Quantization computation: An additional 10 to 20GB of memory is required for the quantization process.
   - Evaluation service: 5 to 10GB of memory is required for the evaluation service to run.

2. **Storage Requirements**: At least 100GB of available storage space is required, and 200GB or more is recommended. Primarily used for:
   - Quantized model storage: Each quantization configuration model may require 10 to 50GB.
   - Evaluation result storage: History records and accuracy cache may require 10 to 50GB.
   - Temporary files: Temporary files during the tuning process may require 20 to 50GB.

3. **NPU Requirements**: At least 1 NPU card is required, and 2 or more are recommended. Primarily used for:
   - Model quantization: The quantization process requires NPU support.
   - Model inference: The evaluation process requires NPU for inference.

#### 2.3.5 System Reliability Specification Impact Analysis

Assumptions and constraints for reliability metrics:

1. **Tuning Success Rate**: Under normal conditions, the tuning success rate for supported model series should reach 80% or above.
2. **Checkpoint Resume Reliability**: The recovery success rate of the historical accuracy cache should reach 99% or above.
3. **Exception Handling**: For common exceptions (such as network interruption and insufficient storage), the system should degrade gracefully or provide clear error messages.

#### 2.3.6 System Compatibility Impact Analysis

This feature does not affect the forward compatibility of the system:

1. **Configuration Compatibility**: The new version of the auto tuning function is compatible with the quantization configuration format and tuning plan format of the old version.
2. **Interface Compatibility**: The interface design of auto tuning considers backward compatibility, and the calling method of the old version remains valid.
3. **Data Compatibility**: The format design of the historical accuracy cache considers version compatibility and supports cross-version use.

#### 2.3.7 Interaction and Conflict Analysis with Other Major Features

1. **Interaction with Quantization Feature**:
   - Auto tuning depends on the quantization function and requires the quantization service to support multiple quantization configurations.
   - Auto tuning does not affect the manual quantization function. Both can coexist.

2. **Interaction with Evaluation Feature**:
   - Auto tuning depends on the evaluation function and requires the evaluation service to support accuracy evaluation and pre-checks.
   - Auto tuning does not affect the manual evaluation function. Both can coexist.

3. **Interaction with Best Practice Library**:
   - After successful tuning, the final quantization configuration can be saved to the best practice library.
   - Configurations in the best practice library can be referenced by auto tuning strategies.

4. **Interaction with Model Adapter**:
   - When tuning strategies require **automatic sensitive layer analysis**, the model adapter must implement **`ModelSlimPipelineInterfaceV1`** (that is, `PipelineInterface`).
   - `standing_high`: Always performs automatic sensitive layer analysis. `binary_fallback`: Skips when non-empty `rollback_candidates` is configured.
   - `standing_high_with_experience`: Additionally requires **`StandingHighWithExperienceInterface`** (`load_model`).
   - For models that do not support auto tuning, the manual quantization function can still be used.

### 2.4 Similar Community and Commercial Software Implementation Analysis

Currently, the main implementation approaches in the field of model quantization auto tuning include:

1. **Neural Network Intelligence (NNI)**: An open-source auto machine learning tool from Microsoft that supports model compression and quantization auto tuning. Its advantage lies in supporting multiple tuning algorithms and distributed tuning, but it primarily targets PyTorch and TensorFlow frameworks, with limited support for NPU devices.

2. **msModelSlim Auto Tuning**: The implementation approach of this feature. The main advantages include:
   - **High Integration**: Deeply integrated with the msModelSlim tool, supporting the complete quantization-evaluation-tuning process.
   - **NPU Optimization**: Deeply optimized for Ascend NPU devices, supporting efficient quantization inference.
   - **Intelligent Skipping**: Intelligently skips invalid evaluations through mechanisms such as garbled text detection, saving computational resources.
   - **Checkpoint Resume**: Supports historical accuracy cache recovery, improving the reliability of the tuning process.
   - **Expert Experience**: Automatically selects optimal strategies based on historical experience, improving the tuning success rate.

Compared with similar approaches, the main advantages of this feature lie in the deep optimization for NPU devices and intelligent tuning strategies, enabling more efficient discovery of quantization solutions that meet accuracy requirements.

## 3. Feature/Function Implementation Principles (Multiple Use Cases Can Be Decomposed)

### 3.1 Objectives

The objective of the auto tuning acceleration feature is to accelerate the auto tuning process through three key optimization points built on the existing accuracy feedback auto tuning function. The specific objectives include:

1. **Resource Optimization**: Save computational resources and reduce tuning costs by intelligently skipping invalid evaluations (such as garbled text detection). The objective is to save over 90% of evaluation time when garbled text is detected.
2. **Reliability Assurance**: Support checkpoint resume to avoid work loss caused by unexpected interruptions, improving the reliability of the tuning process. The objective is to achieve a historical accuracy cache recovery success rate of 99% or above.
3. **Efficiency Improvement**: Automatically select optimal strategies based on expert experience, simplifying user configuration and improving tuning efficiency. The objective is to reduce user configuration time and improve the tuning success rate.
4. **Compatibility Assurance**: All optimization points maintain compatibility with the existing auto tuning function and do not affect the use of existing functions.

### 3.2 Overall Solution

The auto tuning acceleration feature is an optimization built on the existing accuracy feedback auto tuning function, adopting the following design approach:

1. **Pre-check Optimization**: Add a pre-check mechanism to the existing evaluation process to detect whether the model output is garbled before the formal evaluation. If garbled text is detected, skip the dataset evaluation.
2. **History Cache Optimization**: Add a historical accuracy cache mechanism to the existing tuning process to support recovering evaluated configuration results from the historical cache, enabling checkpoint resume.
3. **Strategy Optimization**: Create a new expert_experience strategy module that automatically obtains the algorithm search space based on expert experience, simplifying user configuration.

All optimization points are integrated into the existing auto tuning process and do not affect the normal use of existing functions.

#### Hardware Selection

- **NPU Device**: Use Ascend NPU devices for model quantization and inference, fully leveraging the quantization acceleration capabilities of the NPU.
- **Storage Device**: Use local storage or network storage to save quantized models and evaluation results.

#### Algorithm Selection

- **Tuning Strategy**: Adopt the standing_high strategy as the base tuning strategy, while providing an independent expert_experience strategy module that supports automatic lookup based on model structure types and expert experience.
- **Accuracy Evaluation**: Use AISbench for accuracy evaluation, supporting multiple evaluation datasets.
- **Pre-check Mechanism**: Use pre-check mechanisms such as garbled text detection and expected answer checking to intelligently skip invalid evaluations.

#### Architecture Layout

The existing accuracy feedback auto tuning function adopts a layered architecture design:

1. **Application Layer**: AutoTuningApplication, responsible for coordinating the entire tuning process.
2. **Strategy Layer**: ITuningStrategy, responsible for generating quantization configurations and adjusting strategies.
3. **Service Layer**: Quantization service and evaluation service, responsible for specific quantization and evaluation operations.
4. **Data Layer**: History management module, responsible for managing tuning history records and the accuracy cache.

The **Auto Tuning Acceleration Feature** is integrated into the existing architecture through the following approaches:

1. **Pre-check Optimization**: Add a pre-check mechanism to the evaluation service layer to perform garbled text detection before the formal evaluation.
2. **History Cache Optimization**: Add an accuracy cache mechanism to the data layer to support checkpoint resume.
3. **Strategy Optimization**: Add an expert_experience strategy module to the strategy layer to automatically obtain the search space based on expert experience.

#### Use Case Decomposition

Based on the scenario analysis and system decomposition, three key Use Cases are identified. Each Use Case has a specific impact on the auto tuning function and requires the implementation of corresponding features:

1. **Use Case 1: The user wants to skip invalid evaluations when the model output is garbled during auto tuning**
   - **User Scenario**: During auto tuning, the user finds that the quantized model output is garbled and wants the system to intelligently identify and skip invalid dataset evaluations to save computational resources.
   - **Impact on the Auto Tuning Function**: A pre-check is required before the evaluation to detect whether the model output is garbled. If garbled text is detected, the dataset evaluation is skipped.
   - **Implemented Feature**: Skip dataset evaluation when conversation output is garbled.

2. **Use Case 2: The user wants to continue tuning after an unexpected interruption during auto tuning**
   - **User Scenario**: During auto tuning, if the tuning is unexpectedly interrupted (such as system failure or manual stop), the user wants to reuse the historical records upon restarting auto tuning and continue the accuracy tuning process, avoiding repeated evaluations.
   - **Impact on the Auto Tuning Function**: Checkpoint resume must be supported to recover evaluated quantization configuration results from the historical accuracy cache.
   - **Implemented Feature**: Auto tuning supports checkpoint resume.

3. **Use Case 3: The user wants to automatically obtain the search space based on the model structure type when configuring auto tuning**
   - **User Scenario**: When configuring auto tuning, the user is unfamiliar with the search space configuration for quantization tuning and wants the system to automatically look up the algorithm search space based on the model structure type (such as MHA, MLA, DSA, SWA, or GatedDeltaNet), simplifying the configuration.
   - **Impact on the Auto Tuning Function**: An expert experience-based tuning strategy must be provided to support automatically obtaining the search space based on the model structure type.
   - **Implemented Feature**: Expert experience-based tuning strategy.

#### Integration Principles

1. **Interface Standardization**: All module interfaces adopt standardized interface definitions for easy extension and maintenance.
2. **Unified Data Format**: Use the unified YAML format to save configurations and results for easy parsing and storage.
3. **Standardized Error Handling**: Unified error handling and log recording mechanisms for easy problem identification and debugging.

#### Overall Architecture Diagram

```tex
┌─────────────────────────────────────────────────────────────┐
│                    User Command-Line Interface                │
│                  msmodelslim tune                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AutoTuningApplication (Application Layer)        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Load tuning plan                                  │   │
│  │  2. Initialize tuning strategy                        │   │
│  │  3. Detect historical accuracy cache                  │   │
│  │  4. Iterative tuning loop                             │   │
│  │     - Generate quantization configuration             │   │
│  │     - Attempt history recovery                        │   │
│  │     - Quantize model                                  │   │
│  │     - Evaluate model accuracy (with pre-check)        │   │
│  │     - Save tuning history                             │   │
│  │     - Determine whether to continue                   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Tuning       │ │ Quantization │ │ Evaluation   │
│ Strategy     │ │ Service      │ │ Service      │
│ Layer        │ │ Layer        │ │ Layer        │
│ITuningStrategy│ │IQuantService │ │EvaluateService│
│              │ │              │ │              │
│- standing_high│ │- Model       │ │- Accuracy    │
│- Expert       │ │  quantization│ │  evaluation  │
│  experience   │ │- Config      │ │- Pre-check   │
│  strategy     │ │  generation  │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              History Management Module (Data Layer)          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - Accuracy cache management (accuracy.yaml)          │   │
│  │  - History record management (history.yaml)           │   │
│  │  - Configuration file management (practice configs)   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Figure 1: Overall Implementation Principle Diagram

## 4. Use Case 1 Implementation

### 4.1 Use Case Description

**Use Case Name**: Skip Dataset Evaluation on Conversational Garbled Text

**Use Case Scenario**:

- During the auto tuning process, the quantized model outputs garbled text
- The user expects the system to intelligently identify garbled text and skip invalid dataset evaluations to save computing resources
- The system performs a pre-check before the formal evaluation to detect whether the model output is garbled text
- If garbled text is detected, the system skips the dataset evaluation and directly returns an evaluation result with an accuracy of 0

**Impact on the Auto Tuning Feature**:

- Requires adding a pre-check mechanism to the evaluation process
- Requires implementing the garbled text detection feature
- Requires supporting the logic to skip invalid evaluations

**Implemented Feature**: Skip dataset evaluation when conversational garbled text occurs

### 4.2 Feature Design Approach

During the model accuracy evaluation process, if the quantized model outputs garbled text, continuing with the full dataset evaluation wastes significant computing resources. Therefore, this feature performs a pre-check before the formal evaluation to detect whether the model output is garbled text. If garbled text is detected, the system skips the dataset evaluation and directly returns an evaluation result with an accuracy of 0.

The design approach includes:

1. **Pre-check Mechanism**: Before the formal evaluation, send test messages to detect whether the model output meets expectations
2. **Garbled Text Detection**: Use multiple check items (empty text, repeated characters, normal character ratio, control characters, and repeated patterns) to detect whether the model output is garbled text
3. **Intelligent Skipping**: If garbled text is detected, skip the dataset evaluation and directly return an evaluation result with an accuracy of 0 to save computing resources

### 4.3 Constraints

1. **Model Serving Requirement**: The model must be started in serving mode through vLLM-Ascend, supporting API calls
2. **Pre-check Configuration Requirement**: The precheck field must be configured in the tuning plan configuration file to specify the test cases for garbled text detection
3. **Network Requirement**: If the evaluation service runs remotely, a stable network connection is required

### 4.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

#### Processing Flow

```ASCII
User starts auto tuning
   │
   ▼
AutoTuningApplication.tune()
   │
   ▼
Evaluation service starts model serving
   │
   ▼
EvaluateService.evaluate()
   │
   ├─→ Check whether precheck is configured
   │   │
   │   ├─→ Yes: Execute pre-check
   │   │   │
   │   │   ├─→ GarbledTextRule.check()
   │   │   │   │
   │   │   │   ├─→ Iterate through test cases
   │   │   │   │   │
   │   │   │   │   ├─→ test_chat_via_api() sends test message
   │   │   │   │   │
   │   │   │   │   ├─→ is_garbled_text() detects garbled text
   │   │   │   │   │   │
   │   │   │   │   │   ├─→ Empty text check (EmptyTextCheckItem)
   │   │   │   │   │   ├─→ Repeated character check (RepeatedCharCheckItem)
   │   │   │   │   │   ├─→ Normal character ratio check (NormalCharRatioCheckItem)
   │   │   │   │   │   ├─→ Control character check (ControlCharCheckItem)
   │   │   │   │   │   └─→ Repeated pattern check (RepeatedPatternCheckItem)
   │   │   │   │   │
   │   │   │   │   └─→ If garbled text is detected: Return evaluation result with accuracy of 0
   │   │   │   │
   │   │   │   └─→ If all test cases pass: Continue with formal evaluation
   │   │   │
   │   │   └─→ No: Directly execute formal evaluation
   │   │
   │   └─→ Execute formal dataset evaluation
   │
   └─→ Return evaluation result
```

#### Module Interaction Description

1. **AutoTuningApplication**: Coordinates the entire tuning process and calls the evaluation service for accuracy evaluation
2. **EvaluateService**: Responsible for model accuracy evaluation; checks whether precheck is configured before executing the formal evaluation
3. **GarbledTextRule**: Implements the garbled text detection pre-check rule; detects whether the model output is garbled text through multiple check items
4. **GarbledTextCheckItem**: Interface implementation for various garbled text detection check items, including empty text, repeated characters, normal character ratio, control characters, and repeated pattern check items

### 4.5 Subsystem Interfaces (Module Interface Definitions)

#### New Interfaces

1. **GarbledTextPrecheckConfig** (`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
   - Type: Pydantic BaseModel
   - Function: Garbled text detection pre-check configuration, containing the test case list
   - Fields:
     - `type`: Literal["garbled_text"], fixed value
     - `test_cases`: Optional[List[TestCaseConfig]], test case list

2. **GarbledTextRule** (`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
   - Type: BasePrecheckRule subclass
   - Function: Implements the garbled text detection pre-check rule
   - Methods:
     - `is_garbled_text(text: str, check_items: List[str]) -> bool`: Detects whether the text is garbled text
     - `check(host: str, port: int, served_model_name: str, datasets: List[str]) -> Optional[List[EvaluateAccuracy]]`: Executes the garbled text detection pre-check

3. **GarbledTextCheckItem** (`msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`)
   - Type: ABC abstract base class
   - Function: Garbled text detection check item interface
   - Subclasses:
     - `EmptyTextCheckItem`: Empty text check
     - `RepeatedCharCheckItem`: Repeated character check
     - `NormalCharRatioCheckItem`: Normal character ratio check
     - `ControlCharCheckItem`: Control character check
     - `RepeatedPatternCheckItem`: Repeated pattern check

#### Modified Interfaces

1. **BasePrecheckRule.check()** (`msmodelslim/infra/evaluation/precheck/base.py`)
   - Function extension: Supports returning accuracy evaluation results; if the pre-check fails, returns an evaluation result with an accuracy of 0

### 4.6 Subsystem Detailed Design

#### 4.6.1 Garbled Text Detection Check Item Design

The garbled text detection adopts the chain of responsibility pattern and supports the combined use of multiple check items:

1. **EmptyTextCheckItem**: Checks whether the text is empty
   - Implementation: Checks whether the text is empty after removing whitespace characters
   - Threshold: None

2. **RepeatedCharCheckItem**: Checks for a large number of consecutive repeated characters
   - Implementation: Calculates the maximum length of consecutive repeated characters in the text; if it exceeds 30% of the text length, the text is identified as garbled text
   - Threshold: 0.3 (configurable)

3. **NormalCharRatioCheckItem**: Checks whether the normal character ratio is too low
   - Implementation: Calculates the ratio of Chinese characters, English characters, digits, and common punctuation marks; if the ratio is below 50%, the text is identified as garbled text
   - Threshold: 0.5 (configurable)

4. **ControlCharCheckItem**: Checks for a large number of control characters
   - Implementation: Calculates the ratio of control characters (excluding newline, carriage return, and tab characters); if the ratio exceeds 10%, the text is identified as garbled text
   - Threshold: 0.1 (configurable)

5. **RepeatedPatternCheckItem**: Checks for obvious repeated patterns
   - Implementation: Checks whether the pattern at the beginning of the text repeatedly appears in the text; if the repetition count reaches the threshold and the ratio of the repeated portion to the total length reaches the threshold, the text is identified as garbled text
   - Threshold: min_pattern_count=3, min_pattern_ratio=0.5 (configurable)

#### 4.6.2 Pre-check Flow Design

1. **Configuration Parsing**: Parse the precheck configuration from the tuning plan configuration file and create a GarbledTextPrecheckConfig object
2. **Test Case Execution**: Iterate through the configured test cases; for each test case:
   - Send the test message to the model service through the API
   - Obtain the model response
   - Use the configured check items to detect whether the response is garbled text
   - If garbled text is detected, record a warning log and return an evaluation result with an accuracy of 0
3. **Continue Evaluation**: If all test cases pass, continue with the formal dataset evaluation

### 4.7 DFX Attribute Design

#### 4.7.1 Performance Design

1. **Pre-check Overhead**: The pre-check sends only a small number of test messages (typically 1 to 3), resulting in minimal overhead. Compared with the full dataset evaluation, it can save more than 90% of the time
2. **Check Item Performance**: The implementations of all check items have O(n) time complexity, where n is the text length. The performance overhead is negligible
3. **Impact on Existing Features**: The pre-check is optional. If precheck is not configured, the existing evaluation process is not affected

#### 4.7.2 Upgrade and Expansion Design

1. **Configuration Compatibility**: The pre-check feature of the new version is compatible with the configuration file format of the old version. If the precheck field is not configured, the pre-check is skipped
2. **Check Item Extensibility**: New check items are supported through a registration mechanism, without affecting the use of existing check items

#### 4.7.3 Exception Handling Design

1. **API Call Exception**: If the test message fails to send, record a warning log and continue with the next test case. The evaluation process is not interrupted
2. **Check Item Exception**: If a check item fails to execute, record a warning log and continue with the next check item
3. **Configuration Exception**: If the precheck configuration format is incorrect, record an error log and skip the pre-check. Continue with the formal evaluation

#### 4.7.4 Resource Management Design

1. **Memory Usage**: The pre-check requires only a small amount of memory to store test messages and responses. The memory usage is negligible
2. **Network I/O**: The pre-check sends a small number of HTTP requests. The network I/O overhead is minimal
3. **Computing Resources**: The computing overhead of the pre-check is minimal and does not cause a noticeable impact on system performance

#### 4.7.5 Miniaturization Design

This feature does not affect the specifications of the miniaturization version. The pre-check feature is lightweight, with minimal memory and CPU usage.

#### 4.7.6 Testability Design

Testing should cover the following aspects:

1. **Functional Testing**:
   - Normal text passes all check items
   - Empty text is detected as garbled text
   - Repeated character text is detected as garbled text
   - Control character text is detected as garbled text
   - Repeated pattern text is detected as garbled text
   - Mixed garbled text is detected as garbled text

2. **Boundary Value Testing**:
   - Text length is 0
   - Text length is 1
   - Text length is very large (>10000 characters)
   - Check item threshold boundary values

3. **Exception Scenario Testing**:
   - API call failure
   - Check item execution exception
   - Configuration format error
   - Network interruption

4. **Performance Testing**:
   - Pre-check duration testing
   - Performance testing with a large number of test cases

#### 4.7.7 Security Design

##### 4.7.7.1 Security Design Confirmation

| Security Attribute | Check Item | Check Item Description | Involved | Satisfied |
| --- | --- | --- | --- | --- |
| Access Channel Control | Whether new listening ports are added | New listening ports require refreshing the communication matrix | No | Not involved |
| Access Channel Control | Whether new inter-process or inter-component communication is added | New inter-process or inter-component communication requires refreshing the communication matrix | Yes | Satisfied |
| Access Channel Control | Whether new authentication methods are added | New authentication methods require refreshing the communication matrix and product documentation | No | Not involved |
| Permission Control | Whether file or directory creation is involved | File or directory creation must explicitly specify access permissions | No | Not involved |
| Permission Control | Whether account permissions satisfy the principle of least privilege | Each account in the system should be granted minimum permissions | Yes | Satisfied |
| Permission Control | Whether user privilege escalation exists | User illegal privilege escalation is prohibited | No | Not involved |
| Undisclosed Interface | Whether new GUC parameters are added | New GUC parameters require refreshing product documentation | No | Not involved |
| Undisclosed Interface | Whether new or modified functions, views, or system tables are added | New or modified functions, views, or system tables require refreshing product documentation and considering permission control | No | Not involved |
| Undisclosed Interface | Whether new SQL syntax is added | New SQL syntax requires refreshing product documentation and supporting audit log recording | No | Not involved |
| Undisclosed Interface | Whether new internal tools are added | New internal tools require refreshing product documentation | No | Not involved |
| Undisclosed Interface | Whether commented-out code exists in scripts | Interpreted languages such as Shell and Python prohibit commented-out code. Commented-out code must be deleted | No | Not involved |
| Undisclosed Interface | Whether hidden commands, parameters, or ports exist as access methods | Access methods not used during network maintenance (including but not limited to product production, testing, and maintenance purposes) must be deleted (such as through compilation macros) | No | Not involved |
| Undisclosed Interface | Whether the system has hidden backdoors | The system is prohibited from reserving any undisclosed accounts. All accounts must be managed by the system and documented | No | Not involved |
| Undisclosed Interface | Prohibit providing cracking or network sniffing tools in software released to external users (including software packages and patch packages) | 1. Prohibit providing functions or tools that can modify any user password, have password cracking capabilities, or decrypt files containing sensitive data (such as configuration files containing keys or databases) in software released to external users. 2. Prohibit retaining third-party network sniffing tools such as tcpdump, gdb, strace, readelf network and process debugging tools, cpp, gcc, dexdump, mirror, JDK development and compilation tools, and self-developed debugging tools or scripts used only in the testing phase (such as encryption and decryption scripts used only in the debugging phase, testing functions, and commands that can escalate privileges). If retention is necessary due to business needs, strict access control must be implemented. The reason for retention, usage scenarios, and risks must also be documented. | No | Not involved |
| Sensitive Data Protection | Authentication credentials must not be stored in plaintext in the system and should be encrypted | Authentication credentials (such as passwords and private keys) must not be stored in plaintext in the system and should be encrypted | No | Not involved |
| Sensitive Data Protection | Keys used for sensitive data transmission encryption must not be hardcoded in the code | Hardcoding of passwords and keys is prohibited | No | Not involved |
| Sensitive Data Protection | Whether passwords or keys and other sensitive information are printed in plaintext | Printing plaintext sensitive information (passwords, private keys, pre-shared keys) in logs, debugging information, error messages, and ps command output stored in the system is prohibited | No | Not involved |
| Sensitive Data Protection | Whether passwords are displayed in plaintext | Displaying passwords in plaintext is prohibited | No | Not involved |
| Sensitive Data Protection | Whether default passwords of third-party and open source software are used | Using default passwords of third-party and open source software is prohibited. Refer to Section 1.5 of the Security Design Guide | No | Not involved |
| Sensitive Data Protection | Whether passwords are stored in plaintext in configuration files | Plaintext passwords must not be written into configuration files (except for scenarios where password configuration is required during command-line tool installation, deployment, and use) | No | Not involved |
| Sensitive Data Protection | Whether insecure encryption algorithms are used | Private or industry-known insecure encryption algorithms are prohibited. Refer to Section 6.2 of the Security Design Guide for recommended encryption algorithms | No | Not involved |
| Sensitive Data Protection | Whether sensitive information such as passwords uses secure transmission channels | Sensitive information transmission between untrusted networks must use secure transmission channels or encrypted transmission. Refer to Chapter 10 of the Security Design Guide | Yes | Satisfied |
| Sensitive Data Protection | Whether passwords or keys and other sensitive information in memory are destroyed after use | Passwords or keys in memory must be cleared to zero immediately after use | No | Not involved |
| Sensitive Data Protection | Random numbers used in cryptographic algorithms must be cryptographically secure random numbers | Random numbers used in cryptographic algorithms must be cryptographically secure random numbers. Refer to Section 6.3 of the Security Design Guide | No | Not involved |
| Sensitive Data Protection | Whether unsafe samples exist in documentation | Samples in documentation must be safe and provide correct guidance to users. If potential risks exist in samples, they must be documented | Yes | Satisfied |
| Authentication | Whether an authentication mechanism is provided | New systems must provide an authentication mechanism and enable it by default | No | Not involved |
| Authentication | Whether authentication is performed on the server side | The authentication process must be performed on the server side | No | Not involved |
| Authentication | Whether the server returns valid information after authentication failure | After authentication failure, the server return information must not provide detailed hints that can be used to determine the specific error cause | No | Not involved |
| External Parameter Validation | Whether external input is validated for legitimacy | 1. Using external input data as loop termination conditions, array subscripts, or memory allocation size parameters may cause infinite loops, buffer overflows, memory out-of-bounds, denial of service, and other behaviors. 2. External input such as file paths should be validated for legitimacy to prevent injection risks | Yes | Satisfied |
| Third-Party Component Introduction | Whether new third-party components are introduced | 1. New third-party components must pass security compilation options, virus, vulnerability, open source segment reference, license compliance, and open source component scanning. Refer to the Version Release Network Security Quality Requirements. 2. New third-party components must ensure trusted sources | No | Not involved |

##### 4.7.7.2 Sensitive Data Analysis

This Use Case does not involve the processing of sensitive data. It mainly performs garbled text detection on model output text and does not involve sensitive operations such as user authentication or key management.

##### 4.7.7.3 Design Implementation

The security design of this Use Case is mainly reflected in:

1. **External Input Validation**: Validate the legitimacy of test messages and model responses to prevent injection attacks
2. **Network Communication Security**: If the evaluation service runs remotely, use secure transmission channels such as HTTPS
3. **Error Message Handling**: Error messages do not leak sensitive information and only record necessary debugging information

### 4.8 System External Interfaces

This Use Case does not affect system external interfaces. It is mainly an internal implementation optimization. Users can enable or disable the garbled text detection feature through the precheck field in the tuning plan configuration file.

### 4.9 Self-Test Case Design

The self-test cases are designed as follows:

1. **Normal Text Test**:
   - Input: Normal Chinese text "Hello, World"
   - Expected: Pass all check items and continue with the formal evaluation

2. **Empty Text Test**:
   - Input: Empty string or string containing only whitespace characters
   - Expected: Detected as garbled text by EmptyTextCheckItem, returning an evaluation result with an accuracy of 0

3. **Repeated Character Test**:
   - Input: Text containing a large number of consecutive repeated characters, such as "aaaaaaaaaa..."
   - Expected: Detected as garbled text by RepeatedCharCheckItem, returning an evaluation result with an accuracy of 0

4. **Control Character Test**:
   - Input: Text containing a large number of control characters
   - Expected: Detected as garbled text by ControlCharCheckItem, returning an evaluation result with an accuracy of 0

5. **Repeated Pattern Test**:
   - Input: Text containing obvious repeated patterns, such as "abcabcabc..."
   - Expected: Detected as garbled text by RepeatedPatternCheckItem, returning an evaluation result with an accuracy of 0

6. **Mixed Garbled Text Test**:
   - Input: Text containing multiple garbled text characteristics
   - Expected: Detected as garbled text by at least one check item, returning an evaluation result with an accuracy of 0

7. **API Call Failure Test**:
   - Input: Simulate API call failure
   - Expected: Record a warning log, continue with the next test case, and do not interrupt the evaluation process

8. **Configuration Error Test**:
   - Input: precheck configuration format error
   - Expected: Record an error log, skip the pre-check, and continue with the formal evaluation

## 5. Use Case 2 Implementation

### 5.1 Use Case Description

**Use Case Name**: Checkpoint Resume After Accuracy Tuning Unexpected Interruption

**Use Case Scenario**:

- During the auto tuning process, the tuning is unexpectedly interrupted (such as system failure or manual stop)
- The user expects the system to reuse historical records after restarting auto tuning and continue the accuracy tuning process
- The system automatically detects the historical accuracy cache upon restart
- If a historical accuracy cache exists, the system reuses the evaluated quantization configuration results to avoid repeated evaluations

**Impact on the Auto Tuning Feature**:

- Requires supporting the checkpoint resume function
- Requires implementing the historical accuracy cache mechanism
- Requires supporting the recovery of evaluated configuration results from the historical cache

**Implemented Feature**: Auto tuning supports checkpoint resume

### 5.2 Feature Design Approach

During the auto tuning process, if the tuning is unexpectedly interrupted (such as system failure or manual stop), the system must be able to recover the evaluated quantization configuration results from the historical accuracy cache upon restart. This avoids repeated evaluations of the same configurations and enables checkpoint resume of the tuning process.

The design approach includes:

1. **Accuracy Cache Mechanism**: Save the quantization configuration and accuracy evaluation results of each iteration to the accuracy cache, using the MD5 hash value as the unique identifier of the configuration
2. **History Detection Mechanism**: Detect whether a historical accuracy cache exists at the start of tuning; if it exists, load it into memory
3. **Cache Reuse Mechanism**: In each iteration, first attempt to find a matching quantization configuration in the accuracy cache. If found, directly use the historical evaluation result and skip the quantization, serving startup, pre-check, and evaluation steps

### 5.3 Constraints

1. **Storage Requirement**: Sufficient storage space is required to save the accuracy cache. The accuracy cache file is in YAML format
2. **Path Consistency**: Checkpoint resume requires using the same save_path. The system looks for the accuracy cache in the save_path/history directory
3. **Configuration Consistency**: Checkpoint resume requires the quantization configuration to be completely consistent (matched through the MD5 hash value). If the configuration has changed, reuse is not possible

### 5.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

#### Processing Flow

```ASCII
User starts auto tuning (possibly checkpoint resume)
   │
   ▼
AutoTuningApplication.tune()
   │
   ▼
TuningHistoryManager.load_history()
   │
   ├─→ Detect save_path/history directory
   │   │
   │   ├─→ If accuracy.yaml exists: Load accuracy cache into memory
   │   │
   │   └─→ If not exists: Create empty accuracy cache
   │
   ▼
YamlTuningHistory._load_accuracy_database()
   │
   ├─→ Read accuracy.yaml file
   │
   ├─→ Parse into dictionary format (key is MD5, value is evaluation result)
   │
   └─→ Load into in-memory _accuracy_cache
   │
   ▼
Start iterative tuning loop
   │
   ├─→ Generate quantization configuration (PracticeConfig)
   │
   ├─→ Calculate MD5 hash value of the configuration
   │
   ├─→ Look for matching configuration in accuracy cache
   │   │
   │   ├─→ If found: Directly use historical evaluation result, skip quantization and evaluation steps
   │   │
   │   └─→ If not found: Execute quantization and evaluation steps
   │       │
   │       ├─→ Quantize model
   │       │
   │       ├─→ Evaluate model accuracy
   │       │
   │       └─→ Save to accuracy cache
   │
   └─→ Continue with the next iteration
```

#### Module Interaction Description

1. **AutoTuningApplication**: Coordinates the entire tuning process; loads the historical accuracy cache at the start of tuning and attempts to recover evaluation results from the cache in each iteration
2. **TuningHistoryManager**: Manages tuning history; responsible for loading and saving the accuracy cache
3. **YamlTuningHistory**: Implements YAML-based accuracy cache management, including loading, saving, and querying functions
4. **calculate_practice_md5**: Calculates the MD5 hash value of the quantization configuration for unique identification

### 5.5 Subsystem Interfaces (Module Interface Definitions)

#### New Interfaces

1. **TuningHistoryInfra** (`msmodelslim/app/auto_tuning/practice_history_infra.py`)
   - Type: ABC abstract base class
   - Function: Tuning history operation interface
   - Methods:
     - `get_accuracy(practice: PracticeConfig) -> Optional[EvaluateResult]`: Retrieves the accuracy evaluation result from history
     - `append_history(practice: PracticeConfig, evaluation: EvaluateResult) -> None`: Appends a history record
     - `clear_records() -> None`: Clears history records (but retains the accuracy cache)
     - `get_accuracy_count() -> int`: Returns the number of accuracy records

2. **TuningHistoryManagerInfra** (`msmodelslim/app/auto_tuning/practice_history_infra.py`)
   - Type: ABC abstract base class
   - Function: Tuning history manager interface
   - Methods:
     - `load_history(database: str) -> TuningHistoryInfra`: Loads tuning history

3. **YamlTuningHistory** (`msmodelslim/infra/yaml_practice_history_manager.py`)
   - Type: TuningHistoryInfra implementation class
   - Function: YAML-based tuning history implementation
   - Data Files:
     - `accuracy.yaml`: Accuracy cache, key is MD5, value is evaluation result
     - `history.yaml`: History index, recording the configuration ID and evaluation result of each iteration

4. **calculate_practice_md5** (`msmodelslim/infra/yaml_practice_history_manager.py`)
   - Type: Function
   - Function: Calculates the MD5 hash value of the quantization configuration
   - Implementation: Serializes the configuration to a JSON string and calculates the MD5 hash value

#### Modified Interfaces

1. **AutoTuningApplication._tune()** (`msmodelslim/app/auto_tuning/application.py`)
   - Function extension: Loads the historical accuracy cache at the start of tuning and attempts to recover evaluation results from the cache in each iteration

### 5.6 Subsystem Detailed Design

#### 5.6.1 Accuracy Cache Design

The accuracy cache uses YAML format for storage, with the following structure:

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

The key of the accuracy cache is the MD5 hash value of the quantization configuration, and the value is the evaluation result (a dictionary serialized from the EvaluateResult object).

#### 5.6.2 History Index Design

The history index uses YAML format for storage, with the following structure:

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

The history index records the configuration ID, evaluation result, MD5 hash value, and timestamp of each iteration.

#### 5.6.3 Cache Reuse Mechanism Design

1. **Configuration Matching**: Use the MD5 hash value to match quantization configurations, ensuring the configuration is completely consistent
2. **Cache Lookup**: In each iteration, first calculate the MD5 hash value of the current configuration, then look for a matching configuration in the accuracy cache
3. **Result Reuse**: If a matching configuration is found, directly use the historical evaluation result and skip the quantization, serving startup, pre-check, and evaluation steps
4. **Cache Update**: If no matching configuration is found, execute the quantization and evaluation steps, then save the result to the accuracy cache

#### 5.6.4 Checkpoint Resume Flow Design

1. **History Detection**: At the start of tuning, detect whether the accuracy.yaml file exists in the save_path/history directory
2. **Cache Loading**: If it exists, load the accuracy cache into memory; if it does not exist, create an empty accuracy cache
3. **Iteration Recovery**: In each iteration, first attempt to recover the evaluation result from the accuracy cache. If found, reuse it; if not found, execute the full quantization and evaluation process
4. **Cache Saving**: After each iteration, save the evaluation result to the accuracy cache to ensure reuse upon the next startup

### 5.7 DFX Attribute Design

#### 5.7.1 Performance Design

1. **Cache Loading Performance**: The accuracy cache is loaded into memory, with a lookup performance of O(1), which does not affect tuning performance
2. **MD5 Calculation Performance**: The MD5 hash value calculation overhead is minimal and does not affect tuning performance
3. **Impact on Existing Features**: The checkpoint resume function is optional. If the same save_path is not used, the existing tuning process is not affected

#### 5.7.2 Upgrade and Expansion Design

1. **Data Format Compatibility**: The data format design of the accuracy cache considers version compatibility and supports cross-version use
2. **Storage Extensibility**: The accuracy cache uses YAML format, which is easy to extend and maintain

#### 5.7.3 Exception Handling Design

1. **File Read Exception**: If the accuracy cache file fails to read, record a warning log and create an empty accuracy cache. The tuning process is not interrupted
2. **Data Parsing Exception**: If the accuracy cache data format is incorrect, record an error log and create an empty accuracy cache. The tuning process is not interrupted
3. **Storage Exception**: If the accuracy cache fails to save, record an error log but do not interrupt the tuning process. Reuse may not be possible upon the next startup

#### 5.7.4 Resource Management Design

1. **Memory Usage**: The accuracy cache is loaded into memory. The memory usage depends on the cache size, typically ranging from a few MB to tens of MB
2. **Disk I/O**: The read and write operations of the accuracy cache are infrequent, and the disk I/O overhead is minimal
3. **Storage Space**: The accuracy cache file size depends on the number of evaluation results, typically ranging from a few KB to a few MB

#### 5.7.5 Miniaturization Design

This feature does not affect the specifications of the miniaturization version. The accuracy cache function is lightweight, with minimal memory and storage usage.

#### 5.7.6 Testability Design

Testing should cover the following aspects:

1. **Functional Testing**:
   - First-time tuning: Create an empty accuracy cache
   - Checkpoint resume: Recover evaluation results from the historical accuracy cache
   - Configuration matching: Same configurations can be correctly matched
   - Configuration mismatch: Different configurations cannot be matched

2. **Boundary Value Testing**:
   - Accuracy cache is empty
   - Accuracy cache contains a large number of records (>1000 entries)
   - MD5 hash value collision (theoretically impossible, but needs testing)

3. **Exception Scenario Testing**:
   - Accuracy cache file does not exist
   - Accuracy cache file format is incorrect
   - Accuracy cache file read failure
   - Accuracy cache file save failure
   - Insufficient storage space

4. **Performance Testing**:
   - Accuracy cache loading duration testing
   - Accuracy cache lookup duration testing
   - Cache performance testing with a large number of records

#### 5.7.7 Security Design

##### 5.7.7.1 Security Design Confirmation

The security design of this Use Case is similar to Use Case 1, mainly focusing on the security of file operations and not involving the processing of sensitive data.

##### 5.7.7.2 Sensitive Data Analysis

This Use Case does not involve the processing of sensitive data. It mainly stores and queries quantization configurations and evaluation results, and does not involve sensitive operations such as user authentication or key management.

##### 5.7.7.3 Design Implementation

The security design of this Use Case is mainly reflected in:

1. **File Permission Control**: The accuracy cache file uses secure file permissions to prevent unauthorized access
2. **Data Integrity**: Use MD5 hash values to ensure the uniqueness and integrity of configurations
3. **Error Handling**: Error messages do not leak sensitive information and only record necessary debugging information

### 5.8 System External Interfaces

This Use Case does not affect system external interfaces. It is mainly an internal implementation optimization. Users can achieve the checkpoint resume function by using the same save_path.

### 5.9 Self-Test Case Design

The self-test cases are designed as follows:

1. **First-Time Tuning Test**:
   - Input: New save_path, no historical accuracy cache exists
   - Expected: Create an empty accuracy cache and execute the tuning process normally

2. **Checkpoint Resume Test**:
   - Input: Same save_path, historical accuracy cache exists
   - Expected: Load the historical accuracy cache and reuse evaluated configuration results during iteration

3. **Configuration Matching Test**:
   - Input: Same quantization configuration
   - Expected: Correctly matched, reusing the historical evaluation result

4. **Configuration Mismatch Test**:
   - Input: Different quantization configuration
   - Expected: Not matched, executing the full quantization and evaluation process

5. **Cache File Exception Test**:
   - Input: Accuracy cache file does not exist or format is incorrect
   - Expected: Record a warning log, create an empty accuracy cache, and do not interrupt the tuning process

6. **Large Number of Records Test**:
   - Input: Accuracy cache contains a large number of records (>1000 entries)
   - Expected: Load and lookup normally, with performance meeting requirements

## 6. Use Case 3 Implementation

### 6.1 Use Case Description

**Use Case Name**: Tuning Strategy with Built-in Expert Experience

**Use Case Scenario**:

- When configuring auto tuning, the user is unfamiliar with the search space configuration for quantization tuning
- The user expects the system to automatically look up the algorithm search space based on the model structure type (such as MHA, MLA, DSA, SWA, or GatedDeltaNet)
- The system automatically obtains the recommended algorithm search space based on the model structure type, simplifying the user configuration operation

**Impact on the Auto Tuning Feature**:

- Requires providing an expert experience-based tuning strategy
- Requires implementing the model structure type identification function
- Requires implementing the expert experience table mechanism
- Requires supporting automatic lookup to obtain the search space

**Implemented Feature**: Expert experience-based tuning strategy

### 6.2 Feature Design Approach

The current standing_high strategy requires users to manually input the algorithm search space (anti_outlier_strategies), which has high configuration complexity for users unfamiliar with quantization tuning. This feature creates a new independent tuning strategy module `expert_experience` that supports automatic lookup of the algorithm search space based on the model structure type (such as MHA, MLA, DSA, SWA, or GatedDeltaNet), without requiring users to manually input the search space configuration.

The new strategy module can reuse the core logic of the standing_high strategy (such as the reach-high algorithm), but as an independent strategy implementation, it has the following characteristics:

1. **Independent Module**: Create a new strategy directory `msmodelslim/core/tune_strategy/expert_experience/`, containing independent strategy configuration and implementation
2. **Model Structure Type Identification**: Support identifying the attention mechanism type of the model, such as MHA (Multi-Head Attention), MLA (Multi-Head Latent Attention), DSA (Distributed Sparse Attention), SWA (Sliding Window Attention), and GatedDeltaNet
3. **Expert Experience Table**: Maintain an expert experience table that records the recommended algorithm search spaces corresponding to different model structure types
4. **Automatic Lookup**: Automatically look up the algorithm search space based on the model structure type. If no matching type is found, use the default search space
5. **Strategy Reuse**: Can reuse the reach-high algorithm logic of the standing_high strategy, but automatically obtains the search space through the expert experience table, simplifying user configuration

### 6.3 Constraints

1. **Model Structure Support**: The model adapter must be able to identify and provide model structure type information
2. **Expert Experience Table Maintenance**: The expert experience table must be maintained to record the recommended algorithm search spaces corresponding to different model structure types
3. **Strategy Registration**: The new strategy must be registered in setup.py, including the strategy configuration and strategy implementation entry points
4. **Backward Compatibility**: The new strategy does not affect the existing standing_high strategy. Both can coexist

### 6.4 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from User Entry)

#### Processing Flow

```ASCII
User starts auto tuning (using expert_experience strategy)
   │
   ▼
ExpertExperienceStrategy.__init__()
   │
   ├─→ Obtain model structure type
   │   │
   │   └─→ ModelAdapter.get_attention_type()
   │
   ├─→ Look for matching type in expert experience table
   │   │
   │   └─→ ExpertExperienceTable.get_search_space(attention_type)
   │
   ├─→ If found: Use lookup result as anti_outlier_strategies
   │
   └─→ If not found: Use default search space
   │
   ▼
Create StandingHighStrategy instance (reuse reach-high algorithm logic)
   │
   ├─→ Use automatically obtained anti_outlier_strategies
   │
   └─→ Execute standing_high strategy reach-high algorithm
      │
      └─→ Perform tuning using the automatically obtained search space
```

#### Module Interaction Description

1. **ExpertExperienceStrategy**: New tuning strategy implementation; responsible for automatically obtaining the algorithm search space based on the model structure type
2. **ExpertExperienceTable**: Expert experience table; maintains the recommended algorithm search spaces corresponding to different model structure types
3. **ModelAdapter**: Model adapter; provides model structure type information
4. **StandingHighStrategy**: standing_high strategy implementation; its core logic can be reused by ExpertExperienceStrategy

### 6.5 Subsystem Interfaces (Module Interface Definitions)

#### New Interfaces

1. **ExpertExperienceStrategyConfig** (`msmodelslim/core/tune_strategy/expert_experience/strategy.py`)
   - Type: StrategyConfig subclass
   - Function: Expert experience strategy configuration; inherits the configuration items of StandingHighStrategyConfig, but the anti_outlier_strategies field is optional
   - Fields:
     - `type`: Literal["expert_experience"], fixed value
     - `anti_outlier_strategies`: Optional[List[AutoProcessorConfigList]], optional; if not specified, it is automatically obtained
     - `template`: ModelslimV1ServiceConfig, quantization template configuration
     - `metadata`: Metadata, metadata configuration

2. **ExpertExperienceStrategy** (`msmodelslim/core/tune_strategy/expert_experience/strategy.py`)
   - Type: BaseTuningStrategy subclass, ITuningStrategy implementation
   - Function: Expert experience strategy implementation; automatically obtains the algorithm search space based on the model structure type, then reuses the reach-high algorithm logic of the standing_high strategy
   - Methods:
     - `__init__(config: StrategyConfig, dataset_loader: DatasetLoaderInfra)`: Initializes the strategy and automatically obtains the search space
     - `generate_practice(model: IModel, device: DeviceType) -> Generator[PracticeConfig, Optional[EvaluateResult], None]`: Generates quantization configurations, reusing the logic of the standing_high strategy

3. **ExpertExperienceTable** (`msmodelslim/core/tune_strategy/expert_experience/expert_experience_table.py`)
   - Type: Class
   - Function: Expert experience table; maintains the recommended algorithm search spaces corresponding to different model structure types
   - Methods:
     - `get_search_space(attention_type: str) -> Optional[List[AutoProcessorConfigList]]`: Obtains the recommended algorithm search space based on the model structure type
     - `get_default_search_space() -> List[AutoProcessorConfigList]`: Obtains the default algorithm search space

4. **ModelAdapter.get_attention_type()** (`msmodelslim/model/base.py`)
   - Type: Method (requires implementation by the model adapter)
   - Function: Obtains the attention mechanism type of the model
   - Returns: str, such as "MHA", "MLA", "DSA", "SWA", or "GatedDeltaNet"

#### New Directory Structure

```tex
msmodelslim/core/tune_strategy/expert_experience/
├── __init__.py
├── strategy.py              # ExpertExperienceStrategyConfig and ExpertExperienceStrategy
└── expert_experience_table.py  # ExpertExperienceTable
```

#### Strategy Registration

Register the new strategy in setup.py:

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

### 6.6 Subsystem Detailed Design

#### 6.6.1 New Strategy Module Design

ExpertExperienceStrategy is an independent new strategy module. The design key points are:

1. **Strategy Configuration**: ExpertExperienceStrategyConfig inherits StandingHighStrategyConfig, but the anti_outlier_strategies field is changed to optional
2. **Strategy Implementation**: ExpertExperienceStrategy automatically obtains the search space during initialization, then creates a StandingHighStrategy instance to reuse its core logic
3. **Strategy Reuse**: Reuse the reach-high algorithm of the standing_high strategy through composition, rather than directly modifying the standing_high strategy

#### 6.6.2 Model Structure Type Identification Design

Model structure type identification is implemented through the model adapter. The model adapter must implement the get_attention_type() method to return the attention mechanism type of the model. Common model structure types include:

1. **MHA**: Multi-Head Attention, the standard multi-head attention mechanism
2. **MLA**: Multi-Head Latent Attention, the multi-head latent attention mechanism (such as DeepSeek-V3.2)
3. **DSA**: Distributed Sparse Attention, the distributed sparse attention mechanism
4. **SWA**: Sliding Window Attention, the sliding window attention mechanism
5. **GatedDeltaNet**: Gated Delta Network attention mechanism

#### 6.6.3 Expert Experience Table Design

The expert experience table adopts a dictionary structure, with the model structure type as the key and the recommended algorithm search space as the value. The sample structure is as follows:

```python
EXPERT_EXPERIENCE_TABLE = {
    "MHA": [
        [LinearProcessorConfig(...), ...],  # Strategy 1
        [LinearProcessorConfig(...), ...],  # Strategy 2
        ...
    ],
    "MLA": [
        [LinearProcessorConfig(...), ...],  # Strategy 1
        [LinearProcessorConfig(...), ...],  # Strategy 2
        ...
    ],
    "DSA": [
        [LinearProcessorConfig(...), ...],  # Strategy 1
        ...
    ],
    ...
}
```

The expert experience table is maintained based on historical tuning experience and records the recommended algorithm search spaces corresponding to different model structure types.

#### 6.6.4 Automatic Lookup Mechanism Design

1. **Type Identification**: First obtain the model structure type through the model adapter
2. **Lookup Matching**: Look for the matching type in the expert experience table
3. **Result Usage**: If a matching type is found, use the lookup result; if not found, use the default search space
4. **Strategy Reuse**: Pass the automatically obtained search space to StandingHighStrategy to reuse its reach-high algorithm logic

#### 6.6.5 Strategy Implementation Approach

The implementation approach of ExpertExperienceStrategy:

1. **Initialization Phase**:
   - Check whether the user has specified anti_outlier_strategies
   - If not specified, obtain the model structure type through the model adapter
   - Look for the matching type in the expert experience table to obtain the recommended algorithm search space
   - If not found, use the default search space

2. **Strategy Execution Phase**:
   - Create StandingHighStrategyConfig using the automatically obtained anti_outlier_strategies
   - Create a StandingHighStrategy instance to reuse its reach-high algorithm logic
   - Call StandingHighStrategy.generate_practice() to execute tuning

### 6.7 DFX Attribute Design

#### 6.7.1 Performance Design

1. **Lookup Performance**: The expert experience table lookup performance is O(1), which does not affect tuning performance
2. **Type Identification Performance**: The model structure type identification overhead is minimal and does not affect tuning performance
3. **Impact on Existing Features**: The automatic lookup function is optional. If the user manually specifies the search space, the existing tuning process is not affected

#### 6.7.2 Upgrade and Expansion Design

1. **Expert Experience Table Extensibility**: The expert experience table adopts a dictionary structure, which is easy to extend and maintain. New model structure types can be added at any time
2. **Backward Compatibility**: If the user manually specifies the search space, the user-specified configuration takes priority, maintaining backward compatibility

#### 6.7.3 Exception Handling Design

1. **Type Identification Exception**: If the model adapter cannot identify the model structure type, record a warning log and use the default search space
2. **Lookup Exception**: If no matching type is found in the expert experience table, record a warning log and use the default search space
3. **Configuration Exception**: If the lookup result format is incorrect, record an error log and use the default search space

#### 6.7.4 Resource Management Design

1. **Memory Usage**: The expert experience table is loaded into memory. The memory usage is minimal, typically a few KB
2. **Computing Resources**: The computing overhead of the lookup operation is minimal and does not cause a noticeable impact on system performance

#### 6.7.5 Miniaturization Design

This feature does not affect the specifications of the miniaturization version. The expert experience table function is lightweight, with minimal memory usage.

#### 6.7.6 Testability Design

Testing should cover the following aspects:

1. **Functional Testing**:
   - User specifies search space: Use the user-specified configuration
   - Automatic lookup success: Automatically obtain the search space based on the model structure type
   - Automatic lookup failure: Use the default search space
   - Type identification failure: Use the default search space

2. **Boundary Value Testing**:
   - Expert experience table is empty
   - Expert experience table contains a large number of types (>100 types)
   - Model structure type is unknown

3. **Exception Scenario Testing**:
   - Model adapter does not support type identification
   - Expert experience table format is incorrect
   - Lookup result format is incorrect

4. **Performance Testing**:
   - Lookup duration testing
   - Type identification duration testing

#### 6.7.7 Security Design

##### 6.7.7.1 Security Design Confirmation

The security design of this Use Case is similar to Use Case 1, mainly focusing on the security of configurations and not involving the processing of sensitive data.

##### 6.7.7.2 Sensitive Data Analysis

This Use Case does not involve the processing of sensitive data. It mainly queries and configures model structure types and algorithm search spaces, and does not involve sensitive operations such as user authentication or key management.

##### 6.7.7.3 Design Implementation

The security design of this Use Case is mainly reflected in:

1. **Configuration Validation**: Validate the lookup results to ensure the configuration format is correct
2. **Error Handling**: Error messages do not leak sensitive information and only record necessary debugging information

### 6.8 System External Interfaces

This Use Case affects system external interfaces:

1. **New Strategy Configuration Interface**: The anti_outlier_strategies field in ExpertExperienceStrategyConfig is optional. Users can choose not to specify this field, and the system will automatically obtain it based on the model structure type
2. **Model Adapter Interface**: The model adapter must implement the get_attention_type() method to provide model structure type information
3. **Strategy Selection Interface**: Users can select the "expert_experience" strategy in the tuning plan configuration file instead of the "standing_high" strategy
4. **Strategy Registration Interface**: The new strategy must be registered with entry points in setup.py, including the strategy configuration and strategy implementation

### 6.9 Self-Test Case Design

The self-test cases are designed as follows:

1. **Strategy Selection Test**:
   - Input: User selects the "expert_experience" strategy in the tuning plan configuration file
   - Expected: The system creates an ExpertExperienceStrategy instance instead of a StandingHighStrategy instance

2. **User-Specified Search Space Test**:
   - Input: User specifies anti_outlier_strategies in the configuration file
   - Expected: Use the user-specified configuration without performing automatic lookup

3. **Automatic Lookup Success Test**:
   - Input: Model structure type is "MHA", and the expert experience table contains this type
   - Expected: Automatically obtain the corresponding search space, create a StandingHighStrategy instance, and reuse its logic

4. **Automatic Lookup Failure Test**:
   - Input: Model structure type is "Unknown", and the expert experience table does not contain this type
   - Expected: Use the default search space, record a warning log, create a StandingHighStrategy instance, and reuse its logic

5. **Type Identification Failure Test**:
   - Input: Model adapter does not support type identification
   - Expected: Use the default search space, record a warning log, create a StandingHighStrategy instance, and reuse its logic

6. **Expert Experience Table Exception Test**:
   - Input: Expert experience table format is incorrect
   - Expected: Use the default search space, record an error log, create a StandingHighStrategy instance, and reuse its logic

7. **Strategy Reuse Test**:
   - Input: Tune using the expert_experience strategy
   - Expected: Correctly reuse the reach-high algorithm logic of the standing_high strategy, with tuning results consistent with the standing_high strategy

## 7. Reliability and Availability Design

### 7.1 Redundancy Design

The auto tuning acceleration feature adopts the following redundancy design:

1. **Accuracy Cache Redundancy**: The accuracy cache uses YAML format for persistent storage. Even if the tuning process is interrupted, the historical accuracy cache is retained, supporting checkpoint resume
2. **Configuration Backup**: The quantization configuration of each iteration is saved to the history records, supporting configuration backup and recovery
3. **Log Recording**: Detailed log recording supports problem identification and recovery

### 7.2 Fault Management

#### Fault Detection

1. **Accuracy Evaluation Failure Detection**: If the accuracy evaluation fails, record an error log and continue with the next iteration
2. **Quantization Failure Detection**: If the quantization fails, record an error log and continue with the next iteration
3. **Serving Startup Failure Detection**: If the model serving startup fails, record an error log and continue with the next iteration

#### Fault Isolation

1. **Iteration Isolation**: Each iteration is executed independently. A single iteration failure does not affect other iterations
2. **Module Isolation**: The quantization, evaluation, and pre-check modules are executed independently. A single module failure does not affect other modules

#### Fault Recovery

1. **Automatic Recovery**: Through the checkpoint resume mechanism, the system supports recovering evaluated configuration results from the historical accuracy cache
2. **Manual Recovery**: Users can restart tuning by using the same save_path, and the system automatically recovers the historical accuracy cache

### 7.3 Overload Control Design

1. **Iteration Count Limit**: Supports setting the maximum iteration count to prevent infinite iteration
2. **Timeout Control**: Supports setting the maximum iteration duration to prevent the tuning process from running indefinitely
3. **Resource Monitoring**: Monitors memory, storage, and other resource usage. If resources are insufficient, stop tuning

### 7.4 Service Continuity During Upgrade

1. **Configuration Compatibility**: The new version of the auto tuning function is compatible with the configuration file format of the old version
2. **Data Compatibility**: The format design of the historical accuracy cache considers version compatibility and supports cross-version use
3. **Interface Compatibility**: The interface design of auto tuning considers backward compatibility, and the calling method of the old version remains valid

### 7.5 Human Error Design

1. **Configuration Validation**: Validate the tuning plan configuration file. If the configuration is incorrect, provide clear error messages
2. **Parameter Validation**: Validate command-line parameters. If the parameters are incorrect, provide clear error messages
3. **Log Prompts**: Detailed log recording helps users understand the tuning process and results

### 7.6 Fault Prediction and Prevention Design

1. **Resource Monitoring**: Monitor memory, storage, and other resource usage to provide early warning of resource shortages
2. **Performance Monitoring**: Monitor performance metrics of the tuning process to provide early warning of performance issues
3. **Anomaly Detection**: Detect abnormal conditions (such as accuracy evaluation failure and quantization failure) to provide early warning of potential issues

## 8. Feature Non-functional Quality Attribute Design

### 8.1 Testability

_Focus on describing the testing directions and specifications of the feature, explaining which aspects testers should test, and which boundary values, exception values, and exception scenarios need attention._

### 8.2 Serviceability

_Provide rich maintainability and serviceability measures for the feature, and provide complete documentation for the use, maintenance, and problem handling of the feature._

### 8.3 Evolvability

_Focus on describing the evolvability of the feature architecture and functions._

### 8.4 Openness

_Focus on describing the openness of the external interfaces of the feature, including interface standardization, such as compliance with the __SQL 2011__ standard._

### 8.5 Compatibility

_Focus on describing whether the feature affects the forward compatibility of the system, that is, whether old functions can still be used after upgrading to the new version, and whether the usage behavior remains consistent with the old version._

### 8.6 Scalability and Extensibility

_Effectively meet the requirements of system capacity changes, including the scaling of database nodes and the scaling of database servers themselves._

### 8.7 Maintainability

_Focus on describing the maintainability of the feature, such as diagnostic views and __log__ printing._

### 8.8 Documentation

_Refer to the table below to evaluate the modification points of various types of documentation involved in the feature, and describe the specific modification points._

<table>
    <tr>
        <th>Category</th>
        <th>Manual Name</th>
        <th>Involved (Y/N)</th>
        <th>Brief Description of Modifications or Additions</th>
    </tr>
    <tr>
        <td>White Paper</td>
        <td>Technical White Paper</td>
        <td>N</td>
        <td>XX section adds XX technology</td>
    </tr>
    <tr>
        <td rowspan="8">Product Documentation</td>
        <td>Product Description</td>
        <td>Y</td>
        <td>Technical specifications refreshed to XX</td>
    </tr>
    <tr>
        <td>Feature Description</td>
        <td>Y</td>
        <td>Add XX feature</td>
    </tr>
    <tr>
        <td>Compilation Guide</td>
        <td>Y</td>
        <td>XXX</td>
    </tr>
    <tr>
        <td>Installation Guide</td>
        <td>Y</td>
        <td>Installation cluster section needs to refresh XX scenario</td>
    </tr>
    <tr>
        <td>Administrator Guide</td>
        <td>N</td>
        <td>XXX</td>
    </tr>
    <tr>
        <td>Developer Guide (including development tutorials, SQL reference, system tables and system views, GUC parameter descriptions, error code descriptions, and API reference)</td>
        <td>Y</td>
        <td>Add XXX function in XX section</td>
    </tr>
    <tr>
        <td>Tool Reference</td>
        <td>Y</td>
        <td>Add XX tool</td>
    </tr>
    <tr>
        <td>Glossary</td>
        <td>Y</td>
        <td>Add term XX</td>
    </tr>
    <tr>
        <td>Getting Started</td>
        <td>Quick Tutorial</td>
        <td>N</td>
        <td>XXX</td>
    </tr>
</table>

## 9. Data Structure Design (Optional)

The auto tuning acceleration feature primarily uses YAML format for data storage, including:

1. **Accuracy Cache (accuracy.yaml)**:
   - Structure: Dictionary format, with the MD5 hash value as the key and the evaluation result as the value
   - Purpose: Stores evaluated quantization configurations and accuracy results, supporting checkpoint resume

2. **History Index (history.yaml)**:
   - Structure: List format, where each element contains the configuration ID, evaluation result, MD5 hash value, and timestamp
   - Purpose: Records the configuration and evaluation result of each iteration, supporting historical queries

3. **Quantization Configuration (practice configs)**:
   - Structure: PracticeConfig object in YAML format
   - Purpose: Stores the quantization configuration of each iteration, supporting configuration backup and recovery

## 10. Reference List

1. **msModelSlim Tool Documentation**:
   - Auto Tuning Feature User Guide
   - Tuning Plan Configuration File Format Description
   - API Reference Documentation

2. **Related Code Implementations**:
   - `msmodelslim/app/auto_tuning/application.py`: Auto tuning application layer implementation
   - `msmodelslim/core/tune_strategy/standing_high/strategy.py`: standing_high strategy implementation
   - `msmodelslim/infra/evaluation/precheck/garbled_text_rule.py`: Garbled text detection pre-check implementation
   - `msmodelslim/infra/yaml_practice_history_manager.py`: History management module implementation

3. **Design Principles**:
   - Interface Standardization Principle
   - Unified Data Format Principle
   - Standardized Error Handling Principle
