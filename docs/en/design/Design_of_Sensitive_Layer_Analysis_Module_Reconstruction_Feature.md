# **msModelSlim Sensitive Layer Analysis Module Reconstruction Design Specifications**

|                                           |          |
| ----------------------------------------- | -------- |
| SIG group:                                | msit     |
| Incorporated into the following versions: | 26.0.0   |
| Designer:                                 | libowen  |
| Date:                                     | 20260122 |

**Copyright © 2026 msModelSlim Community**

Your reproduction, use, modification and distribution of this document is subject to the Creative Commons Attribution-ShareAlike 4.0 International Public License ("CC BY-SA 4.0"). For ease of understanding, you can visithttps://creativecommons.org/licenses/by-sa/4.0/Understand the overview (but not the replacement) of CC BY-SA 4.0. You can obtain the complete CC BY-SA 4.0 agreement from the following website:https://creativecommons.org/licenses/by-sa/4.0/legalcode.

**Revision records**

| Date     | Revised version | Revision Description | Authors | Audited   |
| -------- | --------------- | -------------------- | ------- | --------- |
| 20260122 | 1.0.0           | Document Creation    | libowen | panyj1993 |

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

     2.3.4 Impact on System Performance Specifications

     2.3.5 Analysis of Impact on System Reliability Specifications

     2.3.6 Impact Analysis on System Compatibility

     2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

     2.4 Analysis on the Implementation Solution of Similar Community/Commercial Software

3. Feature/Function Implementation Principles

     3.1 Objectives

     3.2 Overall Solution

4. Use Case Implementation

     4.1 Use Case Description

     4.2 Feature Design Ideas

     4.3 Constraints

     4.4 Detailed implementation (module-level or process-level message sequence diagram from user entry)

     4.5 Interfaces Between Subsystems (Mainly Covering the Interface Definition of Modules)

     4.6 Detailed Design of Subsystems

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

5. Reliability and availability design

     5.1 Redundancy Design

     5.2 Fault Management

     5.3 Overload control design

     5.4 Upgrade Without Service Interruption

     5.5 Human Error Design

     5.6 Fault Prediction and Prevention Design

6. Design for features and non-functional quality attributes

     6.1 Testability

     6.2 Serviceability

     6.3 Evolvability

     6.4 Openness

     6.5 Compatibility

     6.6 Scalability/Scalability

     6.7 Maintainability

     6.8 Information

7. (Optional) Data Structure Design

8. List of references

**Table Catalogue**

Table 1 List of feature requirements

Table 2: Security Design Qualification Form

Table 3: List of modified documents

**Figure Catalogue**

Figure 1: Overall implementation principle

**List of abbreviations:**

| Abbreviations Abbreviations | Full spelling                | Chinese explanation Chinese explanation |
| --------------------------- | ---------------------------- | --------------------------------------- |
| MHA                         | Multi-Head Attention         | multiheaded attention mechanism         |
| MLA                         | Multi-Head Latent Attention  | Potential attention mechanism of bulls  |
| DSA                         | Distributed Sparse Attention | distributed sparse attention mechanism  |
| SWA                         | Sliding Window Attention     | Sliding window attention mechanism      |
| NPU                         | Neural Processing Unit       | neural network processing unit          |
| YAML                        | YAML Ain't Markup Language   | YAML Markup Language                    |
| MD5                         | Message Digest Algorithm 5   | Message Digest Algorithm 5              |

## 1. Feature Overview

Sensitive layer analysis is used to identify key layers and structures in the model quantization process, helping users understand quantitative sensitivity and formulate optimization policies. Currently, an independent analysis service at the sensitive layer exists, and the quantitative service has a unified scheduling and execution mechanism. This feature aims to upgrade the analysis capability of the sensitive layer from independent implementation to a scheduleable, reusable, and extensible service-oriented capability.

This reconstruction brings the following benefits: (1) A unified scheduling entry reduces maintenance costs. (2) Align with the quantification process to ensure consistent context and configuration capabilities for sensitivity analysis. 3) Provides standardized carriers for subsequent algorithm extension.

This document describes the design intent, overall solution, and application scenarios of the analysis module reconstruction at the sensitive layer. It focuses on the scheduling and algorithm-based abstraction capabilities. It is intended for R&D, test, and maintenance personnel of the msModelSlim tool.

### 1.1 Scope

This feature is reconstructed based on the analysis module at the sensitive layer and provides the following functions:

1. **Scheduling reconstruction: Reuse the scheduling mechanism of quantitative services and unify the execution orchestration of sensitive layer analysis.**
2. **Process alignment: Sensitivity analysis supports layer-by-layer scheduling and context management consistent with the quantitative process.**
3. **Algorithm abstraction: A pluggable processor and sensitivity index framework is introduced to support sensitivity analysis of linear layers and attention structures.**

**Note: The existing sensitive layer analysis function remains unchanged. Reconstruction is an implementation upgrade. The details about the analysis algorithm are not described in this feature.**

### 1.2 Feature Requirement List

Table 1 List of feature requirements

| Requirement No. | Requirement name                                      | Feature Description                                                                                                                              | Remarks                |
| --------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| 1               | Analysis and scheduling at the sensitive layer        | Reuse the quantitative service scheduling mechanism to implement unified entry and execution orchestration for sensitive layer analysis.         | Planned implementation |
| 2               | Alignment between analysis process and quantification | Supports layer-by-layer scheduling and quantitative context awareness, enabling sensitivity analysis to be embedded in the quantitative process. | Planned implementation |
| 3               | Sensitivity algorithm framework                       | Supports the linear layer, attention structure, and multi-indicator sensitivity calculation framework extension.                                 | Planned implementation |

## 2. Requirement Scenario Analysis

### 2.1 Feature Requirement Source and Value Overview

Currently, the sensitive layer analysis has an independent service implementation, but the scheduling system and process are separated from the quantitative service. As the quantification capability continues to evolve, sensitive layer analysis needs to be more closely integrated into the quantification process to support unified scheduling, configuration, and expansion. The reconstructed sensitive layer analysis will be upgraded from independent implementation to unified capability of combining scheduling and algorithms, reducing repeated construction and improving collaboration efficiency.

### 2.2 Feature Scenario Analysis

#### Scenario Trigger Conditions and Objects

1. **Trigger conditions:**
    
     * Users need to identify sensitive layers before or during quantization.
     * Users want to compare sensitivity differences between different layers or structures
     * The user expects the sensitivity analysis to be performed in line with the quantification process.
2. **Intended:**
    
     * Model quantification engineer: Focus on the quantification solution and sensitive layer positioning.
     * Algorithm Engineer: Focus on Sensitivity Indicators and Explainability
3. **Use the following interface:**
    
     * Command line interface: consistent entry with the quantification tool
     * Configuration file: unified YAML configuration mode

#### Main Application Scenarios

1. **Pre-quantification sensitivity assessment scenario:**
    
     * Perform sensitive layer analysis before quantization to form a candidate layer list.
     * Focus on hierarchical sorting and overall trend
2. **Analysis layer by layer in quantization:**
    
     * Scheduling sensitivity assessment in a quantization process by layer
     * Focus on process consistency and scheduling stability.
3. **Algorithm iteration evaluation scenario:**
    
     * Comparison Between Different Sensitivity Indicators or Processors
     * Focus on scalability and consistency of results

### 2.3 Feature Impact Analysis

Reconstructing the analysis module at the sensitive layer interacts with the following modules:

1. **Analysis service module at the sensitive layer: The module is implemented as an independent service. After reconstruction, the module shares scheduling with the quantization service.**
2. **Quantification service module: reuses the scheduling and execution framework and provides the process context.**
3. **Processor framework module: as the carrier and extension carrier of sensitivity algorithms**
4. **Configuration and metadata module: unified configuration entry and parameter management**
5. **Log and result management module: unified output and traceability**

#### Interaction Analysis with Other Requirements and Features

1. **Interaction with the quantization feature: The sensitive layer analysis depends on the scheduling mechanism of the quantization service. The interfaces and configurations must be consistent.**
2. **Interaction with the evaluation feature: The sensitivity result may be referenced by the evaluation process, and the data format must be aligned with the result structure.**
3. **Interaction with the model adapter: The model adapter is required to provide structural information and hierarchical description.**

#### Platform Difference Analysis

1. **Hardware platform: depends on the quantitative execution capability of the NPU environment.**
2. **Operating system: supports the Linux operating system and requires Python 3.8+.**

#### Compatibility Analysis

1. **Configuration compatibility: The new interface is compatible with the existing sensitive layer analysis configuration format.**
2. **Interface compatibility: Retain the compatibility layer of the original invoking mode.**

#### Constraints and Limitations

1. **Model support restrictions: Only the model types and structure descriptions that have been adapted are supported.**
2. **Process coupling restrictions: Some sensitivity algorithms depend on the quantization context.**

#### 2.3.1 Hardware Limitations

1. **NPU device requirements: NPU devices that support model quantization and inference are required.**
2. **Memory requirements: Sensitivity analysis requires extra memory overhead. It is recommended that at least 32 GB memory be used.**
3. **Storage requirements: Analysis results and intermediate data must be stored. It is recommended that the available space be at least 50 GB.**

**Workaround:**

 * If resources are insufficient, reduce the number of analysis layers or the analysis frequency.

#### 2.3.2 Technical Limitations

**Operating system: Linux**

**Programming language: Python 3.8+**

**Dependency framework:**

 * PyTorch: Model Loading and Quantification Dependency
 * Quantification service component: provides scheduling and execution environment.

**Workaround:**

 * If the dependency version is incompatible, refer to the installation guide to use the specified version.

#### 2.3.3 Impact Analysis on the License

This feature does not introduce new third-party components and uses the existing dependency system. This feature does not affect the license compliance.

#### 2.3.4 Impact on System Performance Specifications

The performance overhead of sensitive layer analysis varies with the analysis granularity and indicator type, and is controllable. After reconstruction, the scheduling and parallel mechanism can be used to reduce the overall time required for a single analysis.

#### 2.3.5 Analysis of Impact on System Reliability Specifications

After reconstruction, analysis at the sensitive layer is executed consistently with the quantification process. Scheduling failures can be isolated and analysis results can be reentrant. The reliability of the main quantification process is not affected.

#### 2.3.6 Impact Analysis on System Compatibility

The new implementation is compatible with the existing configuration and invoking mode, and does not affect the usage of existing users.

#### 2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

1. **Interaction with the quantization process: Shared scheduling and context, and consistent interfaces**
2. **Interaction with automatic optimization: Sensitivity results may be referenced by optimization policies. A unified data structure is required.**

### 2.4 Analysis of implementation solutions for similar community/commercial software

Common implementation modes include independent sensitivity analysis module and quantitative process embedded analysis. The independent solution is easy to implement, but it is difficult to reuse the scheduling and flow. The embedded solution is more consistent but has higher requirements on the framework. This feature chooses to implement sensitivity analysis within the quantitative service scheduling framework to balance consistency and scalability.

## 3. Feature/Function Implementation Principles

### 3.1 Objectives

The objectives of the reconstruction of the analysis module at the sensitive layer are as follows:

1. **Unified scheduling: The sensitive layer analysis has the same scheduling entry and execution mechanism as the quantitative service.**
2. **Process alignment: Sensitivity analysis can be integrated into the quantitative process to support layer-by-layer scheduling.**
3. **Algorithm extension: The sensitivity algorithm is extended in processor and indicator mode and supports multi-structure analysis.**
4. **Compatibility assurance: External compatibility and internal replacement**

### 3.2 Overall Solution

The overall solution is to reconstruct the sensitive layer analysis service by reusing the scheduling mechanism of quantitative services. The key points are as follows:

1. **Scheduling reuse: Add the sensitivity analysis scheduling process within the quantitative service framework.**
2. **Processor abstraction: The sensitivity algorithm is organized in the form of a processor and supports the linear layer and attention structure.**
3. **Unified indicator: A unified indicator interface is used for sensitivity scoring and can be expanded by combination.**

#### Hardware selection

 * **NPU: NPU-based unified execution of quantification and analysis**

#### Algorithm selection

 * **Sensitivity indicator: Uses a pluggable indicator framework and supports multiple sensitivity evaluation methods.**
 * **Structure adaptation: provides general adaptation capabilities for linear layers and attention structures.**

#### Architecture Layout

Analyze the sensitive layer and align the architecture with the hierarchical design of quantitative services.

1. **Application layer: provides unified entry and execution orchestration.**
2. **Scheduling layer: responsible for layer-by-layer scheduling and context management.**
3. **Algorithm layer: processor and indicator sensitivity analysis**
4. **Data layer: unified management of results and metadata**

#### Use Case Breakdown

1. **Use Case: Sensitive Layer Analysis Based on Scheduling and Algorithms**

#### Interconnection Principles

1. **Interface standardization: The scheduling interface is consistent with the quantitative service.**
2. **Unified data format: A unified structure is used for sensitivity results.**
3. **Error handling specifications: consistent with the quantitative service log and exception handling specifications.**

#### Overall solution architecture

```text
flowchart TD
    U[User command line interface msmodelslim analyze] --> A[AnalysisApplication loading configuration/initializing context/organizing analysis process]
    A --> Q[Quant Service scheduling capability]

    subgraph QS[Quant Service Existing scheduling capability]
        S[Scheduler]
        R[Runner]
        P[Processor mechanism]
        S --> R --> P
    end

    Q --> S

    P --> AP[sensitivity analysis Processor]
    AP --> M1[Metric A]
    AP --> M2[Metric B]
    AP --> M3[Metric C]
```

Figure 1: Overall implementation principle

## 4. Use case implementation

### 4.1 Use Case Description

**Use Case Name: Sensitive Layer Analysis Function Implemented Based on Scheduling and Algorithms**

**Use case scenario:**

 * Users want to use a unified entry to trigger sensitive layer analysis.
 * The system reuses the quantization service scheduling mechanism for execution.
 * Consistent context for the analysis process and the quantification process
 * Users need to evaluate the sensitivity of the linear layer and attention structure.
 * Users want to use multiple metrics to evaluate sensitivity results.

**Impact on the analysis function at the sensitive layer:**

 * A unified scheduling entry is required.
 * The analysis process needs to be modified in scheduling mode.
 * Structure adaptation and indicator expansion capabilities are required.

**Implemented feature: unified analysis at the sensitive layer**

### 4.2 Feature Design Ideas

By introducing the scheduling layer and unified entry, the sensitive layer analysis is incorporated into the scheduling system of quantitative services, avoiding independent process repetition and configuration splitting.

### 4.3 Constraints

1. **Configuration consistency requirements: Analyze the structure and parsing mode of configuration and quantitative configuration.**
2. **Scheduling dependency: The analysis process depends on the scheduling component of the quantization service.**
3. **Compatibility requirements: Retain the compatibility logic of the original analysis entry.**

### 4.4 Detailed implementation (module-level or process-level message sequence diagram from user entry)

#### Handling Procedure

```tex
User initiates sensitivity layer analysis
    │
    ▼
AnalysisApplication.run()
    │
    ▼
Scheduling layer initializes context
    │
    ├─→ Load configuration
    ├─→ Parse analysis tasks
    └─→ Register processors and metrics
    │
    ▼
Perform sensitivity analysis layer by layer
    │
    ├─→ Identify layer type and structure
    ├─→ Select and execute sensitivity processor
    ├─→ Calculate sensitivity scores for metric subclasses
    └─→ Aggregate results and output them
```

#### Module Interaction Description

1. **Analysis Application: Unified Entry and Process Orchestration**
2. **Scheduling framework: Reuse the quantitative service scheduling capability.**
3. **Sensitivity processor: executes the analysis algorithm.**
4. **Metric module: Calculate and summarize sensitivity results**

### 4.5 Interfaces Between Subsystems (Mainly Covering the Interface Definition of Modules)

#### New Interface

1. **AnalysisDispatchConfig:**
    
     * Type: Configuration Model
     * Run the following command to define the analysis scheduling parameters at the sensitive layer:
2. **SensitivityProcessor:**
    
     * Type: Abstract interface
     * Function: Defines the unified entry for the sensitivity analysis processor.
3. **SensitivityMetric:**
    
     * Type: Abstract interface
     * Function: Defines the input and output of sensitivity indicators.

#### Modifying an Interface

1. **Quantized scheduling entry:**
    
     * Function extension: Supports registration of analysis scheduling tasks at the sensitive layer.

### 4.6 Detailed Design of Subsystems

#### 4.6.1 Dispatching Adaptation Design

Scheduling adaptation incorporates sensitive layer analysis into the scheduling life cycle of quantitative services through unified configuration and context transfer, avoiding independent maintenance of execution logic.

#### 4.6.2 Processor Organization

Processors are organized by layer type and structure type, and can be combined and executed. They can be expanded and replaced on demand.

#### 4.6.3 Structure Adaptation and Indicator Expansion

Through the structure description object and index registration mechanism, the sensitivity analysis and multi-indicator scoring of linear layer and attention structure are supported.

### 4.7 DFX Attribute Design

#### 4.7.1 Performance Design

1. **Scheduling overhead: The scheduling overhead is controllable and does not affect the original quantization process.**
2. **Analysis overhead: The analysis granularity is configurable, and the overhead can be reduced on demand.**

#### 4.7.2 Upgrade and Capacity Expansion Design

1. **Configuration compatibility: New configurations are compatible with old interfaces.**
2. **Scalability: The processor and metrics can be expanded.**

#### 4.7.3 Exception Handling Design

1. **Scheduling failure processing: Logs are recorded and single-layer failures are isolated.**
2. **Troubleshooting: The error does not affect the main process.**

#### 4.7.4 Resource Management Design

The quantitative service resource management mechanism is reused during analysis to ensure consistency between resource application and release.

#### 4.7.5 Miniaturized Design

In the miniaturized version, the analysis and scheduling at the sensitive layer can be disabled to reduce additional resource consumption.

#### 4.7.6 Testability Design

The test focuses on basic capabilities such as scheduling entry, processor loading, indicator calculation, and result summary.

#### 4.7.7 Security Design

The sensitive layer analysis does not involve new external interfaces or sensitive data storage. The security design uses the existing policies.

### 4.8 External Interfaces

The method of invoking external interfaces is the same as that of the quantization service. A configuration item is added to control the analysis execution at the sensitive layer.

### 4.9 Self-Test Case Design

1. **Entry compatibility test: Analysis can be triggered by both the old and new entries.**
2. **Scheduling process test: layer-by-layer scheduling can be stably executed.**
3. **Exception isolation test: Single-layer exceptions do not affect the overall process.**

## 5. Reliability and availability design

### 5.1 Redundancy Design

Analysis and reconstruction at the sensitive layer adopts the unified scheduling and result output mechanism. Based on the configuration and log redundancy capabilities of the quantitative service, analysis tasks can be traced.

### 5.2 Fault Management

#### Fault detection

1. **Scheduling failure detection: Scheduling failures are recorded in logs and marked with failure layers.**
2. **Processor exception detection: Exception isolation does not affect subsequent layers.**

#### Fault isolation

1. **Hierarchical isolation: Single-layer failure does not affect the overall analysis process.**
2. **Module isolation: KPI failures do not affect the calculation of other KPIs.**

#### Fault recovery

1. **Retry mechanism: configurable hierarchical retry policy**
2. **Result rollback: supports default rollback of missing results.**

### 5.3 Overload control design

1. **Task Traffic Limit: Limit the number of concurrent analysis tasks.**
2. **Granularity control: Supports on-demand analysis granularity reduction.**

### 5.4 Upgrade Without Service Interruption

1. **Configuration compatibility: The configuration format is compatible after reconstruction.**
2. **Interface compatibility: The original invoking mode is retained.**

### 5.5 Design for human-caused errors

1. **Configuration verification: configuration parameter verification and error prompt**
2. **Log prompt: Key steps are recorded in logs for fault locating.**

### 5.6 Fault Prediction and Prevention Design

1. **Resource monitoring: Monitors memory and storage usage.**
2. **Exception warning: Key exceptions can trigger warnings.**

## 6. Design for features and non-functional quality attributes

### 6.1 Testability

*This document describes the test direction and specifications of the feature, and describes the aspects that should be tested, boundary values, abnormal values, and abnormal scenarios that should be noted by the test personnel.*

### 6.2 Serviceability

*Provides various maintainable and serviceable measures for features, and provides complete documentation for feature usage, maintenance, and troubleshooting.*

### 6.3 Evolvability

*Focus on the evolvability of the feature architecture and functions.*

### 6.4 Openness

*Focus on the openness of external interfaces, including the standardization of the interfaces, for example, compliance with the SQL 2011 standard.*

### 6.5 Compatibility

*Focus on whether the feature affects the forward compatibility of the system, that is, whether the old functions are available after the upgrade and whether the usage behavior is consistent with that of the old version.*

### 6.6 Scalability/Scalability

*This feature effectively meets the requirements of system capacity changes, including the scaling of database nodes and database servers.*

### 6.7 Maintainability

*Focus on feature maintainability, such as diagnosis view and log printing.*

### 6.8 Information

*Refer to the following table to evaluate the modification points of various documents involved in the feature evaluation and describe the specific modification points.*

| Category                                                                                                                                                                  | Manual Name           | Involved or Not (Y/N)                                      | Description of the modified or added content |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| White Paper                                                                                                                                                               | Technical white paper | N                                                          | Added the XX technology in section XX.       |
| Product Documentation                                                                                                                                                     | Product Description   | Y                                                          | Updated the technical specifications to XX.  |
|-| Feature Description                                                                                                                                                       | Y                     | Added the XX feature.                                      |
|-| Compilation Guide                                                                                                                                                         | Y                     | XXX                                                        |
|-| Installation Guide                                                                                                                                                        | Y                     | Updated the XX scenario in section "Installing a Cluster." |
|-| Administrator's Guide                                                                                                                                                     | N                     | XXX                                                        |
|-| Developer guide (including the development tutorial, SQL reference, system tables and system views, GUC parameter description, error code description, and API reference) | Y                     | Added the XXX function in section XX.                      |
|-| Tool Reference                                                                                                                                                            | Y                     | Added the XX tool.                                         |
|-| Glossary of terms                                                                                                                                                         | Y                     | New term XX                                                |
| Getting Started                                                                                                                                                           | Easy tutorial         | N                                                          | XXX                                          |

## 7. (Optional) Data Structure Design

Analysis and reconstruction at the sensitive layer mainly uses the unified configuration and result structure, maintains the YAML expression consistent with the quantification service, and keeps the specific data structure abstract and extensible.

## 8. List of references
