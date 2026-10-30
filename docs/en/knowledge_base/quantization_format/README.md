# Quantization Format Terminology Entry

> **Entry Category**: Quantization Data Format<br>
> **English Name**: Quantization Format<br>
> **Application Domain**: Large language model quantization and compression, multimodal understanding and generation model quantization and compression, inference framework weight loading<br>
> **msModelSlim Implementation**: [`msmodelslim/format/`](../../../../msmodelslim/format/)

---

## 1. Overview

[Quantization Format Terminology Entry](./README.md) is the **write-to-disk and load protocol** between the quantization tool and the inference framework: it defines the file structure, tensor naming, and metadata organization of quantized weights so that the target inference engine can dequantize correctly and complete inference. It solves the problem of how the quantization parameters produced by algorithms are consumed by downstream deployment.

---

## 2. Term Description

### 2.1 Format Map

The map is for **selection and navigation**: first choose a format by the target inference stack and model type, then go to the corresponding entry to check the modes and deliverables, or go to the usage guide to confirm mode support, configure, and run. For specific scenarios the entry and usage guide of each format prevail; this table does not elaborate.

| Format | Typical Use Cases | Target Inference Framework | Mode Summary and Recommendations | Distributed Export | Entry | Usage Guide |
| --- | --- | --- | --- | --- | --- | --- |
| AscendV1 | General deployment on the Ascend side, such as LLM and multimodal understanding | vLLM Ascend, SGLang, MindIE | Covers 20+ modes such as W8A8 / W8A16 / W4A4 / MXFP / KV Cache / FA; **Atlas A2 products and Atlas A3 products should prefer W8A8 (static/dynamic)**, choosing W4 only when memory is tighter; **Ascend 950PR&950DT products should prefer MXFP (W8A8_MXFP8 / W4A4_MXFP4, etc.)** | Supported | *[AscendV1 Quantization Format Terminology Entry](ascendv1/term_ascendv1.md)* | *[AscendV1 Usage Guide](ascendv1/ascendv1_usage.md)* |
| compressed-tensors | Quantized weights need to be loaded in HF-ecosystem frameworks such as vLLM, or need to align with the `quantization_config` of the compressed-tensors specification | The HF ecosystem, such as vLLM | Currently only **W8A8 Static / W8A8 Dynamic**; choose Static when the activation scale has been calibrated offline, and Dynamic when calibration-free operation or adaptation to dynamic inputs is required; KV Cache is not supported | Not supported | *[compressed-tensors Quantization Format Terminology Entry](compressed_tensors/term_compressed_tensors.md)* | *[compressed-tensors Usage Guide](compressed_tensors/compressed_tensors_usage.md)* |
| MindIE SD | Delivery of multimodal **generation** models such as diffusion / DiT to MindIE | MindIE (multimodal generation) | A subset of the enumerations for the generation pipeline (including W8A8, MXFP, FAQuant, and so on); **select according to the model best-practice `quant_type`** rather than applying the LLM default AscendV1 path | Supported | *[MindIE SD Quantization Format Terminology Entry](mindie_sd/term_mindie_sd.md)* | *[MindIE SD Usage Guide](mindie_sd/mindie_sd_usage.md)* |

---

## 3. How the Three Formats Differ

| Dimension | AscendV1 | compressed-tensors | MindIE SD |
| --- | --- | --- | --- |
| Positioning | The default quantization write-to-disk protocol for Ascend inference | A quantization format compatible with the HF ecosystem | The dedicated MindIE write-to-disk protocol for multimodal generation |
| Model Scope | Large language models and multimodal understanding models | Linear layer quantization, HF-style weights | Generation models such as diffusion / DiT (for example Wan2.2) |
| Metadata Form | A standalone `quant_model_description.json` + `quant_model_weights*.safetensors` | A `quantization_config` injected into `config.json` + `model*.safetensors` | `quant_model_description*.json` + `quant_model_weight*.safetensors` (the weight file name uses the singular "weight") |
| Saver `type` | `ascendv1_saver` | `compressed_tensors` | `mindie_format_saver` |
| Key Advantage | The finest description JSON granularity: the quantization type is annotated layer by layer, supporting mixed quantization (different layers of the same model can use different modes); the most complete mode enumerations, covering 20+ quantization combinations | Naturally compatible with the HF `from_pretrained` / vLLM automatic-detection path; no extra installation of the `compressed-tensors` package is needed, giving zero-barrier access to the HF ecosystem | Deeply aligned with the MindIE multimodal generation pipeline; supports generation-specific metadata such as `fa_quant_type`, and `dump_config` can capture calibration data |

The three **cannot be loaded interchangeably**: changing the inference stack or the model category usually requires re-exporting. For mode enumerations, field-level deliverables, and limitations see the entries in the table above; for configuration and execution steps see the corresponding usage guide.

To integrate a new format, follow the *[Quantization Format Integration Guide](iformat_integration_guide.md)* and the *[msModelSlim Documentation Standards](../../contributing/development_guide/docs_standards/README.md)*.

---

## 4. Related Workflows

- *[Quantization Format Integration Guide](iformat_integration_guide.md)*: integrate a new write-to-disk format based on IFormat.
- *[AscendV1 Usage Guide](ascendv1/ascendv1_usage.md)*: confirm mode support, configure, and run for AscendV1.
- *[compressed-tensors Usage Guide](compressed_tensors/compressed_tensors_usage.md)*: confirm mode support, configure, and run for compressed-tensors.
- *[MindIE SD Usage Guide](mindie_sd/mindie_sd_usage.md)*: confirm mode support, configure, and run for MindIE SD.
- *[One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md)*: an overview of the one-click quantization commands and configuration protocol.

---

## 5. Related Terms

- [AscendV1 Quantization Format Terminology Entry](ascendv1/term_ascendv1.md): a subordinate concept; the default quantization write-to-disk format on the Ascend side.
- [compressed-tensors Quantization Format Terminology Entry](compressed_tensors/term_compressed_tensors.md): a subordinate concept; a quantization write-to-disk format compatible with the HuggingFace / vLLM ecosystem.
- [MindIE SD Quantization Format Terminology Entry](mindie_sd/term_mindie_sd.md): a subordinate concept; the MindIE write-to-disk format for multimodal generation scenarios.
- [Quantization Algorithms](../quantization_algorithms/README.md): other; the algorithm family on the quantization computing and calibration side, whose output is written to disk by each quantization format.
- [Quantization Modes](../quantization_mode/README.md): a companion term; format enumerations and deliverable fields correspond to the quantization modes.
- [Foundation Model Support Matrix](../model/README.md): other; the selection and validation reference for model x mode x inference stack.
