# Quantization Format Integration Guide

## 1. Applicable Scope

This process is intended for external developers and maintainers who need to integrate a **new quantization write-to-disk format** into msModelSlim. It uses `compressed_tensors` as a complete 1-shot example to explain how to implement format export based on the `IFormat` protocol and how to enable it through YAML configuration.

Not applicable to: users who only consume existing formats (read the *[Quantization Format](README.md)* map first and then jump to the corresponding usage guide).

## 2. Process Relationships and Prerequisites

**Parent process**: after the open-source contribution / new format requirement review is approved.

**Prerequisites**:

- You have read *[Quantization Format](README.md)* and confirmed that a new format must be added (rather than extending an existing handler).
- The loading conventions of the target inference framework (tensor naming and metadata schema) have been clarified.

**Follow-up operations**: complete the documentation after the code is merged: write the format entry and usage guide according to the *[Documentation Standards](../../contributing/development_guide/docs_standards/README.md)* and *[Adding Quantization Format Documentation](../../../../skills/docs-management/scenarios/add-quantization-format.md)*, and update the format map in *[Quantization Format](README.md)*.

## 3. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Target format specification | Upstream project or internal RFC | Tensor key names, dtype, and metadata fields can be verified | Cross-check against the inference-side loading documentation |
| Input | List of QIR module types | `msmodelslim/ir` | FakeQuant types for which handlers must be implemented | List the module class names |
| Deliverable | IFormat implementation and Config | `msmodelslim/format/<name>/` | The Config is added to `QuantFormatConfigUnion` | Unit tests pass |
| Deliverable | `type` that can be enabled in YAML | One-click quantization `spec.save` | Consistent with the Config `Literal` | Configuration deserialization succeeds |
| Deliverable | Unit tests | `test/cases/format/` | Cover tensor keys and metadata | CI / local assertions pass |

## 4. Process Overview

The `IFormat` protocol defines a three-stage export process:

```mermaid
flowchart LR
  prepareExport["prepare_export"] --> traverse["process_module_tensors"]
  traverse --> finalize["finalize_export"]
```

## 5. Procedure

### Step 1: Adapt the Protocol and Base Classes

**Objective**: Understand the responsibility boundaries of `IFormat` / `ExportContext` / `QuantFormatBase`, and use them to complete the protocol alignment and base class selection required for adapting a new format.

**Operations**: Read [`msmodelslim/format/interface.py`](../../../../msmodelslim/format/interface.py) and [`msmodelslim/format/base.py`](../../../../msmodelslim/format/base.py).

| Method | Whether It Must Be Implemented | Responsibility |
| --- | --- | --- |
| `prepare_export()` | No (empty implementation by default) | Makes preparations before quantization. |
| `process_module_tensors(prefix, module)` | **Yes** | Exports the quantized tensors and quantization description information within the module subtree. |
| `finalize_export(model)` | **Yes** | Finalization: closes the writer and writes the full-model metadata. |

`ExportContext` fields: `save_directory`, `source_model_path`, `rank` / `world_size`.

Inheriting `QuantFormatBase` automatically provides module tree traversal, `WrapperIR` handling, and handler mapping dispatch. Subclasses must implement:

```python
def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
    """Mapping table from module type to the persist-to-disk handler."""
    ...

def on_float_module(self, prefix: str, module: nn.Module) -> None:
    """Fallback for unquantized modules: write the original parameters to the target format."""
    ...
```

**Output**: The selected base class and the list of QIR types to be supported.

### Step 2: Define the Config Class

**Operations**: Inherit `QuantFormatConfig` and set a unique `type` Literal:

```python
from typing import Literal
from msmodelslim.format.base import QuantFormatConfig

class MyQuantFormatConfig(QuantFormatConfig):
    type: Literal["my_format"] = "my_format"
    part_file_size: int = 4
```

Reference: [`CompressedTensorsQuantFormatConfig`](../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

**Output**: The Config class source file.

### Step 3: Implement the IFormat Subclass

**Operations**: Implement the handler and the three-stage lifecycle. Minimal example:

```python
from typing import Dict, Type

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.base import QuantFormatBase, ModuleHandler


class MyQuantFormat(QuantFormatBase):
    def prepare_export(self) -> None:
        self.safetensors_writer = ...

    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def finalize_export(self, model: nn.Module) -> None:
        try:
            ...
        finally:
            if self.safetensors_writer is not None:
                self.safetensors_writer.close()
                self.safetensors_writer = None

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear) -> None:
        self.safetensors_writer.write(prefix + ".weight", module.weight.to(torch.int8))
        self.safetensors_writer.write(prefix + ".weight_scale", module.weight_scale.unsqueeze(1))
        if module.bias is not None:
            self.safetensors_writer.write(prefix + ".bias", module.bias)
        self.safetensors_writer.write(prefix + ".input_scale", module.input_scale.to(torch.float32))

    def on_float_linear(self, prefix: str, module: nn.Linear) -> None:
        return self.on_float_module(prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.safetensors_writer.write(name, param.detach())
```

Reference: [`CompressedTensorsQuantFormat`](../../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

**Key Points for Writing Handlers**:

- Each QIR quantized module type must correspond to one handler.
- `WrapperIR`: for non-atomic wrappers, process the wrapped module first and then the wrapper; for atomic wrappers, process only the wrapper.
- Unregistered module types default to `on_float_module()`.
- It is recommended to scan the QIR in `finalize_export()` and derive the metadata by reverse inference (such as the `quantization_config` of compressed-tensors).

**Output**: The Format implementation class.

### Step 4: Register the Format Binding and the YAML Union Type

**Objective**: Enable the YAML `spec.save[].type` to be deserialized into the new Config based on `type`.

**Operations**: Import the new Config in [msmodelslim/format/registry.py](../../../../msmodelslim/format/registry.py) and add it to `QuantFormatConfigUnion`:

```python
from msmodelslim.format.my_format.my_format import MyQuantFormatConfig

QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
        MyQuantFormatConfig,  # Newly added
    ],
    Field(discriminator="type"),
]
```

After it is added, `parse_format_config` and the one-click quantization `spec.save` can recognize `type: "my_format"`.

**Output**: A configuration binding that can be deserialized by Pydantic based on `type`.

### Step 5: Configuration Enabling, Testing, and Documentation Synchronization

**Operations**:

1. Enable in YAML:

   ```yaml
   spec:
     save:
       - type: "my_format"
         part_file_size: 4
   ```

2. Refer to [`test/cases/format/compressed_tensors_format/`](../../../../test/cases/format/compressed_tensors_format): implement a mock writer, construct a minimal QIR model, invoke the three-stage lifecycle, and assert the key names / dtype / metadata.

3. Add the entry and usage guide under `docs/en/knowledge_base/quantization_format/<format_name>/` according to the documentation standards, and register the link in the *[Quantization Format](README.md)* map.

**Output**: Export can be started through the one-click quantization YAML (`spec.save.type`), the unit tests pass, and the documentation map is navigable.

**Pass conditions**: The configuration can be parsed; the unit test assertions pass; the map contains the entry and usage guide links for the new format.

## 6. Acceptance Conditions

- Export performed only by an external script and not registered with the msModelSlim configuration and factory is not considered a completed integration.
- A missing handler for a key QIR type or a metadata write failure must not be merged.
- If the documentation is not registered in the format map, the documentation acceptance fails.

## 7. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Quantization format | Write-to-disk and loading protocol | *[Quantization Format](README.md)* |
| compressed-tensors | 1-shot reference format | *[compressed-tensors](compressed_tensors/term_compressed_tensors.md)* |
| AscendV1 | Ascend default format (for comparison) | *[AscendV1](ascendv1/term_ascendv1.md)* |

## 8. Interface Documentation List

| Interface or Capability | Brief Description | Link |
| --- | --- | --- |
| `IFormat` | Quantization write-to-disk format protocol. Three interfaces must be implemented: `prepare_export`, `process_module_tensors`, and `finalize_export` | [Interface Definition](../../../../msmodelslim/format/interface.py) |
| `QuantFormatFactory` / registry | `QuantFormatConfigUnion` deserialization and format factory construction | [Registration and Factory](../../../../msmodelslim/format/registry.py) |
