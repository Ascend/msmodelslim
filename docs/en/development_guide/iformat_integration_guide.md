# Quantization Format Integration Guide

<!-- md-trans-meta sourceCommit=94328d0a283b5c0669a686d81f7208caa2010f8f translatedAt=2026-08-21T03:13:51.046Z pushedAt=2026-08-21T03:14:01.531Z -->

## Introduction

This document is intended for external developers who need to integrate a **new quantization format for persisting to disk** into msModelSlim. Using `compressed_tensors` as a complete 1-shot example, it explains how to implement format export based on the `IFormat` protocol and enable it through YAML configuration.

> For format selection, see *[Format Support Matrix](../user_guide/quantization_formats/README.md)*. Legacy formats such as AscendV1 and MindIE follow the Legacy Saver path and are outside the scope of this document.

## Export Lifecycle

The `IFormat` protocol defines a three-stage export process:

```mermaid
flowchart LR
  prepareExport["prepare_export()"] --> traverse["process_module_tensors()"]
  traverse --> finalize["finalize_export()"]
```

## IFormat Protocol Interface

Defined in [`msmodelslim/format/interface.py`](../../../msmodelslim/format/interface.py):

| Method | Whether It Must Be Implemented | Responsibility |
|------|-------------|------|
| `prepare_export()` | No (empty implementation by default) | Makes preparations before quantization. |
| `process_module_tensors(prefix, module)` | **Yes** | Exports the quantized tensors and quantization description information within the module subtree. |
| `finalize_export(model)` | **Yes** | Finalization: closes the writer and writes the full-model metadata. |

### ExportContext

The export runtime environment, injected by the framework when constructing an `IFormat` instance:

| Field | Description |
|------|------|
| `save_directory` | Output directory |
| `source_model_path` | Source model path (used to copy HF auxiliary files) |
| `rank` / `world_size` | Distributed rank information |

### QuantFormatBase (Recommended Base Class)

Inheriting [`QuantFormatBase`](../../../msmodelslim/format/base.py) automatically provides:

- Module tree traversal (`named_modules` + `processed_modules` deduplication)

- `WrapperIR` atomic/non-atomic handling

- Handler mapping dispatch

Subclasses must implement:

```python
def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
    """Mapping table from module type to the persist-to-disk handler."""
    ...

def on_float_module(self, prefix: str, module: nn.Module) -> None:
    """Fallback for unquantized modules: write the original parameters to the target format."""
    ...
```

For module types not registered in the handler map, the base class automatically invokes `on_float_module()`. You can also explicitly register `nn.Module: self.on_float_module` in the map as the fallback handler.

## Five-Step Integration Process

The following uses `compressed_tensors` as a one-shot complete example.

### Step 1: Defining the Config Class

Inherit from `QuantFormatConfig` and set a unique `type` Literal:

```python
from typing import Literal
from msmodelslim.format.base import QuantFormatConfig

class MyQuantFormatConfig(QuantFormatConfig):
    type: Literal["my_format"] = "my_format"
    part_file_size: int = 4
```

Reference: [`CompressedTensorsQuantFormatConfig`](../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

### Step 2: Implementing the IFormat Subclass

```python
from typing import Dict, Type

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.base import QuantFormatBase, ModuleHandler


class MyQuantFormat(QuantFormatBase):
    def prepare_export(self) -> None:
        # Create the safetensors writer, etc.
        self.safetensors_writer = ...

    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def finalize_export(self, model: nn.Module) -> None:
        # Write config.json and close the writer.
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

Reference: [`CompressedTensorsQuantFormat`](../../../msmodelslim/format/compressed_tensors_format/compressed_tensors.py)

### Step 3: Registering the Format Binding

Register it in [`msmodelslim/format/registry.py`](../../../msmodelslim/format/registry.py):

```python
class QuantFormatFactory:
    BUILTIN_BINDINGS = (
        (CompressedTensorsQuantFormatConfig, CompressedTensorsQuantFormat),
        (MyQuantFormatConfig, MyQuantFormat),  # // Add a new entry.
    )
```

Or invoke it at runtime:

```python
from msmodelslim.processor.save.registry import register_quant_format
register_quant_format(MyQuantFormatConfig, MyQuantFormat)
```

### Step 4: Adding to the YAML Union Type

Add the new Config class to [`QuantFormatConfigUnion`](../../../msmodelslim/format/registry.py) so that Pydantic can deserialize it based on the `type` field:

```python
QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        MyQuantFormatConfig,  # Add
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
    ],
    Field(discriminator="type"),
]
```

When `import msmodelslim.format` is executed, `QuantFormatFactory.install()` is automatically invoked to complete the registration.

### Step 5: Enabling via YAML Configuration

```yaml
spec:
  save:
    - type: "my_format"
      part_file_size: 4
```

## Key Points for Writing a Handler

### QIR Module Mapping

Each QIR quantization module type must correspond to a handler that writes the module parameters into the target format:

```python
def build_module_handler_map(self):
    return {
        qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
        qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic,
        nn.Linear: self.on_float_linear,
        nn.Module: self.on_float_module,
    }
```

### WrapperIR Processing

`QuantFormatBase` automatically processes `WrapperIR`:

- **Non-atomicity** (`is_atomic() = False`): process the wrapped module first, then the wrapper.

- **Atomicity** (`is_atomic() = True`): process only the wrapper, skipping the wrapped module.

### Fallback for Unquantized Layers

For module types that are not registered in the handler map, `on_float_module()` is called by default, which iterates over `named_parameters` and directly writes the original parameters.

### Reverse Derivation of Metadata

It is recommended to scan the model QIR modules in `finalize_export()` and reverse-derive the format metadata (for example, compressed-tensors' `config.json` → `quantization_config`), rather than accumulating it layer by layer in the handler.

## Testing and Verification

Refer to [`test/cases/format/compressed_tensors_format/`](../../../test/cases/format/compressed_tensors_format):

1. Implement the `MockSafetensorsWriter` in-memory writer.

2. Construct a minimal QIR model (W8A8 Static / Dynamic).

3. Invoke `prepare_export()` → `process_module_tensors()` → `finalize_export().`

4. Assert the safetensors tensor key names, dtype, shape, and config metadata.

## Complete Reference Implementation

| Component | Path |
|------|------|
| IFormat protocol | [`msmodelslim/format/interface.py`](../../../msmodelslim/format/interface.py) |
| QuantFormatBase | [`msmodelslim/format/base.py`](../../../msmodelslim/format/base.py) |
| Registry | [`msmodelslim/format/registry.py`](../../../msmodelslim/format/registry.py) |
| Save processor | [`msmodelslim/processor/save/processor.py`](../../../msmodelslim/processor/save/processor.py) |
| compressed-tensors implementation | [`msmodelslim/format/compressed_tensors_format/`](../../../msmodelslim/format/compressed_tensors_format) |
| Unit tests | [`test/cases/format/compressed_tensors_format/`](../../../test/cases/format/compressed_tensors_format) |

## Related Documents

- *[Format Support Matrix](../user_guide/quantization_formats/README.md)*

- *[compressed-tensors Format Description](../user_guide/quantization_formats/compressed_tensors.md)* — the target format of the one-shot reference implementation

- *[AscendV1 Format Description](../user_guide/quantization_formats/ascendv1.md)* — the Legacy format comparison reference
 