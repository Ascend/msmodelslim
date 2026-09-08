# Model Structure Extraction Feature Design Specification

## Revision Record

| Date | Revision Version | Description | Author | RFC Document |
| -- | -- | -- | -- | -- |
| 2026-05-25 | 1.0 | Initial draft completed | @tongl | https://gitcode.com/Ascend/msmodelslim/issues/228 |

## Background

fast_ops_grapher is a foundational module for extracting computation graphs from PyTorch models. It provides operator-level computation graph visualization and analysis capabilities. The module supports graph extraction and formatted output for multiple model types.

## Solution Design

For details, see the [RFC document](https://gitcode.com/Ascend/msmodelslim/issues/228).

### Overall Architecture

The fast_ops_grapher module consists of the following parts:

1. **Extractors**: Extract computation graphs from different types of models
2. **Exec Observer**: A runtime observer that captures operator calls and builds computation graphs
3. **Formatters**: Format computation graphs into different output formats
4. **Computation Graph Structure**: Defines core data structures such as ComputationGraph, GraphNode, and GraphEdge

### Core Class Relationships

- `BaseExtractor`: The abstract base class for all extractors
- `NativeModuleExtractor`: Extracts computation graphs from any PyTorch nn.Module
- `TransformerExtractor`: Extracts computation graphs from HuggingFace Transformers models
- `TransformerAutoExtractor`: Automatically loads Transformers models and extracts computation graphs
- `ComputationGraph`: Inherits from networkx.DiGraph to manage computation graph nodes and edges
- `GraphNode`: Represents an operator node
- `GraphEdge`: Represents the Tensor data flow between nodes

### Dependency Selection

This solution directly depends on the networkx library. `ComputationGraph` inherits from `networkx.DiGraph` to manage the graph structure. Because networkx is a dependency of PyTorch, this selection is reasonable and does not require additional third-party dependencies.

## Usage Instructions

### Extractor API

The Extractor extracts computation graphs from PyTorch models and returns a `ComputationGraph` object. Three implementations are provided for different model types and usage scenarios.

#### Core Interfaces

- `create` (factory method): Each Extractor provides a `create` static method to create an instance
- `extract_dag`: The main method for extracting computation graphs, returns a `ComputationGraph` object

#### NativeModuleExtractor

Extracts computation graphs from any PyTorch `nn.Module`. This extractor applies to user-defined models or models that are not from the Transformers library.

```python
from msmodelslim.core.graph.fast_ops_grapher import NativeModuleExtractor

extractor = NativeModuleExtractor.create(
    module=my_model,
    args=(input_tensor,),
    kwargs={},
)
graph = extractor.extract_dag()
```

#### TransformerExtractor

Extracts computation graphs from HuggingFace Transformers models.

```python
from msmodelslim.core.graph.fast_ops_grapher import TransformerExtractor

extractor = TransformerExtractor.create(
    model=model,
    tokenizer=tokenizer,
)
graph = extractor.extract_dag()
```

#### TransformerAutoExtractor

Automatically loads a HuggingFace Transformers model from a model path and extracts the computation graph.

```python
from msmodelslim.core.graph.fast_ops_grapher import TransformerAutoExtractor

extractor = TransformerAutoExtractor.create(
    model_path="meta-llama/Llama-2-7b-hf",
    trust_remote_code=False,
)
graph = extractor.extract_dag()
```

### Computation Graph Structure API

The computation graph consists of three core classes: `ComputationGraph`, `GraphNode`, and `GraphEdge`.

#### Import Method

```python
from msmodelslim.core.graph.fast_ops_grapher import ComputationGraph, TensorInfo, OperatorRecord
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import GraphNode, GraphEdge
```

#### Data Classes

- `TensorInfo`: Records the metadata of a Tensor (id, varname, dtype, shape)
- `OperatorRecord`: Records the complete information of an operator execution (op_name, inputs, outputs, traceback)

#### ComputationGraph

The computation graph, inherited from `networkx.DiGraph`:

```python
# Iterate nodes and edges
for node in graph.iter_nodes():
    print(node.operator.op_name)

for edge in graph.iter_edges():
    print(edge.tensor.varname, edge.tensor.shape)

# Export formatted output
dot_str = graph.format("dot")
```

#### GraphNode and GraphEdge

```python
# Node navigation
successors = node.get_successors()
predecessors = node.get_predecessors()

# Edge endpoints
source_node = edge.get_source_node()
target_node = edge.get_target_node()
```

### New Extractor Development Guide

A new Extractor must inherit from `BaseExtractor` and implement the required abstract methods.

#### Required Override Methods

- `create` (static method + factory method): Creates an Extractor instance
- `target_module` (property): Returns the PyTorch model to extract the computation graph from
- `dummy_inputs` (property): Returns the dummy input data for executing the model

#### Optional Override Methods

- `_extract_raw_dag`: Executes the model and extracts the raw computation graph
- `_post_process_dag`: Post-processes the raw computation graph

### New Formatter Development Guide

The Formatter formats a `ComputationGraph` into a specific output format.

#### Registration Mechanism

```python
@register_formatter("my_format")
def my_formatter(graph: ComputationGraph) -> str:
    # Implement formatting logic
    pass
```

#### Formatter Function Specification

- Function signature: `def my_formatter(graph: ComputationGraph) -> str:`
- Access graph content through the `ComputationGraph` interfaces: `iter_nodes()`, `iter_edges()`, `get_node()`, and so on

## Test Design

This feature includes the following tests:

- Unit tests: `test_exec_dag.py`, `test_exec_trace.py`, `test_extractors.py`, `test_formatters.py`
- Integration tests: `test_integration.py`

The tests cover core functions such as Extractor extraction, computation graph construction, and Formatter formatting.

## Appendix: Program Samples

### Sample 1: NativeModuleExtractor Simple Model Sample

This sample demonstrates how to use NativeModuleExtractor to extract a computation graph from a simple custom nn.Module.

```python
"""NativeModuleExtractor usage example."""
import pickle
import torch
import torch_npu
from torch import nn
from msmodelslim.core.graph.fast_ops_grapher import NativeModuleExtractor


class SimpleModel(nn.Module):
    """A simple two-layer linear model for demonstration."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


model = SimpleModel().npu()
input_tensor = torch.randn(1, 10).npu()

extractor = NativeModuleExtractor.create(model, args=(input_tensor,), kwargs={})
graph = extractor.extract_dag()

dot_str = graph.format("dot")
with open("native_module.dot", "w", encoding="utf-8") as f:
    f.write(dot_str)

with open("native_module.pkl", "wb") as f:
    pickle.dump(graph, f)

print("Graph saved to native_module.dot and native_module.pkl")
```

### Sample 2: TransformerExtractor Sample

This sample demonstrates how to use TransformerExtractor to extract a computation graph from a loaded HuggingFace Transformers model.

```python
"""TransformerExtractor usage example."""
import pickle
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.core.graph.fast_ops_grapher import TransformerExtractor

MODEL_PATH = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).npu()

extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
graph = extractor.extract_dag()

dot_str = graph.format("dot")
with open("transformer.dot", "w", encoding="utf-8") as f:
    f.write(dot_str)

with open("transformer.pkl", "wb") as f:
    pickle.dump(graph, f)

print("Graph saved to transformer.dot and transformer.pkl")
```

### Sample 3: TransformerAutoExtractor Sample

This sample demonstrates how to use TransformerAutoExtractor to automatically load a HuggingFace Transformers model and extract the computation graph.

```python
"""TransformerAutoExtractor usage example."""
import pickle
from msmodelslim.core.graph.fast_ops_grapher import TransformerAutoExtractor

MODEL_PATH = "Qwen/Qwen3-0.6B"

extractor = TransformerAutoExtractor.create(model_path=MODEL_PATH)
graph = extractor.extract_dag()

dot_str = graph.format("dot")
with open("transformer_auto.dot", "w", encoding="utf-8") as f:
    f.write(dot_str)

with open("transformer_auto.pkl", "wb") as f:
    pickle.dump(graph, f)

print("Graph saved to transformer_auto.dot and transformer_auto.pkl")
```

### Sample 4: DeepSeek V4 Sample

This sample demonstrates how to use NativeModuleExtractor to extract a computation graph from a DeepSeek V4 model. The sample reduces the number of layers (num_hidden_layers is reduced to 5) and scales down the model size (dim and moe_inter_dim). This approach covers various intra-layer and inter-layer patterns while reducing device memory usage and speeding up extraction.

```python
"""DeepSeek V4 fast_ops_grapher example."""
import pickle
import torch
import torch_npu
from msmodelslim.core.graph.fast_ops_grapher import NativeModuleExtractor
from msmodelslim.model.deepseek_v4.model import ModelArgs, Transformer

torch.set_default_device('npu')
model_args = ModelArgs(
    num_hidden_layers=5,  # Reduce to first 5 layers to cover various intra-layer and inter-layer patterns
    dim=512,              # Reduce model size to lower device memory usage and speed up extraction
    moe_inter_dim=256,    # Reduce model size to lower device memory usage and speed up extraction
)

model = Transformer(model_args).eval()
print(f'{model=}')

x = torch.randint(0, model_args.vocab_size, (1, 1)).npu()
extractor = NativeModuleExtractor.create(model, args=(x,), kwargs={})

graph = extractor.extract_dag()

with open("deepseek_v4.dot", "w", encoding="utf-8") as f:
    dot_str = graph.format("dot")
    f.write(dot_str)
with open("deepseek_v4.pkl", "wb") as f:
    pickle.dump(graph, f)

print("Graph saved to deepseek_v4.dot and deepseek_v4.pkl")
```
