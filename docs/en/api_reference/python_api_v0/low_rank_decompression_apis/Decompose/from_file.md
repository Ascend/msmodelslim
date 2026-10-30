# from_file

## Function

Decompose class method. If other from_xxx() methods have been called and the config_file specified during class initialization is valid, it loads the decomposition information from the saved file.

## Prototype

```python
from_file()
```

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
decomposer = decomposer.from_file()
```
