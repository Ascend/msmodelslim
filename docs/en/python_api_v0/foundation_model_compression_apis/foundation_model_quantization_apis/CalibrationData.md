# CalibrationData

## Function

Hybrid calibration dataset API. The CalibrationData class mixes specified datasets and supports user-defined datasets.

## Prototype

```python
CalibrationData(config_path, save_path, tokenizer=None, model=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ----- |--------|-------------|------------------------------------------------|
| config_path | Input | Dataset config path. | Required.<br>Data type: string. |
| save_path | Input | Hybrid dataset save file path. | Optional. If empty, the dataset is not saved to a file.<br>If selected, it must be a JSON file.<br>Data type: string. |
| tokenizer | Input | Tokenizer instance. | Optional.<br>Data type: Subclass of PreTrainedTokenizer generated based on the model. |
| model | Input | Model instance to be quantized. | Optional. Used to obtain positive samples.<br>Data type: PyTorch model. |

## API Description

### add_custormized_dataset_processor

```python
# Add user-defined dataset interface. Must be called before set_sample_size(sample_size). Optional.
# Input dataset_name: user-defined dataset name, type string. Must match the user-defined dataset name used in set_sample_size(sample_size).
# Input processor: user-defined dataset processing class instance, inherits from DatasetProcessorBase.
#               Must override DatasetProcessorBase.process_data(indexes) and DatasetProcessorBase.verify_positive_prompt(prompts, labels).
CalibrationData.add_custormized_dataset_processor(dataset_name=customized_dataset_name, processor=customized_processor)
```

### set_sample_size

```python
# Set sample size. Required.
# Input sample_size: dict type, for example, sample_size = {"dataset_name": size}.
#               dataset_name: string, case-sensitive. Must be a dataset name configured in config.json or a user-defined dataset name.
#               size: int, > 0. If empty or 0, sample size for the corresponding dataset in the mixed calibration set is 0. If non-int, error. If exceeding dataset size or max sample limit, uses max sample size.
CalibrationData.set_sample_size(sample_size=sample_size)
```

### set_batch_size

```python
# Set batch size. Optional.
# Input batch_size: int, > 0. Default is 1.
CalibrationData.set_batch_size(batch_size=batch_size)
```

### set_shuffle_seed

```python
# Set random seed for reproducible results on the same device. Optional.
# Input shuffle_seed: int. Default is 0.
CalibrationData.set_shuffle_seed(shuffle_seed=shuffle_seed)
```

### process

```python
# Run interface. Required.
# Output: mixed_dataset of type list, for example, [{"prompt": prompt1, "ans": ans1}, {"prompt": prompt2, "ans": ans2}]
CalibrationData.process()
```

## Calling Instructions

Please refer to [Usage of Mixed Calibration Datasets](../../../feature_guide/traditional_quantization_v0/foundation_model_quantization_and_calibration.md#usage-of-mixed-calibration-datasets)
