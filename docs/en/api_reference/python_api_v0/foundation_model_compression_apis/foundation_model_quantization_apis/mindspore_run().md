# run()

## Function

Runs the quantization algorithm. After initializing the Calibrator, execute quantization through the run() function.

## Prototype

```python
calibrator.run()
```

## Sample

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()    # Load according to the actual model situation.
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
calibrator.run() 
calibrator.save("./quant_model.ckpt")
```
