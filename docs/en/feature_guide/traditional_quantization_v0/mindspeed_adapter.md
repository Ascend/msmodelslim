# MindSpeed Adapter

## Overview

The original `llm_ptq` module primarily supports the quantization and compression of foundation models based on the Transformers framework. This module provides a quantization adapter for ModelLink models, which enables direct quantization of the MindSpeed-LLM model.

## Preparations

- This feature is supported only on the following products.
    - Atlas training products
    - Atlas A2 training products/Atlas 800I A2 inference products/A200I A2 Box heterogeneous components

- Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).
- Install the following dependencies for the foundation model quantization tool.
  If you run the following commands as a non-root user, add `--user` to the end of each installation command, such as `pip3 install onnx --user`.

```bash
pip3 install numpy==1.25.2
pip3 install transformers       # The version must be 4.29.1 or later. For the LLaMA model, the 4.29.1 version must be installed.
pip3 install accelerate==0.21.0  # If a model needs to be quantized in multi-NPU parallel mode, the version must be 0.28.0 or later.
pip3 install tqdm==4.66.1
```

- Install the MindSpeed-LLM library. For details, see [MindSpeed LLM Installation Guide](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/mindspore/install_guide.md).

## Function

## Constraints

Currently, the model adapter is verified to support only w8a8 quantization, alongside the m3 and m5 algorithms of the outlier suppression module. Quantization execution is supported only on the NPU. CPU-based quantization is not supported.

## Quantization Procedure (Using LLaMA2-7B Legacy as an Example)

1. Obtain the open-source weights and convert them into a model supported by MindSpeed-LLM. You can use the MindSpeed-LLM [weight conversion script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/convert_ckpt.py). A conversion script usage tutorial is available [here](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/tools/checkpoint_convert_hf_mcore_large_params.md).

    ```bash
    python convert_ckpt.py \
        --model-type GPT \
        --load-model-type hf \
        --save-model-type mg \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --load-dir ./model_from_hf/llama-2-7b-hf/ \
        --save-dir ./model_weights/llama-2-legacy/ \
        --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model \
        --model-type-hf llama2
    ```

2. Design the quantization function. The following example shows a w8a8 quantization configuration:

    ```python
    def quant(model):
        # Prepare calibration data and modify the data as needed. In W8A16 Label-Free mode, skip this step.
        dataset_calib = [["Where is the capital of China?"],
                    ["Please write a poem:"],
                    ["How can I learn Python?"]]

        from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, CalibratorAdapter, Linear    # import the quantization configuration interfaces.
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
        # Convert the model to adapt to MindSpeed-LLM.
        model = ModelAdapter(model)
        # Configure fallback layers. The following example shows how to perform mlp.dense_4h_to_h fallback configuration.
        disable_names = []
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        for name, mod in model.named_modules():
            if isinstance(mod, Linear) and "mlp.dense_4h_to_h" in name:
                disable_names.append(name)
        # Quantization configuration (modify it as needed)
        # Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
        quant_config = QuantConfig(
            w_bit=8,  
            a_bit=8,         
            disable_names=disable_names, 
            dev_type='npu',
            mm_tensor=False
        )  
        # Use the CalibratorAdapter interface to define calibration by passing the loaded original model, quantization configuration, and calibration data.
        calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
        calibrator.run()     # Use run() to perform quantization.
        calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      # Save model quantization parameters by using save(). Modify the path as needed.
        print('Save quant weight success!')
    ```

3. Insert the quantization function designed above into the inference script. The following example shows how to insert the `quant` function into the `main` function by using the built-in inference accuracy test script [evaluation.py](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/evaluation.py) of MindSpeed-LLM. If `trust_remote_code` is set to `True`, code files in the weights directory of the floating-point model may be executed. Ensure that the source of the floating-point model is secure and reliable.

    ```python
    ...
    def main():
        initialize_megatron(extra_args_provider=add_text_generate_args,
                            args_defaults={'no_load_rng': True,
                                        'no_load_optim': True})
        args = get_args()
        model = MegatronModuleForCausalLM.from_pretrained(
            model_provider=model_provider,
            pretrained_model_name_or_path=args.load, 
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True, local_files_only=True)
        quant(model) # Insert the quantization function written previously.
        rank = dist.get_rank()
        if 'mmlu' in args.task:
            a = time.time()
            mmlu(args, LLMChat(args, model, tokenizer))
            if rank == 0:
                logger.info(f'MMLU Running Time:, {time.time() - a}')
    ...

    ```

4. Modify the MindSpeed-LLM inference script `evaluate_llama2_7B_ptd.sh` to execute the quantization process described above. The following example shows how to modify the script model path by using a legacy startup configuration.

    ```bash
    ...
    TOKENIZER_PATH=./model_from_hf/llama-2-7b-hf/  # Path to the Hugging Face open-source model.
    CHECKPOINT=./model_weights/llama-2-legacy/  # Path to the weights generated through conversion.
    ...
    python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
    ...
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 1  \
    --padded-vocab-size 32000 \ # Configure the corresponding parameters for the model.
    ...
    ```

5. Execute the inference script to complete quantization and verify the fake-quantization accuracy.

    ```bash
    bash examples/legacy/llama2/evaluate_llama2_7B_ptd.sh
    ```
