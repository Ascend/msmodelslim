# Qwen3-32B W8A8 Accuracy Tuning Case

## Overview

### Case Objective

**Objective**: Quantize the Qwen3-32B model to W8A8 and keep the accuracy loss of the quantized model within a controllable range compared with the floating-point model.

**Initial state**: Full static quantization (per_channel/per_tensor) combined with the Smooth Quant outlier suppression algorithm produced garbled chat output after quantization, and the model cannot be used normally.

## Preparation

Install msModelSlim. For details, see [msModelSlim Installation Guide](../getting_started/install_guide.md).

## Tuning Procedure

### Step 1: Confirming That the Accuracy Issue Is Real

Before tuning, first rule out environmental interference and make sure the issue is real:

- **Inference engine check**: The floating-point model reproduces the original accuracy on the target inference engine.
- **Result check**: The quantized model evaluation output is obviously abnormal, with garbled chat responses, which confirms a quantization accuracy problem.
- **Determine the fluctuation range**: The AIME25 evaluation dataset shows an abnormal accuracy drop.

### Step 2: Adjusting the Outlier Suppression Algorithm (Primary Step)

The initial Smooth Quant configuration caused garbled output. According to the tuning strategy, different outlier suppression algorithms are tried in sequence:

| Outlier Suppression Algorithm| AIME25 Accuracy (%)| Quantization Time (s)| Remarks|
|---------------|-----------------|--------------|------|
| Smooth Quant | Garbled chat| 326 | Initial configuration. Accuracy drops sharply.|
| Iterative Smooth (symmetric; alpha: 0.5)| 53.33 | 324 | Better than Smooth Quant, but accuracy is still not enough.|
| Iterative Smooth (asymmetric; alpha: 0.5)| 63.33 | 305 | The asymmetric scheme improves accuracy by 10 percentage points and meets expectations.|
| Iterative Smooth (symmetric; alpha: 0.9)| 66.67 | 319 | Accuracy improves further after `alpha` adjustment.|
| Flex Smooth Quant | 63.33 | 1380 | Accuracy is comparable to Iterative Smooth (asymmetric; alpha: 0.5), but it takes much longer.|

**Tuning result**:

Considering both accuracy and quantization time, the final choice is **Iterative Smooth (symmetric; alpha: 0.9)**. The analysis is as follows:

**1. Accuracy comparison:**

- Iterative Smooth (symmetric; alpha: 0.9) reaches 66.67% accuracy, which is the highest among the symmetric solutions.
- Compared with Iterative Smooth (symmetric; alpha: 0.5), which reaches 53.33%, accuracy improves by 13.34 percentage points.
- Although Iterative Smooth (asymmetric; alpha: 0.5) reaches 63.33%, the symmetric solution with alpha: 0.9 reaches 66.67%, which is higher.
- Compared with Flex Smooth Quant at 63.33%, accuracy improves by 3.34 percentage points.

**2. Quantization time comparison**:

- Iterative Smooth (symmetric; alpha: 0.9) takes 319 seconds, which is similar to the 324 seconds of Iterative Smooth (symmetric; alpha: 0.5).
- Compared with Flex Smooth Quant at 1,380 seconds, quantization time drops by 76.9%, which significantly improves efficiency.

**Final decision**:

Based on the above analysis, **Iterative Smooth (symmetric; alpha: 0.9)** is the optimal choice for the scenario.

### Step 3: Choosing the Quantization Algorithm

After the outlier suppression algorithm is determined, optimize the quantization algorithm configuration.

#### Quantization Method Comparison

| Weight Quantization Method| Activation Quantization Granularity| AIME25 Accuracy (%)| Quantization Time (s)| Remarks|
|-------------|-------------|----------------|-------------|------|
| minmax | per_tensor (static quantization)| 66.67 |319 | Baseline configuration, based on the result of step 2|
| minmax | per-token (dynamic quantization)|80.00 | 289 | Accuracy improves by 13.33 percentage points after per-token is used for activation.|
| ssz | per_tensor (static quantization)| 63.33 | 408 | ssz is used for weight quantization, but the static quantization accuracy decreases.|
| ssz | per-token (dynamic quantization)| 70.00 | 348 | The ssz + per-token solution is less accurate than the minmax + per-token solution.|

#### Tuning Result

Considering both accuracy and quantization time, the final choice is **minmax + per-token (dynamic quantization)**. The analysis is as follows:

**1. Accuracy comparison**:

- minmax + per-token reaches 80.00% accuracy, which is 13.33 percentage points higher than minmax + per_tensor at 66.67%.
- ssz + per_tensor reaches 63.33%, which is 3.34 percentage points lower than the minmax + per_tensor configuration. This shows that ssz performs worse than minmax in this INT8 quantization scenario.
- ssz + per-token reaches 70.00%, which is 10 percentage points lower than minmax + per-token at 80.00%. This also shows that ssz is not as effective as minmax in this INT8 dynamic quantization scenario.

**2. Quantization time comparison**:

- minmax + per-token takes 289 seconds, which saves 30 seconds, or 9.4%, compared with minmax + per_tensor at 319 seconds.
- ssz + per-token takes 348 seconds, which is 59 seconds, or 20.4%, slower than minmax + per-token.

**3. Overall comparison**:

- **Accuracy**: minmax + per-token gives the highest accuracy at 80.00%, better than ssz + per-token at 70.00% and all static quantization solutions.
- **Quantization time**: minmax + per-token has the shortest time at 289 seconds. It saves 59 seconds, or 17.0%, compared with ssz + per-token at 348 seconds, and it also saves 30 seconds compared with minmax + per_tensor at 319 seconds.
- **Method complexity**: minmax is simple and fast. ssz uses iterative search and is more complex. For INT8 quantization, minmax should be the first choice.

**Final decision**:
Based on the analysis above, **minmax + per-token (dynamic quantization)** gives the best balance between accuracy and quantization time. It not only gives the highest accuracy at 80.00%, which is 10 percentage points higher than ssz + per_token, but also has the shortest quantization time at 289 seconds. It is also simpler to implement. Compared with the step 2 configuration at 66.67%, accuracy improves by 13.33 percentage points, or 20% relative improvement, which lays a good foundation for later tuning.

### Step 4: Adjusting the Calibration Set

After step 3, accuracy reaches 80.00%, which already meets the target. To show the full tuning workflow and verify the effect of calibration set adjustment, this section uses the GPQA dataset for validation. Because GPQA contains more questions, it shows accuracy differences between configurations more clearly. This section uses Iterative Smooth with a static quantization strategy as the baseline configuration.

#### Calibration Set Optimization Strategy

The quality of the calibration set directly affects the accuracy of the quantization parameters.

| Adjustment Strategy| Operation| Calibration Set Change| Optimization Objective|
|---------|---------|-----------|---------|
| Initial calibration set| 10 random samples| 10 samples| Establish the baseline configuration.|
| Increase the amount of data.| Increase from 10 samples to 30 samples.| From 10 to 30 samples| Improve the accuracy of quantization parameter estimation.|
| Match the application scenario.| Replace random data with Chinese dialogue data.| 30 samples (Chinese dialogue)| Make the calibration data closer to the real application scenario.|
| Balance the data distribution.| Mix samples drawn from GPQA, C-Eval, MMLU, and other datasets.| 30 samples (mixed from multiple datasets)| Improve the diversity and balance of the data distribution.|
| Remove abnormal data.| Remove 3 abnormal samples that cause accuracy to drop.| From 30 to 27 samples| Reduces the interference of abnormal samples on quantization parameters.|
| Add bad cases.| Add 5 bad-case samples from the floating-point model on GPQA.| From 27 to 32 samples| Help the quantized model learn hard samples and improve accuracy.|

#### Tuning Process

Add the bad-case samples from the AISBench evaluation results to the quantization calibration set and regenerate the quantized weights. The procedure is as follows.

1. **Obtain bad-case samples**: Extract a small number of bad-case samples from the AISBench results. For example, one bad-case sample is:

    ```text
    What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

    Choices:
    (A)10^-11 eV
    (B)10^-8 eV
    (C)10^-9 eV
    (D)10^-4 eV
    Format your response as follows: "The correct answer is (insert answer here)"
    ```

2. **Format conversion**:

   - **JSONL format**: Refer to `msmodelslim/lab_calib/mix_calib.jsonl` and place the text after the `"inputs_pretokenized"` field, as shown below:

     ```json
     {"inputs_pretokenized":"What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nChoices:\n(A)10^-11 eV\n(B)10^-8 eV\n\n(C)10^-9 eV\n(D)10^-4 eV\nFormat your response as follows: \"The correct answer is (insert answer here)\""}
     ```

   - **JSON format**: Refer to `msmodelslim/lab_calib/qwen3_cot_w4a4.json` and put the text directly into the string list.

3. **Requantization**: Use the adjusted calibration set for quantization and regenerate the quantized weights.

#### Tuning Result

| Quantization Strategy| GPQA Accuracy (%)| Remarks|
|-------------|---------------|------|
| Iterative Smooth + static quantization| 46.97 | Baseline configuration|
| Iterative Smooth + static quantization + bad-case calibration adjustment| 55.56 | Accuracy improves by 8.59 percentage points compared with the baseline. This shows that bad-case samples help the quantized model learn hard samples and improve accuracy.|

### Step 5: Quantization Rollback (Alternative Option)

Quantization rollback keeps quantization-sensitive layers at the original floating-point precision to improve the accuracy of the quantized model. When the accuracy is still not enough after steps 1 to 4, you can further optimize with rollback. This section validates the rollback effect on the GPQA dataset and shows the complete tuning workflow.

#### Use Cases

Quantization rollback applies in these situations:

- Accuracy still does not meet the target after steps 1 to 4.
- You need a finer balance between accuracy and performance.
- Certain layers are extremely sensitive to quantization and must stay at high precision.

#### Tuning Process

**1. Sensitive layer analysis**

Use the sensitive layer analysis tool provided by msModelSlim to identify quantization-sensitive layers. For details, see [Quantization Sensitive Layer Analysis Guide](../feature_guide/sensitive_layer_analysis/analyze_api_usage.md).

Run the analysis command:

```bash
msmodelslim analyze \
    --model_type Qwen3-32B \
    --model_path ${model_path}
```

The layers are sorted from the highest sensitivity score to the lowest. The top sensitive layers are:

```text
layers.3.mlp.down_proj
layers.63.mlp.down_proj
layers.2.mlp.down_proj
layers.1.mlp.down_proj
layers.4.mlp.down_proj
layers.6.mlp.down_proj
layers.7.mlp.down_proj
layers.5.mlp.down_proj
layers.0.mlp.down_proj
layers.31.mlp.down_proj
layers.62.mlp.down_proj
layers.5.mlp.gate_proj
layers.5.mlp.up_proj
layers.32.mlp.down_proj
layers.8.mlp.gate_proj
layers.8.mlp.up_proj
layers.6.mlp.gate_proj
layers.6.mlp.up_proj
```

**Analysis result**: `mlp.down_proj` ranks near the top in sensitivity and is harder to quantize, so it should be rolled back first.

**2. Quantization configuration modification**

In the YAML quantization config, roll back the nine most sensitive layers, all of which are `mlp.down_proj` layers, by using the `exclude` field:

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: True
      enable_subgraph_type:
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include:
        - "*"
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"
          dtype: "int8"
          symmetric: false
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: 
        - "*"
      exclude:
        - 'model.layers.3.mlp.down_proj'
        - 'model.layers.63.mlp.down_proj'
        - 'model.layers.2.mlp.down_proj'
        - 'model.layers.1.mlp.down_proj'
        - 'model.layers.4.mlp.down_proj'
        - 'model.layers.6.mlp.down_proj'
        - 'model.layers.7.mlp.down_proj'
        - 'model.layers.5.mlp.down_proj'
        - 'model.layers.0.mlp.down_proj'
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

**3. Regeneration of the quantized weights**

Run quantization again with the updated config to generate a quantized model that includes the rolled-back layers.

#### Tuning Result

| Quantization Strategy| GPQA Accuracy (%)| Remarks|
|-------------|---------------|------|
| Iterative Smooth + static quantization| 46.97 | Baseline configuration|
| Iterative Smooth + static quantization + rollback of the top 9 layers| 51.51 | Accuracy improves by 4.54 percentage points compared with the baseline. This shows that rolling back quantization-sensitive layers can improve accuracy, but it also adds performance overhead and increases model size.|

## Final Configuration Summary

### Tuning Path Review

| Step| Key Operation| AIME25 Accuracy (%)| Accuracy Improvement| Remarks|
|------|---------|-----------------|---------|------|
| Initial state| Smooth Quant + minmax + static quantization| Garbled output| - | Initial configuration, not usable|
| Step 2| Iterative Smooth (symmetric; alpha: 0.9)| 66.67% | +66.67% | Outlier suppression is improved and garbled output is fixed.|
| Step 3| minmax + per-token (dynamic quantization)| 80.00% | +13.33% | Activation granularity is improved and the accuracy target is reached.|

**Note**: After step 3 reached 80.00% accuracy, the preset accuracy target is met. Steps 4 and 5 use the GPQA dataset to validate the effect of calibration set adjustment and quantization rollback.

### Final Configuration

**Outlier suppression algorithm**: Iterative Smooth (symmetric; alpha: 0.9)

**Quantization configuration**:

- **Weight quantization**: `minmax`, `per_channel`, int8 data type, symmetric quantization
- **Activation quantization**: `minmax`, `per_token` granularity, dynamic quantization, int8 data type, symmetric quantization

### Tuning Experience Summary

1. **Outlier suppression is the key**. Switching from Smooth Quant to Iterative Smooth (symmetric; alpha: 0.9) improves accuracy from garbled output to 66.67% and makes the model usable.

2. **Activation granularity matters a lot**. Switching from per_tensor static quantization to per-token dynamic quantization improved accuracy from 66.67% to 80.00%, a gain of 13.33 percentage points, or 20% relative improvement. Keep in mind that dynamic quantization may reduce inference performance.

3. **Quantization method choice matters**. In INT8 quantization scenarios, minmax performs better than ssz. It gives higher accuracy at 80.00% versus 70.00%, takes less time at 289 seconds versus 348 seconds, and is simpler to implement, so it is the best choice for this scenario.

4. **Calibration set quality affects accuracy**. On GPQA, adding bad-case samples to optimize the calibration set improves accuracy from 46.97% to 55.56%, a gain of 8.59 percentage points. This shows that scenario-matched data and hard samples have a significant impact on quantization accuracy.

5. **Quantization rollback is the last resort**. On GPQA, rolling back nine quantization-sensitive layers improves accuracy from 46.97% to 51.51%, a gain of 4.54 percentage points. Rollback can improve accuracy effectively, but it also adds performance overhead and increases model size, so you should use it only when other optimization methods cannot meet the requirement.
