# PDMIX Inter-Phase Mixed Quantization Algorithm Terminology Entry

> **Entry Category**: [Quantization Algorithms](../README.md#2-quantization-algorithms)<br>
> **English Name**: pdmix<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>

---

## 1. Overview

PDMIX (Prefill-Decode Mix) is a mixed quantization method that switches the activation quantization granularity according to the inference phase. It uses a quantization strategy in the prefill phase that better accommodates the per-token dynamic range, and a static strategy in the decode phase that favors runtime efficiency, balancing accuracy and performance; its core characteristics are phase adaptivity, shared quantized weights, and explicit exploitation of the numerical distribution differences between prefill and decode.

---

## 2. Term Description

Conventional W8A8 static quantization uses static activation quantization parameters, which easily produce a large quantization error in long-context or distribution-shift scenarios, requiring many layers to be rolled back to control the accuracy loss and thereby losing the performance benefit. PDMIX observes that the prefilling phase is sensitive to accuracy while the decoding phase is sensitive to performance, and therefore uses dynamic and static quantization in the respective phases, balancing accuracy and performance with only a small number of fallbacks.

### 2.1 Core Idea

The core idea of PDMIX is "phase-adaptive activation quantization": the weight quantization stays unchanged, while activation quantization uses per-token dynamic quantization in the prefilling phase (token-level granularity, with quantization parameters computed online) and per-tensor static quantization in the decoding phase (quantization parameters computed offline, reducing inference latency). Because the weight quantization stays consistent across phases, only one copy of the quantized weights needs to be stored.

The core benefit of PDMIX is putting "accuracy first" and "decode latency first" into the phases where they fit best.

### 2.2 Working Mechanism

Prefilling and decoding in large model inference have clearly different workloads. Prefilling processes many tokens at once, the activation distribution differs considerably across tokens, and the extra per-token statistics cost is easily amortized by the large matrix computation; a dynamic per-token scale can therefore achieve better range adaptation at a relatively small cost. Decoding processes only a few tokens per step and is more sensitive to latency and small-operator overhead; if a dynamic reduction were performed to obtain the scale at every step, the quantization itself could become a visible cost.

PDMIX therefore lets the two phases share the same low-bit weight representation while using different strategies for activations: prefill uses dynamic, fine-grained quantization to prioritize accuracy, and decode uses an offline, coarser-grained static scale to reduce runtime statistics. It is not "a different bit width per phase" but a switch of the quantization-parameter generation method and sharing granularity according to the phase characteristics.

### 2.3 Mathematical Description

Let the activation value be $X$; the prefilling phase uses per-token dynamic quantization:

$$
Q_{\text{prefill}} = \operatorname{clamp}\left(\operatorname{round}\left(\frac{X}{S_{\text{token}}}\right) + Z_{\text{token}}, \; Q_{\min}, Q_{\max}\right)
$$

The decoding phase uses per-tensor static quantization:

$$
Q_{\text{decode}} = \operatorname{clamp}\left(\operatorname{round}\left(\frac{X}{S_{\text{tensor}}}\right) + Z_{\text{tensor}}, \; Q_{\min}, Q_{\max}\right)
$$

- $X$: activation values
- $S_{\text{token}}$: the per-token dynamic scaling factor (computed online during prefilling)
- $Z_{\text{token}}$: the per-token dynamic zero point
- $S_{\text{tensor}}$: the per-tensor static scaling factor (computed offline)
- $Z_{\text{tensor}}$: the per-tensor static zero point
- $Q_{\min}$, $Q_{\max}$: the quantization numerical range ($[-128, 127]$ for INT8)

Overall, PDMIX is equivalent to a segmented quantization scheme of "dynamic during prefilling, static during decoding", with weight quantization combined with per-channel.

If the dynamic scale computation cost is denoted $C_s$ and the main matrix-multiplication cost $C_g$, the extra relative overhead can be roughly viewed as $C_s/C_g$. As the token count grows during prefill, $C_g$ is large and this ratio is usually small; during decode, $C_g$ per step is smaller and this ratio may grow considerably. PDMIX does not change the basic form $q=\operatorname{round}(x/S)+Z$; it changes when and at what granularity $S,Z$ are estimated.

### 2.4 Key Properties

- **Phase adaptive**: per-token for prefilling and per-tensor for decoding, balancing accuracy and performance.
- **Single copy of weights**: the weight quantization is consistent across phases, so only one set of quantized weights needs to be stored.
- **Low fallback cost**: compared with static quantization, rolling back a small number of layers suffices to control the accuracy loss.
- **Inference acceleration**: static quantization in the decoding phase reduces quantization-parameter computation and improves throughput.
- **A phase strategy rather than mixed bit widths**: the core difference lies in activation scale generation and sharing granularity; prefill/decode are not required to use different bit widths.

The risk mainly comes from out-of-distribution inputs for the static decode scale: if the calibration data does not cover the large magnitudes of certain decoding tokens, the static per-tensor range may saturate.

### 2.5 Use Cases

- Long-context or distribution-shift scenarios where static quantization loses considerable accuracy and many layers must be rolled back.
- Generative model inference acceleration scenarios that want the performance benefit of static quantization while controlling the accuracy loss.

### 2.6 Usage Restrictions

- Currently only MindIE inference deployment is supported.
- Only one quantization mode, W8A8 PDMIX, is supported (mixed dynamic/static INT8 activations, per_channel INT8 weights).
- The essence of PDMIX is that prefill and decode use different activation quantization granularities; if the granularity or symmetry combination of the two phases is changed, it must be re-evaluated whether the expected trade-off between accuracy and decoding overhead still holds.

---

## 3. Related Terms

- [Linear Quantization](../linear_quant/term_linear_quant.md): the parent concept; PDMIX is a method of activation quantization within linear quantization.
- [MinMax](../minmax/term_minmax.md): the application target; PDMIX's quantization parameters are computed with the MinMax algorithm.
- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md): a companion term in the same family of generative model inference optimization solutions.

---

## 4. Reference Documents

1. [PDMIX Parameter Configuration Workflow Guide](./usage_pdmix.md)
