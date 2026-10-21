# msModelSlim Quick Start

## 1. Overview

MindStudio ModelSlim (msModelSlim) is a model compression tool for the Ascend ecosystem, covering the quantization and compression of dense LLMs, MoE, and multimodal models. This document uses Qwen3.6-27B as an example to walk you through compressing model weights into the W8A8 format through one-click quantization, and completing inference deployment verification based on vLLM-Ascend.

**Experience map (core operations take about 10 minutes, excluding network transfer time such as image and model downloads)**

| Step | Stage | Core Tool | Operation Time | Principle Learning |
|:--:|:-------|:---------------|:----------------:|:-----:|
| 1 | Container environment preparation | vLLM-Ascend container | About 1 minute (excluding image download time) | 5 minutes |
| 2 | Model file preparation | ModelScope | About 1 minute (excluding model download time) | 2 minutes |
| 3 | Model quantization | msModelSlim | About 3 minutes | 5 minutes |
| 4 | Quantization result verification | vLLM-Ascend | About 5 minutes | 10 minutes |

## 2. Procedure

### 2.1 Environment Preparation (Required)

🛑 **This section is mandatory! Skipping this section may cause multiple subsequent operations to fail.**

This tutorial is **only supported** in the standardized vLLM-Ascend container and is not supported on bare metal, virtual machines, or other non-standard container environments.

#### 2.1.1 Prerequisites

Before you begin, confirm that the server meets the following requirements:

| Item       | Requirement                                           | Verification Method                              |
|----------|----------------------------------------------|-----------------------------------|
| **Hardware computing power** | The Linux server is equipped with at least 2 NPU cards (Atlas A2 products or Atlas A3 products), with drivers and firmware installed. | Run `npu-smi info` and confirm that the NPU card status is normal.   |
| **Container runtime** | Docker is installed and running (recommended version ≥ 18.0).                  | Run `docker ps`; no error indicates that the service is running normally.      |
| **Script execution** | Python 3 is installed on the host machine, with a version no earlier than 3.8 and no later than 3.12 (3.10 or 3.11 is recommended). | Run `python3 -V` on the host machine and confirm that the output version is within the 3.8-3.12 range. |
| **Network communication** | curl (any version) is installed.                              | Run `curl -V`; version information output indicates that it is installed.       |
| **Disk space** | At least 100 GB of free disk space (for downloading model weights).                   | Run `df -h` to check disk space usage.                    |

> 👉 After confirming that the prerequisites are met, if the environment has public network access, all commands in this chapter can be executed directly by **Copy/Paste** without manual input or concatenation.

#### 2.1.2 Host Machine: Automatically Identifying and Configuring Image Environment Variables

Run the following command on the host machine. The command reads the NPU PCI ID, matches the image version, and sets an environment variable for use in subsequent steps.

```bash
dev_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2)
source /dev/stdin <<< "$(
  case "$dev_id" in
    'd802' )
      echo 'export MY_STUDY_VAR_VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.18.0"'
      echo 'echo -e "\e[32m[PASS] Successfully auto-selected image: $MY_STUDY_VAR_VLLM_IMAGE\e[0m"'
      ;;
    'd803' )
      echo 'export MY_STUDY_VAR_VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.18.0-a3"'
      echo 'echo -e "\e[32m[PASS] Successfully auto-selected image: $MY_STUDY_VAR_VLLM_IMAGE\e[0m"'
      ;;
    * )
      echo 'unset MY_STUDY_VAR_VLLM_IMAGE'
      echo 'echo -e "\033[31m[FAIL] Get device ID: '"$dev_id"'. Learning is not supported in the current environment.\033[0m" >&2'
      ;;
  esac
)"
```

>[!NOTE]
>
> **Command Principle**
>
> Obtain the NPU PCI ID through `lspci`, automatically match the official vLLM-Ascend image, and assign the image address to the environment variable `MY_STUDY_VAR_VLLM_IMAGE` for subsequent use.
> All images come from the official vLLM-Ascend repository published on Quay.io. For image details, see [vLLM-Ascend Official Image Repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags).

If `[PASS]` is output, the identification succeeded; continue to the next step. If `[FAIL]` is output, the possible causes are as follows:

1. The hardware is not within the supported range: this tutorial supports only Atlas A2 products and Atlas A3 products. Switch to compatible hardware and retry.
2. Underlying environment issue: `lspci` is not installed, or the current user does not have permission to run `lspci -n -D`. Contact the environment administrator for confirmation.

#### 2.1.3 Host Machine: Pulling the Image

Run the following command on the host machine:

```bash
docker pull ${MY_STUDY_VAR_VLLM_IMAGE}
```

If the pull fails due to enterprise intranet restrictions, refer to [Obtaining Docker Images in an Isolated Intranet](#31-obtaining-docker-images-in-an-isolated-intranet).

#### 2.1.4 Host Machine: Downloading the Container Startup Script

On the host machine, run:

```bash
cd ~ && curl -fLO --retry 3 https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py && chmod +x ctr_in.py
```

If the download fails due to network restrictions, see [Transferring the Container Startup Script](#32-transferring-the-container-startup-script).

#### 2.1.5 Host Machine: Starting the Container

Run the following command on the host machine. The terminal displays the container creation information and waits for confirmation. Press **Enter** to complete the creation.

```bash
~/ctr_in.py ${MY_STUDY_VAR_VLLM_IMAGE}
```

**Expected result**:

After about 10 seconds, the terminal displays the container startup success log and the root shell prompt, indicating that the container has been started successfully:

Startup success log:

```text
Container 'xxxxxx' started successfully.
```

root shell prompt:

```text
[root@xxxxxx ~]#
```

>[!NOTE]
>
> In the preceding example, `xxxxxx` indicates the container name. The actual name is automatically generated by the container startup script. Refer to the actual terminal output.

#### 2.1.6 Inside the Container: Installing msModelSlim

After entering the container, install msModelSlim and the required transformers version:

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ \
    transformers==5.2.0 \
    https://gitcode.com/Ascend/msmodelslim/releases/download/tag_MindStudio_26.1.0.B100_002/msmodelslim-26.1.0-py3-none-any.whl
```

>[!NOTE]
>
> **transformers Version Selection**
>
> The transformers version depends on the model to be quantized. The Qwen3.6-27B model used in this example must run in a transformers 5.2.0 environment.

If the installation fails due to enterprise intranet restrictions, refer to [Installing Python Dependencies Offline](#33-installing-python-dependencies-offline).

#### 2.1.7 Inside the Container: Verifying Environment Installation

After installation is complete, run the one-click verification:

```bash
python3 -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU is unavailable"; print("PyTorch:", torch.__version__)' && msmodelslim --help >/dev/null && echo -e "\e[32m[PASS] NPU environment and msmodelslim check passed.\e[0m"
```

If `[PASS]` is output, it indicates that the NPU driver, PyTorch, and msModelSlim are all ready. The environment preparation is complete, and you can proceed to the quantization stage.

### 2.2 Performing Quantization

#### 2.2.1 Inside the Container: Preparing Model Files

>[!NOTE]
>
> **Efficient Operation Tips**
>
> The model files are large (approximately 50 GB), and even at full gigabit bandwidth, downloading takes about 10 minutes. It is recommended that after executing the download command, you first read the subsequent sections to learn the quantization principles and deployment process, so that you can perform the subsequent operations more efficiently once the download is complete.

Run the following command to download the original Qwen3.6-27B weights from ModelScope:

```bash
modelscope download --model Qwen/Qwen3.6-27B --local_dir ~/qwen36_27b_base
```

#### 2.2.2 Inside the Container: Preparing NPU Cards

The quantization process requires NPU compute acceleration. Ensure that at least one idle NPU card is available. Run the following command to automatically select an idle card:

```bash
free_npu=$(npu-smi info | grep -oE "No running processes found in NPU\s+[0-9]+" | head -n 1 | awk '{print $NF}')
if [ -n "$free_npu" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=$free_npu
    echo -e "\e[32m[PASS] Successfully exported ASCEND_RT_VISIBLE_DEVICES=$free_npu\e[0m"
else
    echo -e "\e[31m[FAIL] All NPUs are busy. Please release NPUs and try again.\e[0m" >&2
fi
```

If `[PASS]` is output, an idle NPU card has been successfully specified and you can proceed to the next step. If `[FAIL]` is output, release the NPU resources first and then retry the preceding command.

>[!NOTE]
>
> **NPU Card Selection Mechanism**
>
> **Function**: The environment variable `ASCEND_RT_VISIBLE_DEVICES` specifies the NPU IDs visible to the current process (single or multiple), allowing device switching without modifying the code.
>
> **Index Mapping Rules**:
>
> After this variable is set, the **logical indexes of the visible devices are renumbered starting from 0**. Subsequent operations must use the new indexes instead of the original NPU IDs.
>
> - `=1`: Only NPU 1 is visible, and its new index is **0**.
> - `=1,2,3`: NPUs 1, 2, and 3 are visible, and their new indexes are **0, 1, 2** in order.
>
> ⚠️ **Note**: This environment variable is a trial feature and may change in later versions. Do not use it in production environments.

#### 2.2.3 Inside the Container: Performing Model Quantization

Run the following command to use the one-click quantization feature. The system automatically matches the best-practice configuration for this model and completes quantization in W8A8 mode (quantizing both model weights and activations to 8-bit):

```bash
msmodelslim quant --model_path ~/qwen36_27b_base --save_path ~/qwen36_27b_w8a8 --device npu --model_type Qwen3.6-27B --quant_type w8a8 --trust_remote_code true
```

Quantization takes approximately 4 minutes. The following output indicates completion:

```text
msmodelslim.app.naive_quantization - INFO - ===========SUCCESS===========
```

If quantization is aborted or reports an error and the preceding success indicator does not appear, troubleshoot as follows:

1. Confirm that the NPU card status is normal: run `npu-smi info` and check whether the `Health` status of the target card is `OK` and whether the AI Core utilization is abnormal. If abnormal, release resources or replace the card first.
2. Check the environment variable settings: run `echo ${ASCEND_RT_VISIBLE_DEVICES}` to confirm that the card ID referenced by this variable actually exists and is not occupied, and that it does not contain spaces or invalid IDs.
3. Check whether the device memory is sufficient (OOM): check whether the terminal error log contains a keyword similar to `Out of memory`. If it does, the current card has insufficient device memory. Confirm whether the device memory of this card meets the model quantization requirements, and switch to an idle NPU card with sufficient device memory before retrying.

#### 2.2.4 Inside the Container: Viewing the Quantization Output

**1. View the quantization result files**.

```bash
ls -al ~/qwen36_27b_w8a8
```

The output directory structure is similar to the following, where files marked "[Quantization]" are the quantization output files, and files marked "[Original]" are the inference configuration files copied from the original model (only the main files are listed, not a complete list):

```text
~/qwen36_27b_w8a8/
├── Qwen3.6-27B_best_practice.yaml                  # [Quantization] Quantization configuration protocol (full record of quantization settings, reproducible)
├── quant_model_description.json                    # [Quantization] Quantized weight descriptor (tensor-wise quantization types and metadata; required for inference engine loading)
├── quant_model_weights-00001-of-00008.safetensors  # [Quantization] Quantized weight shard 1/8 (INT8 weights)
├── ...                                             # [Quantization] Remaining weight shards (00002–00007)
├── quant_model_weights-00008-of-00008.safetensors  # [Quantization] Quantized weight shard 8/8 (INT8 weights)
├── config.json                                     # [Original] Model configuration (architecture, layers, hidden size, etc.)
├── tokenizer_config.json                           # [Original] Tokenizer configuration (special tokens, vocab size, preprocessing logic)
├── tokenizer.json                                  # [Original] Tokenizer vocabulary (token-to-ID mappings)
├── chat_template.jinja                             # [Original] Chat template (multi-turn prompt formatting)
└── generation_config.json                          # [Original] Generation configuration (temperature, Top-P, max length, sampling strategy)
```

> Note: The actual number of weight shards is determined by the total size of the quantized weights together with `part_file_size` in the best practice configuration (4 GB in this example). Refer to the actual generated result.

**2. Verify the quantization compression.**

Run the following command to compare the directory sizes before and after quantization:

```bash
du -sh ~/qwen36_27b_base
du -sh ~/qwen36_27b_w8a8
```

Expected result: the original weights are more than 50 GB, and the quantized weights are more than 30 GB, a size reduction of about 40%. This indicates that quantization significantly reduces the model size.

### 2.3 Quantized Model Function Verification

This section uses vLLM-Ascend to deploy the quantized model and perform an inference test.

#### 2.3.1 Inside the Container: Restoring the vLLM Runtime Environment

The msModelSlim quantization phase depends on transformers==5.2.0 as required by Qwen3.6-27B, whereas the vLLM runtime requires transformers 4.x. Therefore, before inference, transformers must be restored to version 4.57.6, which is included in the image:

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ transformers==4.57.6
```

#### 2.3.2 Inside the Container: Preparing NPU Cards

The inference service requires two cards for tensor parallelism. Run the following commands to automatically select two idle cards:

```bash
# 1. Obtain the IDs of up to two idle cards
free_npus_raw=$(npu-smi info | grep -oE "No running processes found in NPU\s+[0-9]+" | head -n 2 | awk '{print $NF}')
npu_count=$(echo "$free_npus_raw" | wc -w)
# 2. Check whether the requirement for two cards is met and set the control environment variable. An error is reported if only one card is available or no idle card exists
if [ "$npu_count" -eq 2 ]; then
    export_val=$(echo "$free_npus_raw" | paste -s -d ',')
    export ASCEND_RT_VISIBLE_DEVICES=$export_val
    echo -e "\e[32m[PASS] Successfully exported ASCEND_RT_VISIBLE_DEVICES=$export_val\e[0m"
else
    echo -e "\e[31m[FAIL] Insufficient free NPUs (Found $npu_count, Need 2). Please release NPUs and try again.\e[0m" >&2
fi
```

If `[PASS]` is output, idle NPU cards have been automatically selected and you can proceed to the next step. If `[FAIL]` is output, release the NPU resources first and then retry the preceding command.

#### 2.3.3 Inside the Container: Starting the Service

Start the vLLM-Ascend online inference service (the command keeps occupying the current terminal):

```bash
vllm serve ~/qwen36_27b_w8a8 \
    --port 5678 \
    --served-model-name Qwen3.6-27B-W8A8 \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

>[!NOTE]
>
> **Knowledge point (optional reading): Description of the main vLLM startup parameters**
>
> - `--quantization ascend`: specifies the Ascend quantization inference backend used to load the W8A8 weights generated by msModelSlim.
> - `--served-model-name`: the model name exposed by the service (in this example, `Qwen3.6-27B-W8A8`). After the service starts, the `model` field in client requests must be consistent with this value; otherwise, the request reports that the model does not exist.
> - `--tensor-parallel-size 2`: specifies the degree of tensor parallelism, sharding the model across 2 NPU cards.
> - `--max-model-len 8192`: the maximum sequence length (number of tokens). Requests exceeding this length will be rejected.
> - `--compilation-config`: enables the FULL_DECODE_ONLY graph mode, which compiles the Decode stage into a static graph to accelerate inference.
> - `--additional-config`: enables CPU-NPU NUMA affinity binding to reduce Host-Device communication latency.

Startup takes approximately 4 to 5 minutes and produces a large amount of logs. The main time-consuming stages are as follows (the durations are measured values from a single test and are for reference only).

| No. | Stage | Duration     | Description |
|:--:|:--|:-------|:--|
| 1  | Configuration parsing and plugin activation | About 10 seconds | Loads the vLLM-Ascend platform plugin and parses the model architecture and scheduling parameters. |
| 2  | Worker startup and HCCL handshake | About 50 seconds | Starts multi-card Worker processes, establishes communication links, and assigns TP ranks. |
| 3  | CPU-NPU affinity binding | About 10 seconds | Binds Workers to NPU-proximal CPU cores and interrupts according to the NUMA topology. |
| 4  | Model weight loading | About 30 seconds | Loads the quantized weight shards (about 30 GB, evenly distributed across 2 cards, about 15 GB per card) into global memory. |
| 5  | Graph compilation and operator fusion | About 80 seconds | Performs Dynamo bytecode conversion (20s) + CANN operator compilation (48s) + fusion warm-up. |
| 6  | NPU Graph capture | About 30 seconds | Precompiles static execution paths for 22 batch sizes (1 to 152). |

When logs similar to the following appear, the service has started successfully:

```text
(APIServer pid=6036) INFO:     Started server process [6036]
(APIServer pid=6036) INFO:     Waiting for application startup.
(APIServer pid=6036) INFO:     Application startup complete.
```

> Some Warning logs may appear during startup. They can be ignored as long as the preceding success log appears. For details, see [Is It Normal to See Warning Logs When Starting vLLM?](#43-is-it-normal-to-see-warning-logs-when-starting-vllm).

If the service startup is aborted and the preceding success log does not appear, locate the problem based on the error message output by the terminal. Common errors and their handling methods are as follows:

1. Port occupied: The error message contains `Address already in use` or `bind: address already in use`. Change the port using `--port`, or terminate the process that is using this port.
2. Insufficient device memory (OOM): The error message contains something like `Out of memory`. You can run `npu-smi info` to check the device memory usage of the target card, and use `ASCEND_RT_VISIBLE_DEVICES` to switch to an idle card.

#### 2.3.4 Inference Verification

Because the vLLM service continues to occupy the current terminal after startup, open a new terminal on the host machine. You can run the following commands either inside or outside the container because the network is accessible in both environments:

**Step 1: Send a warmup request**.

```bash
curl -s http://localhost:5678/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-27B-W8A8", "prompt": "This is a warm-up request.", "max_tokens": 256}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

The first response may be slow or return garbled characters. This is normal for the first inference after the service starts. Wait for the response, which can be ignored. For the reason, see the explanation below (optional reading).

>[!NOTE]
>
> **Key point (optional reading): Reasons why the first request takes a long time or returns garbled characters**
>
> This behavior is caused by several one-time initialization operations triggered by the first inference after the service starts. It mainly includes the following three aspects:
>
> 1. First execution of the NPU Graph: The static graph path does not execute the complete data flow until the first actual inference, and the log shows the `Replaying aclgraph` message.
> 2. JIT compilation of Triton operators: Operators such as FlashAttention perform dynamic compilation and auto tuning during their first execution, causing a delay of several seconds.
> 3. Dirty data in the KV Cache: After global memory is preallocated, it is not cleared byte by byte. The first attention computation may read invalid data, resulting in garbled output.
>
> The industry commonly adopts the "**Warmup**" mechanism to address such issues. After the service starts, a test request is sent and its result is discarded; production traffic is accepted only after system initialization is complete.

**Step 2: Send a formal inference request**.

```bash
curl -s http://localhost:5678/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-W8A8", "prompt":"Write a Python function to calculate the Fibonacci sequence.", "max_tokens":256}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

Wait for the inference to complete. If the response includes the reasoning process for writing code or contains code output, the inference is successful.

### 2.4 Cleaning Up Resources

#### 2.4.1 Stopping the Inference Service and Releasing NPU Card Resources

Press `Ctrl+C` in the vLLM startup terminal to stop the service and release the occupied NPU card resources. Run `npu-smi info` to confirm that the NPU cards have no unexpected utilization, and manually kill any residual processes if necessary.

#### 2.4.2 (Optional) Deleting the Container to Free Up Disk Space

If the container environment in this tutorial is no longer needed, run the following command on the **host machine** to select and delete the target container to free up disk space:

```bash
~/ctr_in.py -d
```

🎉 At this point, you have completed the full workflow of one-click quantization with msModelSlim and inference deployment with vLLM-Ascend. To learn about more features, see advanced documents such as *[User Guide](../user_guide/README.md)*.

## 3. Appendix: Solutions for Intranet Environments Without Public Network Access

### 3.1 Obtaining Docker Images in an Isolated Intranet

**Solution 1: Configuring a Docker Proxy for Direct Pulling**

Applicable to most Linux distributions with Docker version ≥ 18.0 (compatibility is not guaranteed in all scenarios; adjust according to the actual environment if issues occur).

Edit the Docker service proxy configuration file `/etc/systemd/system/docker.service.d/http-proxy.conf` (replace the username, password, proxy address, and port according to the actual environment).

```text
[Service]
Environment="HTTP_PROXY=http://username:password@proxy.example.com:8080"
Environment="HTTPS_PROXY=http://username:password@proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

After saving, reload and restart the Docker service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

After that, you can run `docker pull` normally.

**Solution 2: Importing the Image Offline**

If the proxy solution is not feasible, first run [Host Machine: Automatically Identifying and Configuring Image Environment Variables](#212-host-machine-automatically-identifying-and-configuring-image-environment-variables) on the intranet NPU server and record the complete value of `MY_STUDY_VAR_VLLM_IMAGE`. Then, on a relay machine that has public network access and the same CPU architecture, run:

```bash
VLLM_IMAGE='Full image address'   # Replace with the value of MY_STUDY_VAR_VLLM_IMAGE
docker pull "${VLLM_IMAGE}"
docker save -o vllm-ascend.tar "${VLLM_IMAGE}"
```

Transfer `vllm-ascend.tar` to the intranet server via a USB drive or other means, and then load it:

```bash
docker load -i vllm-ascend.tar
docker images | grep vllm-ascend
```

After loading is complete, continue to [Transferring the Container Startup Script](#32-transferring-the-container-startup-script) to transfer the startup script, and then return to [Host Machine: Starting the Container](#215-host-machine-starting-the-container) to start the container. If you have switched to a new host machine shell session, re-execute [Host Machine: Automatically Identifying and Configuring Image Environment Variables](#212-host-machine-automatically-identifying-and-configuring-image-environment-variables) to restore the environment variables.

### 3.2 Transferring the Container Startup Script

Open the following link in a browser with public network access, download the `ctr_in.py` script, and transfer it to the `~/` directory of the intranet server:

```text
https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py
```

Run the following commands on the host machine of the intranet server:

```bash
cd ~
chmod +x ctr_in.py
ls -l ctr_in.py
```

After confirming that the file exists and has execution permissions, return to [Host Machine: Starting the Container](#215-host-machine-starting-the-container) to start the container.

### 3.3 Installing Python Dependencies Offline

Use the intranet pip source to install dependencies whenever possible. If no intranet software source is available, download the required installation packages in a relay environment that has public network access and matches the intranet NPU server in both CPU architecture and Python version, as follows:

```bash
mkdir -p offline_wheels
python3 -m pip download <package_name> --dest offline_wheels
```

Transfer the `offline_wheels` directory to the intranet server, copy it to the user home directory inside the container, and then execute the following inside the container:

```bash
pip3 install --no-index --find-links="${HOME}/offline_wheels" <package_name>
```

After the installation is complete, return to [Inside the Container: Verifying Environment Installation](#217-inside-the-container-verifying-environment-installation) to run the verification command. There is no need to run the online installation command again.

### 3.4 Obtaining Model Weights Offline

If the environment where the container resides has no public network access and the model weights cannot be downloaded directly, run the following command on a machine that has public network access:

```bash
modelscope download --model Qwen/Qwen3.6-27B --local_dir ./qwen36_27b_base
```

After the download is complete, transfer the entire `qwen36_27b_base` directory to the intranet server, and copy it into the container on the host machine by running the following command:

```bash
docker cp ./qwen36_27b_base <container_name>:/root/qwen36_27b_base
```

Replace `<container_name>` with the actual container name (you can view it by running `docker ps`). After the copy is complete, run `ls ~/qwen36_27b_base` inside the container to confirm that it contains files such as `config.json` and `model-*.safetensors`, and then return to [Inside the Container: Preparing NPU Cards](#222-inside-the-container-preparing-npu-cards) to continue with quantization.

## 4. FAQs

### 4.1 How Do I Re-enter the Container After Exiting It?

On the host machine, choose either of the following methods:

**Method 1 (recommended): Use the container startup script**.

```bash
~/ctr_in.py
```

Select the target container as prompted. If only one container is running, the script enters it automatically.

**Method 2: Use the native Docker command.**

```bash
docker exec -it alice_YYMMDD_HHMMSS bash
```

Replace `alice_YYMMDD_HHMMSS` with the actual container name. You can run `docker ps` first to view it.

### 4.2 What Should I Do if "Permission Denied" Is Displayed When Running Docker Commands?

The current user may not have been added to the Docker user group. Run the following command on the host machine with root privileges:

```bash
sudo usermod -aG docker "${USER}"
```

After running the command, log out of the current session and log in again, or run `newgrp docker` to make the user group change take effect immediately. Then run `docker ps` to verify.

> The Docker user group has high system privileges. Add only trusted users to this group, and avoid routinely operating as root.

### 4.3 Is It Normal to See Warning Logs When Starting vLLM?

If `Application startup complete` appears in the final output, the WARNING logs during startup do not affect functionality. They mainly fall into the following categories:

1. **GPU-specific parameter reset**: Parameters such as `--disable-cascade-attn` and `--disable-flashinfer-prefill` apply only to NVIDIA GPUs. In the Ascend environment, vLLM automatically resets them to `False` and ignores them.
2. **FULL_DECODE_ONLY graph mode risk warning**: This mode is experimental, and the warning indicates that capturing too many batches may cause insufficient device memory. If `Application startup complete` appears in the final output, graph capture succeeded and the service can be used normally.
3. **CUDA Graph capture limit**: The message `Capping cudagraph capture sizes` indicates that the system automatically adjusted the maximum captured batch size based on the available Mamba cache blocks, which is normal adaptation behavior.
4. **Gloo communication fallback**: The message `Unable to resolve hostname` indicates that the Gloo communication library cannot resolve the hostname and has automatically fallen back to the loopback address, which does not affect single-machine multi-card inference.

### 4.4 What Should I Do If the Model Weight Download Fails or Is Interrupted?

If the download fails or is interrupted, take the following actions:

1. Network unavailable or timeout: run `curl -I https://www.modelscope.cn` to check network connectivity. If you are on an enterprise intranet, configure a proxy (for example, `export HTTPS_PROXY=http://proxy.example.com:8080`) and then retry, or obtain the weights offline instead (see [Obtaining Model Weights Offline](#34-obtaining-model-weights-offline)).
2. Insufficient disk space: run `df -h` to confirm that the remaining space is greater than 100 GB, clean up unnecessary files, and then download again.
3. The download is interrupted: run the download command again to resume the download. If it fails repeatedly, delete the `~/qwen36_27b_base` directory and retry.
