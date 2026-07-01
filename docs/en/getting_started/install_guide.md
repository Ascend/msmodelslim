# msModelSlim Installation Guide

## 1. Installation Description

Before installing this tool, you need to install CANN. For details, see [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the Ascend NPU driver and CANN software (including the Toolkit and ops), and configure environment variables.

This tool supports three installation methods: [Online Installation](#21-online-installation), [Offline Installation](#22-offline-installation), and [Installation from Source](#23-installation-from-source). Choose the method that best fits your environment.

## 2. Installation Methods

### 2.1 Online Installation

If your device has Internet access, you can run a single command to automatically download and install the tool. Visit the [Ascend community](https://www.hiascend.com/developer/software/mindstudio/download), select the target CANN version, and choose "Online Installation". The system will guide you through the subsequent operations.

### 2.2 Offline Installation

For devices that are not connected to the Internet, such as those on an enterprise intranet, download the complete offline installation package on a device that has Internet access and then transfer the package to the target device for installation. Visit the [Ascend community](https://www.hiascend.com/developer/software/mindstudio/download), select the target CANN version, and choose "offline installation". The system will guide you through the subsequent operations.

### 2.3 Installation from Source

#### 2.3.1 Preparing for the Installation

Prepare the Python environment: Python 3.8 or later is required.

#### 2.3.2 Building and Installation from Source

```shell
# 1. Clone the msModelSlim repository using git clone.
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2. Go to the msModelSlim directory and run the installation script.
cd msmodelslim
bash install.sh
# If the following information is displayed, msModelSlim is successfully installed.
Successfully installed msmodelslim-{version}

# Note: To perform sparse quantization and compression, install CANN 8.2.RC1 or later and proceed with the following operations.
# 3. Go to the `site_packages` package management directory under the Python environment, where ${python_envs} specifies the Python environment path.
cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/  
# In the following example, /usr/local/ is the user directory and the Python version is 3.11.10.
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 4. Build the weight_compression component, where ${install_path} specifies the installation directory of the CANN software.
sudo bash build.sh ${install_path}/ascend-toolkit/latest
# If the following information is displayed, the build is successful and the build directory is generated.
build created successfully

# 5. Grant the required permissions to the generated build directory.
chmod -R 550 build
```

>[!NOTE]
>
> 1. When using the `msModelSlim` command-line tool, do not run commands in the `msModelSlim` source code directory. Doing so may cause conflicts between the source code path and the installation path when Python imports modules, leading to command execution failures.
> 2. If you encounter errors when installing `msmodelslim`, see the [FAQ](../appendix/faq.md) for solutions. If the issue persists, submit an [issue](https://gitcode.com/Ascend/msmodelslim/issues) with your operating environment and complete error logs attached. We will troubleshoot the issue for you as soon as possible.

## 3. Uninstallation

To uninstall the tool, perform the following steps:

1. Download the script.

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.0.0/ms_install.py
   ```

   > [!NOTE]
   >
   > - Internet access is required to download the script. If your target environment is offline or does not allow internet access, download the script on an internet-connected device first, then copy it to the target device.
   > - If the command does not respond, or you encounter connection failures, SSL certificate errors, or other issues, refer to the [FAQ](https://www.hiascend.com/developer/blog/details/02176213671719317003).

2. Uninstall the tool.

   ```bash
   python ms_install.py uninstall {tools_name}
   ```

   Replace `{tools_name}` with the name of the tool to be uninstalled. You can run the `python ms_install.py help` command to query the tool name, which is displayed under the `Available Tools` field in the command output.

   If the uninstallation is successful, the following information is displayed:

   ```ColdFusion
   Successfully uninstalled 1 tool ({tools_name})
   ```

## 4. Upgrade

Upgrades follow the "uninstall first, then install" process. Simply run the installation command. The tool will automatically remove the previous version and guide you through the upgrade process.
