# msModelSlim Tool Installation Guide

<!-- md-trans-meta sourceCommit=652caa58cbf1feddc29ea6bd484adddf6557db00 translatedAt=2026-08-21T03:31:35.580Z pushedAt=2026-08-21T03:32:03.130Z -->

## 1. Installation Notes

1. This tool supports three installation methods: [Online Installation](#21-online-installation), [Offline Installation](#22-offline-installation), and [Source Installation](#23-source-installation). Select the most suitable option based on your actual environment.

2. The Python version required by this tool must be no earlier than 3.8 and no later than 3.12.

3. If you use Ascend NPU devices, you need to install TorchNPU and its related dependencies. For the installation of the TorchNPU package, see [Ascend for PyTorch Installation](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/installation_guide/installation_via_binary_package.md).

## 2. Installation Methods

### 2.1 Online Installation

If your device has Internet access, you can complete the download and installation of the tool automatically with a single command. See the MindStudio [Download](https://www.hiascend.com/en/developer/software/mindstudio/download?versionId=150&ids=45%2C42671e3507254c1fb6f56b379948506f%2C129%2C49%2C) page on the Ascend Community, select the corresponding CANN version, and choose **Online** as the installation method. The system will guide you through the subsequent operations.

### 2.2 Offline Installation

For devices in environments without external network access, such as enterprise intranets, first download the complete offline installation package on a machine with network access, and then transfer it to the target device for installation. See the MindStudio [Download](https://www.hiascend.com/en/developer/software/mindstudio/download?versionId=150&ids=45%2C42671e3507254c1fb6f56b379948506f%2C129%2C49%2C) page on the Ascend Community, select the corresponding CANN version, and choose "Offline" as the installation method to obtain the corresponding installation package and operation instructions.

### 2.3 Source Installation

**The source build and installation steps are as follows:**

1. Clone the msmodelslim code using git.

   ```shell
   git clone https://gitcode.com/Ascend/msmodelslim.git -b 26.1.0
   ```

2. Go to the msmodelslim directory and run the installation script.

   ```shell
   cd msmodelslim
   bash install.sh
   ```

   When the following information is printed, msModelSlim has been built and installed successfully.

   ```ColdFusion
   Successfully installed msmodelslim-{version}
   ```

**If sparse quantization and compression are required, install CANN 8.2.RC1 or later. After the source build and installation is complete, continue with the following operations:**

1. Go to the `site-packages` package management path in the Python environment, where `${python_envs}` is the Python environment path.

   ```shell
   cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
   # The following uses /usr/local/ as the user directory and Python 3.11.10 as an example
   cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
   ```

2. Compile the weight_compression component, where `${install_path}` is the installation directory of the CANN software.

   ```shell
   sudo bash build.sh ${install_path}/ascend-toolkit/latest
   ```

   The following information is printed, indicating that the compilation is successful and the build folder is generated.

   ```ColdFusion
   [100%] Built target compress_excutor
   ```

3. The `build` folder is generated in the previous compilation step. Grant the relevant permissions to the build folder.

   `chmod -R 550 build`

>[!NOTE]
>
> 1. When using the `msModelSlim` command-line tool, do not run commands directly in the source code directory of `msModelSlim`. This may cause command execution failure due to conflicts between the source code path and the installation path when Python imports modules.
> 2. If an error occurs during the installation of `msModelSlim`, first refer to *[FAQs](../support/faq.md)* to find a solution. If the problem remains unresolved, you are welcome to submit an [Issue](https://gitcode.com/Ascend/msmodelslim/issues) with your runtime environment and complete error logs, and we will troubleshoot it as soon as possible.
> 3. Currently, only the Atlas 300I Duo series products support compression after sparse quantization.

## 3. Installation Verification

After the installation is complete, run the following commands to verify whether the tools are installed successfully:

```shell
msmodelslim --help
```

If the output does not report an error and the help information is displayed, the installation is successful.

## 4. Uninstallation

You can perform uninstallation  by following these steps:

1. Download the script.

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.1.0/ms_install.py
   ```

   >[!NOTE]
   >
   > - An internet connection is required for downloading. If the environment does not allow internet access or is offline, download the script in an environment with internet access first and then copy it to the target device.
   > - If the command does not respond or errors such as connection failure or SSL certificate errors occur, see [FAQs](https://www.hiascend.com/developer/blog/details/02176213671719317003).

2. Perform the uninstallation.

   ```bash
   python ms_install.py uninstall {tools_name}
   ```

   Where `{tools_name}` is configured as the name of the tool to be uninstalled. You can query it by running the `python ms_install.py help` command. The tool name is displayed under the `Available Tools` field in the printed information.

   If the uninstallation is successful, the following information is printed:

   ```ColdFusion
   Successfully uninstalled 1 tool ({tools_name})
   ```

## 5. Upgrade

Upgrade means "uninstall first and then install". Directly run the installation command, and the tool will automatically uninstall the old version and guide you through the overwrite installation.<br>
You can run the `pip show msmodelslim` command to view the version information of the current environment, and then select the version to upgrade to. When upgrading, pay attention to the version compatibility. For details, see *[Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md)*.
