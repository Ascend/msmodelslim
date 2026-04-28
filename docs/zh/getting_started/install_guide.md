# msModelSlim工具安装指南

## 安装说明

本文介绍msModelSlim工具的安装，当前工具支持源码编译安装和pip安装两种方式。

## 安装前准备

准备python环境：需要 Python 3.8 或更高版本。

## 源码编译安装

>[!NOTE]  
>使用 `msModelSlim` 命令行工具时，请勿在 `msModelSlim` 的源码目录下直接运行命令。这可能会因 Python 在导入模块时出现源码路径和安装路径冲突，导致命令执行失败。  
>若安装 `msmodelslim` 时遇到报错，请先查阅 [FAQ](../appendix/faq.md) 寻找解决方案。如问题仍未解决，欢迎提交 [Issue](https://gitcode.com/Ascend/msmodelslim/issues)，并附上您的运行环境及完整的错误日志，我们将尽快为您排查。

### 基于Atlas A2 训练、推理产品，Atlas A3 训练、推理系列产品安装

```shell
# 1.git clone msmodelslim代码
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2.进入到msmodelslim的目录并运行安装脚本
cd msmodelslim
bash install.sh
```

### 基于Atlas 300I Duo 系列产品安装

>[!NOTE]  
> 1.Atlas 300I Duo 卡仅支持单卡单芯片处理器量化。  
> 2.如果需要进行稀疏量化和压缩，则需要安装8.2.RC1及以上版本CANN，请参考[CANN快速安装](https://www.hiascend.com/cann/download)安装昇腾NPU驱动和CANN软件（包括Toolkit和ops包）并配置环境变量。

```shell
# 1.git clone msmodelslim代码
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2.进入到msModelSlim的目录并运行安装脚本
cd msmodelslim
bash install.sh

# 注：如果需要进行稀疏量化和压缩，则继续以下操作。
# 3.进入python环境下的site_packages包管理路径，其中${python_envs}为Python环境路径。
cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/  
# 以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 4.编译weight_compression组件，其中${install_path}为CANN软件的安装目录。
sudo bash build.sh ${install_path}/ascend-toolkit/latest

# 5.上一步编译操作会得到build文件夹，给build文件夹相关权限
chmod -R 550 build
```

## pip安装

pip安装包括从**PyPI远程安装**和**下载whl包本地安装**两种方式。

### PyPI远程安装

```bash
pip install msmodelslim
```

### 下载whl包安装

请参考[版本说明](../appendix/release_notes.md)中的“whl包获取”章节，下载msModelSlim的whl软件包。

获取到whl软件包后执行如下命令进行安装。

```bash
sha256sum {name}.whl # 验证whl包，若校验码一致，则whl包在下载中没有受损
```

```bash
pip install ./msmodelslim-{version}-py3-none-any.whl # 安装whl包
```

## 安装后配置

如果是昇腾NPU设备，请参考如下配置

### CANN安装

请参考[CANN快速安装](https://www.hiascend.com/cann/download)安装昇腾NPU驱动和CANN软件（包括Toolkit和ops包）并配置环境变量。

### PTA安装

PyTorch安装请参考《[Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)》配置与安装。

## 卸载

```shell
pip uninstall msmodelslim -y
```

## 升级

若需要升级工具，可参照[源码编译安装](#源码编译安装)、[PyPI远程安装](#pypi远程安装)或[下载whl包安装](#下载whl包安装)中的步骤直接安装新版本，新版本会覆盖原有的版本。
