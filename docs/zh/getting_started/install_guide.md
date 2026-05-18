# msModelSlim工具安装指南

## 1. 安装说明

安装本工具前需要安装CANN，具体操作请参见《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit和ops包），并配置环境变量。

本工具支持[在线安装](#21-在线安装)、[离线安装](#22-离线安装)、[源码安装](#23-源码安装)三种安装方式，请根据您的实际环境选择最合适的方案。

## 2. 安装方式

### 2.1 在线安装

若您的设备具备互联网访问能力，可通过一条命令自动完成工具的下载与安装。请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择对应的CANN版本，并在安装方式中选择“在线安装”，系统将引导您完成后续操作。

### 2.2 离线安装

对处于企业内网等无外网环境的设备，请先在可联网的机器上下载完整的离线安装包，再将其传输至目标设备进行安装。请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择对应的CANN版本，并在安装方式中选择“离线安装”，获取对应的安装包及操作指引。

### 2.3 源码安装

#### 2.3.1 安装前准备

准备Python环境：需要 Python 3.8 或更高版本。

#### 2.3.2 源码构建安装

```shell
# 1.git clone msmodelslim代码。
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2.进入到msmodelslim目录并运行安装脚本。
cd msmodelslim
bash install.sh
# 打印如下信息时，表示msModelSlim构建安装成功。
Successfully installed msmodelslim-{version}

# 注：如果需要进行稀疏量化和压缩，需要安装CANN 8.2.RC1及以上版本，并继续以下操作。
# 3.进入python环境下的site_packages包管理路径，其中${python_envs}为Python环境路径。
cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/  
# 以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例。
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 4.编译weight_compression组件，其中${install_path}为CANN软件的安装目录。
sudo bash build.sh ${install_path}/ascend-toolkit/latest
# 打印如下信息，表示编译成功，生成build文件夹。
build created successfully

# 5.上一步编译操作会得到build文件夹，给build文件夹相关权限。
chmod -R 550 build
```

>[!NOTE]
>
> 1. 使用 `msModelSlim` 命令行工具时，请勿在 `msModelSlim` 的源码目录下直接运行命令。这可能会因 Python 在导入模块时出现源码路径和安装路径冲突，导致命令执行失败。  
> 2. 若安装 `msmodelslim` 时遇到报错，请先查阅 [FAQ](../appendix/faq.md) 寻找解决方案。如问题仍未解决，欢迎提交 [Issue](https://gitcode.com/Ascend/msmodelslim/issues)，并附上您的运行环境及完整的错误日志，我们将尽快为您排查。

## 3. 卸载

可通过如下步骤卸载：

1. 下载脚本。

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.0.0/ms_install.py
   ```

   > [!NOTE]
   >
   > - 需要联网环境才能下载，若环境不允许联网或离线状态，请先在可联网的环境下载该脚本后拷贝到目标设备。
   > - 若执行命令无响应或出现连接失败、SSL证书错误等问题，请参见[FAQ](https://www.hiascend.com/developer/blog/details/02176213671719317003)。

2. 执行卸载。

   ```bash
   python ms_install.py uninstall {tools_name}
   ```

   其中{tools_name}配置为需要卸载的工具名称，可通过`python ms_install.py help`命令查询，在打印信息中的Available Tools字段下显示工具名称。

   卸载成功打印如下信息：

   ```ColdFusion
   Successfully uninstalled 1 tool ({tools_name})
   ```

## 4. 升级

升级即“先卸后装”。直接执行安装命令，工具将自动卸载旧版本，并引导您完成覆盖安装。
