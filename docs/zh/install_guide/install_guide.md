# msModelSlim 安装指南

## 1. 安装说明

1. 本工具支持[在线安装](#21-在线安装)、[离线安装](#22-离线安装)、[源码安装](#23-源码安装)三种安装方式，请根据您的实际环境选择最合适的方案。

2. 本工具依赖的 Python 版本不低于3.8，且不高于3.12。

3. 若使用昇腾NPU设备，则需安装TorchNPU及其对应的相关依赖，TorchNPU包的安装请参考《[Ascend for PyTorch 安装](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/installation_guide/quick_install.md)》。

## 2. 安装方式

### 2.1 在线安装

若您的设备具备互联网访问能力，可通过一条命令自动完成工具的下载与安装。请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download?versionId=142&ids=45%2C26958bcc909e4cd48fa56d4c4a43ebec%2C90%2C49)页面，选择对应的CANN版本，并在安装方式中选择“在线安装”，系统将引导您完成后续操作。

### 2.2 离线安装

对处于企业内网等无外网环境的设备，请先在可联网的机器上下载完整的离线安装包，再将其传输至目标设备进行安装。请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download?versionId=145&ids=45%2C26958bcc909e4cd48fa56d4c4a43ebec%2C90%2C50%2C)页面，选择对应的CANN版本，并在安装方式中选择“离线安装”，获取对应的安装包及操作指引。

### 2.3 源码安装

如需使用最新代码的功能，或对源码进行修改以增强功能，可下载本仓库代码，自行编译、打包工具并完成安装。

#### 2.3.1 环境准备

源码编译统一使用 MindStudio 标准构建环境。

- 日常开发或使用已发布镜像，请参考《[MindStudio工具开发环境安装指导](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/dev_env_setup.md)》。
- 需要从基础操作系统复现环境、执行源码构建验证或单元测试验证时，必须参考《[MindStudio统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》，从openEuler基础镜像现场构建环境镜像。

本文档后续的源码编译和单元测试命令，均在上述指定镜像容器或现场构建的环境镜像的容器中执行，CANN 软件包版本、GCC 版本和 Python 版本以统一镜像制作指南为准，本仓库不重复维护。

镜像构建完成后，必须使用统一镜像制作指南第 7 章给出的 `ctr_in.py` 命令，在交互式终端中启动并进入容器。不得使用普通 `docker run` 创建容器，也不得使用 `docker exec <容器名> bash -c '<命令>'` 替代交互式环境；否则可能跳过 Python、GCC 和 CANN 环境初始化。

进入 `ctr_in.py` 打开的交互式容器 Shell 后，执行如下命令克隆本仓库：

```bash
cd ~
git clone https://gitcode.com/Ascend/msmodelslim.git
```

#### 2.3.2 执行编译

保持在 `ctr_in.py` 打开的同一个交互式容器 Shell 中，在仓库根目录执行以下命令，自动完成依赖下载与构建：

```bash
cd ~/msmodelslim
python3 build.py
```

构建成功后，安装包将生成在 `artifacts/` 目录下。

#### 2.3.3 执行单元测试（可选）

此步骤非安装必需。如需验证代码基本功能，可执行单元测试：

```bash
cd ~/msmodelslim
python3 build.py test
```

命令返回码为 0，且测试用例均无失败，表示单元测试通过。

#### 2.3.4 安装

将 `artifacts/` 目录下安装包复制到目标设备，并执行以下命令安装：

```bash
pip install artifacts/msmodelslim*.whl
```

打印类似如下信息，表示安装成功：

```text
Successfully installed msmodelslim-{version}
```

#### 2.3.5 稀疏量化和压缩

若需进行稀疏量化和压缩，请安装CANN 8.2.RC1及以上版本，完成如上源码构建安装后，请继续以下操作：

1. 进入python环境下的site-packages包管理路径，其中${python_envs}为Python环境路径。

   ```shell
   cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
   # 以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例。
   cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
   ```

2. 编译weight_compression组件，其中${install_path}为CANN软件的安装目录。

   ```shell
   sudo bash build.sh ${install_path}/ascend-toolkit/latest
   ```

   打印如下信息，表示编译成功，生成build文件夹。

   ```ColdFusion
   [100%] Built target compress_excutor
   ```

3. 上一步编译操作会得到build文件夹，给build文件夹相关权限。

   `chmod -R 550 build`

>[!NOTE]
>
> 1. 使用 `msModelSlim` 命令行工具时，请勿在 `msModelSlim` 的源码目录下直接运行命令。这可能会因 Python 在导入模块时出现源码路径和安装路径冲突，导致命令执行失败。
> 2. 若安装 `msModelSlim` 时遇到报错，请先查阅《[FAQ](../support/faq.md)》寻找解决方案。如问题仍未解决，欢迎提交 [Issue](https://gitcode.com/Ascend/msmodelslim/issues)，并附上您的运行环境及完整的错误日志，我们将尽快为您排查。
> 3. 目前仅Atlas 300I Duo系列产品支持在稀疏量化后进行压缩。

## 3. 验证安装

安装完成后，执行以下命令验证工具是否安装成功：

```shell
msmodelslim --help
```

若输出不报错，且能显示帮助信息，则表明安装成功。

## 4. 卸载

可通过如下步骤卸载：

1. 下载脚本。

   ```bash
   curl -O https://inst.obs.cn-north-4.myhuaweicloud.com/26.1.0/ms_install.py
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

## 5. 升级

升级即“先卸后装”。直接执行安装命令，工具将自动卸载旧版本，并引导您完成覆盖安装。<br>
可通过`pip show msmodelslim`命令查看当前环境的版本信息，再选择需要升级的版本。升级版本时需要关注版本配套关系，请参见《[版本说明](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md)》。
