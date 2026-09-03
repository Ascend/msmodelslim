# 适配器注册指南

在 `config/config.ini` 中注册模型与入口。

## 机制（setup.py entry point 如何生效）

`install.sh` 走 `setup.py` 的 develop/editable 安装，把 `config.ini` 中的注册项生成为 Python **entry point**（console 插件），`msmodelslim` 运行期通过 `PluginModelFactory` 按 `--model_type` 的值查找插件：

- `[ModelAdapter]`：`<适配组> = <型号值1>, <型号值2>, ...` —— 每个型号值就是用户 `--model_type` 会传入的字符串（如 `Qwen3-4B`）。同一适配组的多个型号共享同一套适配器实现。
- `[ModelAdapterEntryPoints]`：`<适配组> = <loader 的 python 导入路径>:<Loader 类名>` —— loader 指向键即适配组名（与 `[ModelAdapter]` 的组名一致）；Loader 负责根据具体型号实例化对应的 `ModelAdapter`。
- `[ModelAdapterDependencies]`：`<适配组> = {"transformers": "==4.51.0"}` —— **语义是"依赖告警 + 包装"，不是硬失败**：环境版本不匹配时通常告警并尝试继续（或做轻量兼容包装），不会因版本锁直接拒载。不要误以为必须精确装到锁的版本；也不要因为"只是告警"就无视，运行期异常往往由此而来。

## 示例

```ini
[ModelAdapter]
my_model = MyModel-7B, MyModel-13B

[ModelAdapterEntryPoints]
my_model = msmodelslim.model.my_model.loader:MyModelLoader
```

## 注册 → 重装 → 验证配方

1. 新建适配器源码目录（如 `msmodelslim/model/my_model/`，含 `__init__.py`、`loader.py`、`model_adapter.py`）。
2. 在 `config/config.ini` 补上述三个小节（按需）的条目。
3. 执行 `bash install.sh` 重新安装，让 entry point 生效。
4. **验证：在仓库根目录之外**执行（源码树 import 遮蔽会假失败，见 installation 技能 F0-2）：
   - 查看注册是否生效：列出 entry points，确认新适配器出现在列表中且数量 +1；
   - 负向对照：用一个未注册型号触发，确认回落 `default` 适配器（证明正向命中不是 default 兜底）。

## 适配面提示（V0 vs V1）

`adaptation/calibration` 模板实现的是 `ModelSlimPipelineInterfaceV1` 五接口。同仓 dense 模型适配器普遍继承 `DefaultModelAdapter`，其中还含 **V0 兼容方法**（如 `load_model`、`handle_dataset_by_batch`）。是否补 V0 方法取决于下游走哪套配置：

- 调优/量化走 `modelslim_v1`（apiversion 为 `modelslim_v1`）→ 只需 V1 五接口；
- 若需消费 `modelslim_v0` 的 best-practice 配置（`lab_practice` 中的旧格式文件）或旧接口调用方 → 需补 V0 方法。

判断依据以实际要复用的配置/调用方为准，技能未在别处强制。
