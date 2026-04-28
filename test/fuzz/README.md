# 使用 Atheris 进行 Python fuzz 测试

## 基本步骤

- fuzz 测试主要用于挖掘接口漏洞，不太需要在代码改动时每次都做检查，可以在迭代快结束或者其他有需要的时候手动执行
- 推荐在 x86 的服务器上执行测试，arm 服务器上需要自己编译 llvm，比较麻烦

```sh
#安装 atheris
pip3 install atheris coverage

# 从automl 项目路径执行 fuzz 测试，因为依赖 `test/resources/sample_net_mindspore.py` 定义 mindspore 模型
./test/fuzz/run_fuzz.sh
# >>>> ABS_DIR_PATH: {automl_path}/test/fuzz
# >>>> TEST_PATH: {automl_path}/test/fuzz/mindspore_quant_api/create_quant_config/config_file
# INFO: Instrumenting test
# ...
# Done 1000 in 13 second(s)
#
# >>>> TEST_PATH: {automl_path}/test/fuzz/mindspore_quant_api/save_model/file_name
# INFO: Instrumenting test
# ...
# Done 1000 in 131 second(s)
#
# >>>> Done fuzz, generating coverage result...
# Wrote HTML report to automl_fuzz_coverage/index.html
```

- 执行完毕后的结果文件包括：
  - `automl_fuzz_test_save_path` 测试过程中产生的临时文件，没有删除，用于检查实际生成的文件名是否合法
  - `automl_fuzz_coverage` 测试覆盖率结果
