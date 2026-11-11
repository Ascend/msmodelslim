# QuantConfig

## 功能说明

量化参数配置类，保存量化过程中配置的参数。

## 函数原型

```python
QuantConfig(w_bit=8, a_bit=8, w_signed=True, a_signed=False, w_sym=True, a_sym=False, input_shape=None, act_quant=True, act_method=0, quant_mode=0, disable_names=None, amp_num=0, squant_mode='squant' , keep_acc=None, sigma=25, is_fp=False, disable_first_layer=True, disable_last_layer=True, is_optimize_graph=True, is_dynamic_shape=False, use_onnx=True, num_input=0, quant_param_ops=None, atc_input_shape=None, graph_optimize_level=0, shut_down_structures=None, device_id=0, om_method='aoe')
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| w_bit | 输入 | 权重量化bit。| 可选。<br>数据类型：int。<br>默认为8，暂不支持其他bit量化，不支持修改。|
| a_bit | 输入 | 激活层量化bit。| 可选。<br>数据类型：int。<br>默认为8，暂不支持其他bit量化，不支持修改。|
| w_signed | 输入 | 是否对权重进行符号量化。| 可选。<br>数据类型：bool。<br>默认为True。|
| a_signed | 输入 | 是否对激活值进行符号量化。| 可选。<br>数据类型：bool。<br>默认为False。<br>使用relu的CV模型建议设置为False，其他模型建议设置为True。|
| w_sym | 输入 | 权重是否对称量化。| 可选。<br>数据类型：bool。<br>默认为True。|
| a_sym | 输入 | 激活值是否对称量化。| 可选。<br>数据类型：bool。<br>默认为False。|
| input_shape | 输入 | 当输入模型支持动态shape时，用户需指定input_shape参数，用以生成量化时的校正数据。| 可选，当模型支持动态shape时必须指定。<br>数据类型：list [list [int]]。<br>默认值：None <br>当模型有多个输入时，按照顺序指定input_shape，例如：\[[1, 3,224, 224], [1, 3, 640, 640]]。|
| act_quant | 输入 | 是否对激活值进行量化。| 可选。<br>数据类型：bool。<br>默认为True。<br>暂不支持修改。|
| act_method | 输入 | 激活值量化方法。| 可选。<br>数据类型：int。<br>可选值[0,1,2]，默认为0。<br>(1) 0代表Data-Free场景的量化方法（具体由sigma参数决定）<br>(2) 1代表Label-Free场景的min-max observer方法。Label-Free场景推荐选1。<br>(3) 2代表Label-Free场景的histogram observer方法。|
| quant_mode | 输入 | 量化模式。| 可选。<br>数据类型：int。<br>可选值为[0,1]，默认为0。<br>(1)0代表Data-Free量化模式。<br>(2)1代表Label-Free量化模式。|
| disable_names | 输入 |需排除量化的节点名称，即手动回退的量化层名称。<br>如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等。| 可选。<br>数据类型：list[str]。<br>默认值[]。|
| amp_num | 输入 |混合精度量化回退层数。<br>精度降低过多时，可增加回退层数，推荐优先回退3~7层，如果精度恢复不明显，再增加回退层数。| 可选。<br>数据类型：int。<br>默认为0。|
| squant_mode | 输入 | 量化方式。| 可选。<br>数据类型：String。<br>支持配置为默认值'squant' （Data-Free量化算法），暂不支持修改 。|
| keep_acc | 输入 | 精度保持策略。<br>(1)admm和round_opt是用来改善权重量化，减少权重量化误差，推荐在Label-Free模式下使用，适当改善量化效果。<br>(2)easy_quant用来改善激活量化，减少激活量化误差，推荐在Label-Free模式下使用，通常能够起到较好的改善效果。| 可选。<br>数据类型：dict。<br>包含以下三种精度保持策略：<br>(1)admm策略：数据类型[bool, int]，bool配置是否开启，int配置优化迭代次数。(2)easy_quant：（推荐）数据类型[bool, int]，bool配置是否开启，int配置优化迭代次数。(3)round_opt：数据类型[bool]，bool配置是否开启。<br>输入模板为：keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False} 。|
| sigma | 输入 |Label-Free的量化统计方法。<br>建议输入值为0或25，卷积类模型使用sigma统计效果更好，transformers类模型min-max统计效果更好。| 可选。<br>数据类型：int。<br>默认为25。<br>(1)sigma=25时，使用sigma统计方法。<br>(2)sigma=0时，使用Min-Max统计方法。|
| is_fp | 输入 |是否启用逐层量化校准。| 可选。<br>数据类型：bool。<br>默认值为False，暂不支持修改。|
| disable_first_layer | 输入 |是否自动回退首层量化层。| 可选。<br>数据类型：bool。<br>默认为True。|
| disable_last_layer | 输入 |是否自动回退尾层量化层。| 可选。<br>数据类型：bool。<br>默认为True。|
| is_optimize_graph | 输入 |是否进行图优化。| 可选。<br>数据类型：bool。<br>默认为True。|
| is_dynamic_shape | 输入 |指定输入的模型是否支持动态shape。| 可选。输入模型支持动态shape时，另一配置参数input_shape也必须指定。<br>数据类型：bool。<br>默认为False。<br>True：输入的模型支持动态shape。False：输入的模型为静态shape。|
| use_onnx | 输入 |指是否使用onnx_runtime进行量化校准（onnx_runtime仅支持<2GB的模型量化），若模型大于2GB，建议关闭该参数，使用ACL进行校准。| 可选。<br>数据类型：bool。<br>默认为True。<br>True：onnx_runtime量化校准。False：ACL量化校准。|
| num_input | 输入 |网络输入数据的数量。| 可选。<br>数据类型：int。<br>默认为0，若use_onnx配置为False，则必须手动输入模型输入数据的数量。|
| quant_param_ops | 输入 | 选择需要量化的网络层。| 可选。<br>数据类型：list。<br>默认值：['Conv', 'Gemm', 'MatMul']。<br>若使用ACL量化校准辅助量化时（即use_onnx配置为False），则该参数需配置为['Conv']。|
| atc_input_shape | 输入 | ATC工具转om模型的输入数据shape。|<br>可选。<br>数据类型：String。<br>默认为None。<br>若use_onnx配置为False，则必须手动输入模型的输入shape，输入格式要求如下：<br>(1)若模型为单个输入，则shape信息为"input_name:n,c,h,w"；指定的节点必须放在双引号中。<br>(2)若模型有多个输入，则shape信息--input_shape="input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"；不同输入之间使用英文分号分隔，input_name必须是转换前的网络模型中的节点名称。|
| graph_optimize_level | 输入 | 图优化级别。| 可选。<br>数据类型：int。<br>取值如下：<br>(1)0：默认为0，对浮点模型和量化后模型都不进行图优化。<br>(2)1：只对浮点模型进行图优化。<br>(3)2：对浮点模型和量化后的模型都进行图优化。|
| shut_down_structures | 输入 | 需关闭的图优化结构列表。| 可选。<br>数据类型：list。<br>默认为None，即对所有可优化结构进行量化。<br>取值范围：['ChangeGAPCONVOptimization', 'ChangeResizeOptimization', 'CombineMatmulOptimization', 'DeleteConcatOptimization', 'DoubleFuseBatchNormOptimization', 'DoubleReshapeOptimization', 'FastClipOptimization', 'FuseBatchNormOptimization', 'FuseDivMatmulOptimization', 'GeluErf2FastGeluOptimization', 'GeluErf2SigmoidOptimization', 'GeluErf2TanhOptimization', 'GeluTanh2SigmoidOptimization', 'LayerNormOptimization', 'Matmul2GemmOptimization', 'PatchMerging2ConvOptimizationV0', 'PatchMerging2ConvOptimizationV1', 'PatchMerging2ConvOptimizationV2', 'PatchMerging2ConvOptimizationV3', 'RemoveDoubleResizeOptimization', 'ReplaceAscendQuantOptimizationV1', 'ReplaceAscendQuantOptimizationV2', 'ReplaceConcatQuantOptimizationV1', 'ReplaceConcatQuantOptimizationV2', 'ReplaceConcatQuantOptimizationV3', 'ReplaceConcatQuantOptimizationV4', 'ReplaceConcatQuantOptimizationV5', 'ReplaceConcatQuantOptimizationV6', 'ReplaceConcatQuantOptimizationV7', 'ReplaceConcatQuantOptimizationV8', 'ReplaceConcatQuantOptimizationV9', 'ReplaceHardSigmoidOptimization', 'ReplaceLeakyReluOptimization', 'ReplaceMaxPoolBlockOptimizationV1', 'ReplaceMaxPoolBlockOptimizationV2', 'ReplaceRelu6Optimization', 'ReplaceReluOptimization', 'ReplaceReshapeTransposeOptimizationV1', 'ReplaceReshapeTransposeOptimizationV2', 'ReplaceReshapeTransposeOptimizationV3', 'ReplaceResizeQuantOptimization', 'ReplaceSigmoidOptimizationV1', 'ReplaceSigmoidOptimizationV2', 'ReplaceSoftmaxOptimizationV1', 'ReplaceSoftmaxOptimizationV2', 'Resize2ConvTransposeOptimization', 'SimplifyShapeOptimization', 'SimplifyShapeOptimizationV2'] 。|
| device_id | 输入 | 昇腾AI处理器的DEVICE ID。| 可选。<br>数据类型：int。<br>取值范围[0,7]，默认值为0。|
| om_method | 输入 | onnx模型转换为om模型的方式。| 可选。<br>数据类型：String。<br>支持配置为'aoe'和'atc'，默认为'aoe'，即通过aoe工具进行转换。|

## 调用示例

```python
from msmodelslim.onnx.squant_ptq import QuantConfig
config = QuantConfig(disable_names=[],
                       quant_mode=0,
                     amp_num=0,
                     a_sym=True,
                     keep_acc={'admm': [False, 1000], 'easy_quant': [True, 1000], 'round_opt': False},
                     disable_first_layer=True,
                     disable_last_layer=True
)
```
