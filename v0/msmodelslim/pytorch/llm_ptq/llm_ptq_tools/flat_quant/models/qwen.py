#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import (ModelStructureBridge, 
                                            AttnNormLinearPair, 
                                            AttnLinearLinearPair, 
                                            MLPNormLinearPair, 
                                            MLPLinearLinearPair,
                                            remove_after_substring)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_bridge_registry import (
    ConfigMatcher, 
    CompositeMatcher, 
    ModuleNameMatcher,
    model_bridge_registry
)


class QwenStructureBridge(ModelStructureBridge):
    """Structure bridge for Qwen models."""
    
    @classmethod
    def create_matcher(cls):
        """Creates a matcher for Qwen models."""
        # Matcher based on module names, using safer patterns
        # Limit character classes to reasonable ranges, avoid greedy matching
        module_matcher = ModuleNameMatcher([
            r"^[a-zA-Z0-9_.]{0,100}input_layernorm[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}post_attention_layernorm[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}self_attn\.q_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}self_attn\.k_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}self_attn\.v_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}self_attn\.o_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}mlp\.gate_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}mlp\.up_proj[a-zA-Z0-9_.]{0,50}$",
            r"^[a-zA-Z0-9_.]{0,100}mlp\.down_proj[a-zA-Z0-9_.]{0,50}$"
        ], match_all=True)  # Must match all
        
        # Composite matcher: config match OR module match
        return CompositeMatcher([module_matcher])
    
    def analyze_structure(self):
        """Analyzes the Qwen model structure and registers all relevant structure pairs."""
        attn_norm_linear_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        attn_linear_linear_names = ["self_attn.o_proj"]
        mlp_norm_linear_names = ["mlp.gate_proj", "mlp.up_proj"]
        mlp_linear_linear_names = ["mlp.down_proj"]
        
        # Table-driven configuration for structure analysis
        structure_configs = [
            {
                "pattern": "input_layernorm",
                "linear_names": attn_norm_linear_names,
                "pair_class": AttnNormLinearPair
            },
            {
                "pattern": "self_attn.v_proj",
                "linear_names": attn_linear_linear_names,
                "pair_class": AttnLinearLinearPair
            },
            {
                "pattern": "post_attention_layernorm",
                "linear_names": mlp_norm_linear_names,
                "pair_class": MLPNormLinearPair
            },
            {
                "pattern": "mlp.up_proj",
                "linear_names": mlp_linear_linear_names,
                "pair_class": MLPLinearLinearPair
            }
        ]
        
        for name, _ in self.model.named_modules():
            for config in structure_configs:
                if config["pattern"] in name:
                    linears = []
                    clean_name = remove_after_substring(name, config["pattern"])
                    for linear_name in config["linear_names"]:
                        linears.append(clean_name.replace(config["pattern"], linear_name))
                    prefix_name = '.'.join(clean_name.split('.')[:3])
                    self.register_structure_pair(config["pair_class"](self.config, clean_name, linears, prefix_name))
        
        self._layers_name = "model.layers"


# Auto-register QwenStructureBridge
model_bridge_registry.register(
    QwenStructureBridge, 
    QwenStructureBridge.create_matcher(),
    priority=100
)