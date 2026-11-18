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

from typing import Dict, Type, Optional, Union, List
import re
from abc import ABC, abstractmethod
from torch.nn import Module

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import ModelStructureBridge


class ModelMatcher(ABC):
    """Abstract base class for model matchers, used to determine if a model matches a specific bridge."""
    
    @abstractmethod
    def match(self, model: Module) -> bool:
        """Determines if the model matches."""
        pass


class ConfigMatcher(ModelMatcher):
    """Matcher based on model configuration."""
    
    def __init__(self, config_patterns: Dict[str, Union[str, List[str]]]):
        """Initializes the configuration matcher.
        
        Args:
            config_patterns: Configuration pattern dictionary, keys are config attribute names, 
                             values are expected values or lists of values.
                             Example: {"model_type": "qwen", "architectures": ["QwenLMHeadModel"]}
        """
        self.config_patterns = config_patterns
    
    def match(self, model: Module) -> bool:
        config = getattr(model, 'config', None)
        if config is None:
            return False
            
        for attr_name, expected_values in self.config_patterns.items():
            if not hasattr(config, attr_name):
                return False
                
            attr_value = getattr(config, attr_name)
            
            if isinstance(expected_values, list):
                if attr_value not in expected_values:
                    return False
            else:
                if isinstance(expected_values, str) and isinstance(attr_value, str):
                    if not re.match(expected_values, attr_value, re.IGNORECASE):
                        return False
                elif attr_value != expected_values:
                    return False
                    
        return True


class ModuleNameMatcher(ModelMatcher):
    """Matcher based on module names."""
    
    def __init__(self, module_patterns: List[str], match_all: bool = True):
        """Initializes the module name matcher.
        
        Args:
            module_patterns: List of module name patterns, supports regular expressions.
            match_all: Whether all patterns need to be matched; False means any pattern is sufficient.
        """
        self.module_patterns = module_patterns
        self.match_all = match_all
    
    def match(self, model: Module) -> bool:
        """Matches the model based on its module names."""
        module_names = [name for name, _ in model.named_modules()]
        
        if self.match_all:
            for pattern in self.module_patterns:
                if not any(re.search(pattern, name) for name in module_names):
                    return False
            return True
        else:
            for pattern in self.module_patterns:
                if any(re.search(pattern, name) for name in module_names):
                    return True
            return False


class CompositeMatcher(ModelMatcher):
    """Composite matcher, supports logical combination of multiple matchers."""
    
    def __init__(self, matchers: List[ModelMatcher], logic: str = "AND"):
        """Initializes the composite matcher.
        
        Args:
            matchers: List of matchers.
            logic: Logical operation, "AND" or "OR".
        """
        self.matchers = matchers
        self.logic = logic.upper()
        if self.logic not in ["AND", "OR"]:
            raise ValueError("logic must be 'AND' or 'OR'")
    
    def match(self, model: Module) -> bool:
        """Applies composite matching logic."""
        if not self.matchers:
            return False
            
        if self.logic == "AND":
            return all(matcher.match(model) for matcher in self.matchers)
        else: # OR
            return any(matcher.match(model) for matcher in self.matchers)


class ModelBridgeRegistry:
    """Model bridge registry, responsible for managing and automatically selecting appropriate bridges."""
    
    def __init__(self):
        self._registry: List[tuple[ModelMatcher, Type[ModelStructureBridge]]] = []
        self._priority_registry: List[tuple[int, ModelMatcher, Type[ModelStructureBridge]]] = []
    
    def register(self, 
                 bridge_class: Type[ModelStructureBridge], 
                 matcher: ModelMatcher,
                 priority: int = 0):
        """Registers a bridge.
        
        Args:
            bridge_class: The bridge class.
            matcher: The model matcher.
            priority: Priority, higher value means higher priority.
        """
        if not issubclass(bridge_class, ModelStructureBridge):
            raise TypeError(f"bridge_class must be a subclass of ModelStructureBridge, got {bridge_class}")
        
        # Insert at the appropriate position to maintain priority sorting
        entry = (priority, matcher, bridge_class)
        inserted = False
        for i, (p, _, _) in enumerate(self._priority_registry):
            if priority > p:
                self._priority_registry.insert(i, entry)
                inserted = True
                break
        if not inserted:
            self._priority_registry.append(entry)
        
        # Also maintain the old registry format for backward compatibility
        self._registry.append((matcher, bridge_class))

    def get_bridge(self, model: Module, config: dict = None) -> ModelStructureBridge:
        """Gets the matching bridge instance.
        
        Args:
            model: The model to process.
            config: Optional configuration dictionary.
            
        Returns:
            ModelStructureBridge: The matching bridge instance.
            
        Raises:
            ValueError: When no matching bridge is found.
        """
        # Find matching bridge in priority order
        for _, matcher, bridge_class in self._priority_registry:
            if matcher.match(model):
                return bridge_class(model, config)
        
        raise ValueError(f"No matching bridge found for model {model.__class__.__name__}. "
                        f"Please register a suitable bridge for this model type.")

    def clear(self):
        """Clears the registry."""
        self._registry.clear()
        self._priority_registry.clear()


# Global registry instance
model_bridge_registry = ModelBridgeRegistry()


def get_model_bridge(model: Module, config: dict = None) -> ModelStructureBridge:
    """Convenience function: Gets a model bridge.
    
    Args:
        model: The model to process.
        config: Optional configuration dictionary.
        
    Returns:
        ModelStructureBridge: The matching bridge instance.
    """
    return model_bridge_registry.get_bridge(model, config)