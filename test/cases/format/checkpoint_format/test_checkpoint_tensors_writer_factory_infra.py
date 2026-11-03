#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

import pytest

from msmodelslim.format.checkpoint_format.checkpoint_tensors_writer_factory_infra import (
    CheckpointSafetensorsWriterFactoryInfra,
    CheckpointSafetensorsWriterInfra,
)


def test_CheckpointSafetensorsWriterInfra_is_abstract_when_instantiated():
    with pytest.raises(TypeError):
        CheckpointSafetensorsWriterInfra()


def test_CheckpointSafetensorsWriterFactoryInfra_is_abstract_when_instantiated():
    with pytest.raises(TypeError):
        CheckpointSafetensorsWriterFactoryInfra()
