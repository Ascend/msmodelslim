#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

import pytest

from msmodelslim.format.checkpoint_format.checkpoint_json_writer_factory_infra import (
    CheckpointJsonWriterFactoryInfra,
    CheckpointJsonWriterInfra,
)


def test_CheckpointJsonWriterInfra_is_abstract_when_instantiated():
    with pytest.raises(TypeError):
        CheckpointJsonWriterInfra()


def test_CheckpointJsonWriterFactoryInfra_is_abstract_when_instantiated():
    with pytest.raises(TypeError):
        CheckpointJsonWriterFactoryInfra()
