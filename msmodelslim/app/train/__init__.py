#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Train app package: proxy dispatch lives under ``app.train.proxy``."""

from msmodelslim.app.train.proxy import TrainProxyApplication, TrainTask, TrainTaskConfig

__all__ = ["TrainProxyApplication", "TrainTask", "TrainTaskConfig"]
