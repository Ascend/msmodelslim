# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from functools import wraps
from typing import Callable


def singleton(cls):
    _instances = {}

    @wraps(cls)
    def _get_instances(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return _get_instances


@singleton
class AciLogger(object):
    def __init__(self):
        self.logger = logger

    def info(self, msg: str):
        return self.logger.info(msg)

    def error(self, msg: str):
        return self.logger.error(msg)
    
    def warning(self, msg: str):
        return self.logger.warning(msg)
    
    def debug(self, msg: str):
        return self.logger.debug(msg)


class CallParams:
    """
    to save call params
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs
    
    def call(self, func: Callable):
        func(*self.args, **self.kwargs)