#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
# pylint: disable=unnecessary-lambda

from functools import wraps
import logging


class MsgConst:
    """
    Class for log messages const
    """

    SPECIAL_CHAR = ["\n", "\r", "\u007f", "\b", "\f", "\t", "\u000b", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


def get_logger():
    amc_logger = logging.getLogger("msmodelslim-logger")
    amc_logger.propagate = False
    amc_logger.setLevel(logging.INFO)
    if not amc_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        amc_logger.addHandler(stream_handler)
    return amc_logger


logger = get_logger()


logger_critical = logger.critical
logger_debug = logger.debug
logger_error = logger.error
logger_info = logger.info
logger_warning = logger.warning


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg, *args):
        for char in MsgConst.SPECIAL_CHAR:
            if isinstance(msg, str):
                msg = msg.replace(char, ' ')
        return func(msg, *args)

    return func_level


@filter_special_chars
def critical_filter(msg, *args):
    logger_critical(msg, *args)


@filter_special_chars
def debug_filter(msg, *args):
    logger_debug(msg, *args)


@filter_special_chars
def error_filter(msg, *args):
    logger_error(msg, *args)


@filter_special_chars
def info_filter(msg, *args):
    logger_info(msg, *args)


@filter_special_chars
def warning_filter(msg, *args):
    logger_warning(msg, *args)


setattr(logger, 'critical', critical_filter)
setattr(logger, 'debug', debug_filter)
setattr(logger, 'error', error_filter)
setattr(logger, 'info', info_filter)
setattr(logger, 'warning', warning_filter)


LOG_LEVEL = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}

LOGGER_FUNC = {
    "debug": lambda msg: logger.debug(msg),
    "info": lambda msg: logger.info(msg),
    "warn": lambda msg: logger.warning(msg),
    "warning": lambda msg: logger.warning(msg),
    "error": lambda msg: logger.error(msg),
    "critical": lambda msg: logger.critical(msg),
}


def set_logger_level(level="info"):
    if level.lower() in LOG_LEVEL:
        logger.setLevel(LOG_LEVEL.get(level.lower()))
    else:
        logger.warning("Set %r log level failed.", level)


def progress_bar(iterable, desc: str = None, total: int = -1, interval: int = 1):
    if total == -1 and hasattr(iterable, "__len__"):
        total = len(iterable)

    format_str = "" if desc is None else (desc + ": ")
    if isinstance(total, int) and total > 0:
        format_str += "[%d/{}]".format(total)
    else:
        format_str += "[%d]"

    if not (isinstance(interval, int) and interval > 0):
        interval = 1

    prev_terminator = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = '\r'
    for item_id, item in enumerate(iterable, start=1):
        if item_id % interval == 0:
            logger.info(format_str, item_id)
        yield item
    logging.StreamHandler.terminator = prev_terminator
    logger.info("")
