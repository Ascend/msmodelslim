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
# pylint: disable=chained-comparison,logging-fstring-interpolation,no-member,unspecified-encoding

import re
import os
import sys
import shutil
import stat
import json
import csv  # noqa: F401
from example.common.security.type import check_dict_character, check_type
from msmodelslim.utils.logging import LOGGER_FUNC
from msmodelslim import logger


PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_512G = 549755813888  # 512G, 512 * 1024 * 1024 * 1024

# group not writable, others no permission, max stat is 750
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH
# group not writable, others not writable, max stat is 755
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def is_endswith_extensions(path, extensions):
    result = False
    if isinstance(extensions, (list, tuple)):
        for extension in extensions:
            if path.endswith(extension):
                result = True
                break
    elif isinstance(extensions, str):
        result = path.endswith(extensions)
    return result


def get_valid_path(path, extensions=None):
    check_type(path, str, "path")
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")

    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be soft link: {}.".format(path))

    real_path = os.path.realpath(path)

    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise ValueError("The length of filename should be less than 256.")
    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if extensions and not is_endswith_extensions(path, extensions):  # Check whether the file name endswith extension
        raise ValueError("The filename {} doesn't endswith \"{}\".".format(path, extensions))

    return real_path


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def check_others_not_writable(path):
    dir_stat = os.stat(path)
    is_writable = (
        bool(dir_stat.st_mode & stat.S_IWGRP)  # 组可写
        or bool(dir_stat.st_mode & stat.S_IWOTH)  # 其他用户可写
    )
    if is_writable:
        logger.warning("The file path %r may be insecure because it can be written by others.", path)


def check_path_owner_consistent(path):
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid() and os.getuid() != 0:
        logger.warning("The file path %r may be insecure because is does not belong to you.", path)


def check_dirpath_before_read(path):
    path = os.path.realpath(path)
    dirpath = os.path.dirname(path)
    check_others_not_writable(dirpath)
    check_path_owner_consistent(dirpath)


def get_valid_read_path(path, extensions=None, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    check_dirpath_before_read(path)
    real_path = get_valid_path(path, extensions)
    if not is_dir and not os.path.isfile(real_path):
        raise ValueError("The path {} doesn't exist or not a file.".format(path))
    if is_dir and not os.path.isdir(real_path):
        raise ValueError("The path {} doesn't exist or not a directory.".format(path))

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        if os.geteuid() == 0:
            logger.warning(
                "The file %r doesn't belong to the current user or group. current user is root, continue", path
            )
        else:
            raise ValueError("The file {} doesn't belong to the current user or group.".format(path))
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise ValueError("The file {} is group writable, or is others writable.".format(path))
    if not os.access(real_path, os.R_OK) or file_stat.st_mode & stat.S_IRUSR == 0:  # At least been 400
        raise ValueError("Current user doesn't have read permission to the file {}.".format(path))
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError("The file {} exceeds size limitation of {}.".format(path, size_max))
    return real_path


def check_write_directory(dir_name, check_user_stat=True):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise ValueError("The file writen directory {} doesn't exist.".format(dir_name))

    file_stat = os.stat(real_dir_name)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        if os.geteuid() == 0:
            logger.warning(
                "The file writen directory %r doesn't belong to the current user or group"
                " current user is root, continue",
                dir_name,
            )
        else:
            raise ValueError(
                "The file writen directory {} doesn't belong to the current user or group.".format(dir_name)
            )
    if not os.access(real_dir_name, os.W_OK):
        raise ValueError("Current user doesn't have writen permission to file writen directory {}.".format(dir_name))


def get_write_directory(dir_name, write_mode=0o750):
    real_dir_name = get_valid_path(dir_name)
    if os.path.exists(real_dir_name):
        logger.info("write directory exists, write file to directory %r", dir_name)
    else:
        logger.warning("write directory not exists, creating directory %r", dir_name)
        os.makedirs(name=real_dir_name, mode=write_mode, exist_ok=True)
    return real_dir_name


def get_valid_write_path(path, extensions=None, check_user_stat=True, is_dir=False, warn_exists=True):
    real_path = get_valid_path(path, extensions)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)

    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise ValueError("The file {} exist and is a directory.".format(path))
        if check_user_stat and os.stat(real_path).st_uid != os.getuid():  # Has to be exactly belonging to current user
            raise ValueError("The file {} doesn't belong to the current user.".format(path))
        if check_user_stat and os.stat(real_path).st_mode & WRITE_FILE_NOT_PERMITTED_STAT > 0:
            raise ValueError("The file {} permission for others is not 0, or is group writable.".format(path))
        if not os.access(real_path, os.W_OK):
            raise ValueError("The file {} exist and not writable.".format(path))
        if warn_exists:
            logger.warning("%r already exist. The original file will be overwritten.", path)
    return real_path


def yaml_safe_load(
    path, extensions=("yml", "yaml"), size_max=MAX_READ_FILE_SIZE_4G, key_max_len=512, check_user_stat=True
):
    import yaml

    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    with open(path) as yaml_file:
        raw_dict = yaml.safe_load(yaml_file)
    check_dict_character(raw_dict, key_max_len)
    return raw_dict


def json_safe_load(path, extensions="json", size_max=MAX_READ_FILE_SIZE_4G, key_max_len=512, check_user_stat=True):
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    with open(path) as json_file:
        raw_dict = json.load(json_file)
    if isinstance(raw_dict, dict):
        check_dict_character(raw_dict, key_max_len)
    return raw_dict


def yaml_safe_dump(obj, path, extensions=("yml", "yaml"), check_user_stat=True):
    import yaml

    check_dict_character(obj)
    write_path = get_valid_write_path(path, extensions, check_user_stat)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as yaml_file:
        yaml.safe_dump(obj, yaml_file)


def json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    if isinstance(obj, dict):
        check_dict_character(obj)
    write_path = get_valid_write_path(path, extensions, check_user_stat)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(obj, json_file, indent=indent)


def file_safe_write(obj, path, extensions=None, check_user_stat=True):
    """File write with trunc, the original file will be overwritten if exists."""
    if not isinstance(obj, str):
        raise TypeError("obj must be str.")
    write_path = get_valid_write_path(path, extensions, check_user_stat)
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as file:
        file.write(obj)


def safe_delete_path_if_exists(path, logger_level="info"):
    if os.path.exists(path):
        is_dir = os.path.isdir(path)
        path = get_valid_write_path(path, check_user_stat=True, is_dir=is_dir, warn_exists=False)  # Check if writable
        logger_func = LOGGER_FUNC[logger_level]
        if os.path.isfile(path):
            logger_func(f"File '{path}' exists and will be deleted.")
            os.remove(path)
        else:
            logger_func(f"Folder '{path}' exists and will be deleted.")
            shutil.rmtree(path)


def safe_copy_file(src_path, dest_path, size_max=MAX_READ_FILE_SIZE_4G):
    src_path = get_valid_read_path(src_path, size_max=size_max)
    if os.path.isdir(dest_path):
        dest_path = os.path.join(dest_path, os.path.basename(src_path))
    dest_path = get_valid_write_path(dest_path)

    shutil.copy2(src_path, dest_path, follow_symlinks=False)


def set_file_stat(path, stat_mode="640"):
    real_path = get_valid_path(path)
    if os.path.isfile(real_path) and is_belong_to_user_or_group(os.stat(real_path)):
        os.chmod(real_path, int(stat_mode, 8))


class SafeWriteUmask:
    """Write with preset umask
    Usage:
    As a decorator:
    >>> @SafeWriteUmask
    >>> def function():
    >>>     ...

    In with block:
    >>> with SafeWriteUmask(), open(..., "w") as ...:
    >>>     ...
    """

    def __init__(self, func=None, umask=0o027):
        self.func = func
        self.umask = umask
        self.ori_umask = None

    def __call__(self, *args, **kwargs):
        self.__enter__()
        out = self.func(*args, **kwargs)
        self.__exit__()
        return out

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)


def txt_safe_load(path, extensions=("txt"), size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, delimiter=","):
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    txt_data = []

    with open(path, 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.strip()  # 去除首尾空白字符
            if line:  # 跳过空行
                # 按分隔符分割并去除每个元素的空白字符
                row = [item.strip() for item in line.split(delimiter)]
                txt_data.append(row)

    return txt_data
