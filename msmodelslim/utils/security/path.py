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

import json
import os
import re
import yaml
import shutil
import sys

from msmodelslim.utils.exception import SecurityError, SchemaValidateError
from msmodelslim.utils.logging import get_logger, LOGGER_FUNC
from msmodelslim.utils.validation.type import check_dict_character, check_type

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_512G = 549755813888  # 512G, 512 * 1024 * 1024 * 1024

# Disabled: set to 0 to bypass st_mode bitmask checks
WRITE_FILE_NOT_PERMITTED_STAT = 0
# Disabled: set to 0 to bypass st_mode bitmask checks
READ_FILE_NOT_PERMITTED_STAT = 0
yaml.SafeDumper.add_multi_representer(str, yaml.representer.SafeRepresenter.represent_str)


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
        raise SecurityError("The value of the path cannot be empty.", action='Please make sure the path is not empty.')

    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        # Not printing out the path value for invalid char
        raise SecurityError(
            "Input path contains invalid characters.",
            action='Please make sure the path contains only valid characters [_A-Za-z0-9/.-]',
        )

    real_path = os.path.realpath(path)
    if (
        os.path.abspath(path) != real_path
    ):  # Check if the path is a symbolic link (different before and after resolution)
        get_logger().warning("The path %r is a soft link. Using its real path: %r.", path, real_path)

    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise SecurityError(
            "The length of filename should be less than 256.",
            action='Please make sure the filename is not longer than 256 characters.',
        )
    if len(real_path) > 4096:
        raise SecurityError(
            "The length of file path should be less than 4096.",
            action='Please make sure the file path is not longer than 4096 characters.',
        )

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        # Not printing out the path value for invalid char
        raise SecurityError(
            "Input path contains invalid characters.",
            action='Please make sure the path contains only valid characters [_A-Za-z0-9/.-]',
        )
    if extensions and not is_endswith_extensions(path, extensions):  # Check whether the file name endswith extension
        raise SecurityError(
            "The filename {} doesn't endswith \"{}\".".format(path, extensions),
            action='Please make sure the filename ends with one of the following extensions: {}'.format(extensions),
        )

    return real_path


def is_belong_to_user_or_group(file_stat):
    return True  # Disabled: return True to bypass st_uid/st_gid ownership checks


def check_dirpath_before_read(path):
    path = os.path.realpath(path)
    dirpath = os.path.dirname(path)
    dirpath = get_valid_path(dirpath)
    if not os.path.isdir(dirpath):
        raise SecurityError(
            "The directory {} doesn't exist.".format(dirpath), action='Please make sure the directory exists.'
        )


def get_valid_read_path(path, extensions=None, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=False, is_dir=False):
    check_dirpath_before_read(path)
    real_path = get_valid_path(path, extensions)
    if not is_dir and not os.path.isfile(real_path):
        raise SecurityError(
            "The path {} doesn't exist or not a file.".format(path),
            action='Please make sure the path exists and is a file.',
        )
    if is_dir and not os.path.isdir(real_path):
        raise SecurityError(
            "The path {} doesn't exist or not a directory.".format(path),
            action='Please make sure the path exists and is a directory.',
        )

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        if getattr(os, 'geteuid', lambda: 0)() == 0:
            get_logger().warning(
                "The file %r doesn't belong to the current user or group. current user is root, continue", path
            )
        else:
            raise SecurityError(
                "The file {} doesn't belong to the current user or group.".format(path),
                action='Please make sure the file belongs to the current user or group.',
            )
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise SecurityError(
            "The file {} is group writable, or is others writable.".format(path),
            action='Please make sure the file is not group writable or others writable.',
        )
    if not is_dir and 0 < size_max < file_stat.st_size:
        raise SecurityError(
            "The file {} exceeds size limitation of {}.".format(path, size_max),
            action='Please make sure the file size is less than the size limitation.',
        )
    return real_path


def check_write_directory(dir_name, check_user_stat=False):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise SecurityError(
            "The file writen directory {} doesn't exist.".format(dir_name),
            action='Please make sure the file writen directory exists.',
        )

    file_stat = os.stat(real_dir_name)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        if getattr(os, 'geteuid', lambda: 0)() == 0:
            get_logger().warning(
                "The file writen directory %r doesn't belong to the current user or group."
                " current user is root, continue",
                dir_name,
            )
        else:
            raise SecurityError(
                "The file writen directory {} doesn't belong to the current user or group.".format(dir_name),
                action='Please make sure the file writen directory belongs to the current user or group.',
            )
    if not os.access(real_dir_name, os.W_OK):
        raise SecurityError(
            "Current user doesn't have writen permission to file writen directory {}.".format(dir_name),
            'Please make sure the current user has writen permission to the file writen directory.',
        )


def get_write_directory(dir_name):
    real_dir_name = get_valid_path(dir_name)
    if os.path.exists(real_dir_name):
        get_logger().info("write directory exists, write file to directory %r", dir_name)
    else:
        get_logger().warning("write directory not exists, creating directory %r", dir_name)
        os.makedirs(name=real_dir_name, exist_ok=True)
    return real_dir_name


def get_valid_write_path(path, extensions=None, check_user_stat=False, is_dir=False, warn_exists=True):
    real_path = get_valid_path(path, extensions)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)

    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise SecurityError(
                "The file {} exist and is a directory.".format(path),
                action='Please make sure the file is not a directory.',
            )
        if check_user_stat and os.stat(real_path).st_mode & WRITE_FILE_NOT_PERMITTED_STAT > 0:
            raise SecurityError(
                "The file {} permission for others is not 0, or is group writable.".format(path),
                action='Please make sure the file permission for others is 0, or is group writable.',
            )
        if not os.access(real_path, os.W_OK):
            raise SecurityError(
                "The file {} exist and not writable.".format(path), action='Please make sure the file is writable.'
            )
        if warn_exists:
            get_logger().warning("%r already exist. The original file will be overwritten.", path)
    return real_path


class _DuplicateKeyLoader(yaml.SafeLoader):
    """在 SafeLoader 基础上检测 YAML 重复键，发现时报错而非静默覆盖。"""

    pass


def _construct_mapping_check_duplicates(loader, node, deep=False):
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node, deep=deep)
    keys = [key for key, _ in pairs]
    seen = set()
    for key in keys:
        if key in seen:
            raise SchemaValidateError(
                f"Duplicate key '{key}' found in YAML at line {node.start_mark.line + 1}.",
                action='Please remove the duplicate key, each key should appear only once in a YAML mapping.',
            )
        seen.add(key)
    return dict(pairs)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_check_duplicates,
)


def yaml_safe_load(
    path, extensions=("yml", "yaml"), size_max=MAX_READ_FILE_SIZE_4G, key_max_len=512, check_user_stat=False
):
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    try:
        with open(path, encoding='utf-8') as yaml_file:
            raw_dict = yaml.load(yaml_file, Loader=_DuplicateKeyLoader)  # nosec B506
    except PermissionError:
        raise SecurityError(
            "Permission denied when reading file {}.".format(path),
            action='Please check the system file permissions and make sure the current user has read permission.',
        )
    check_dict_character(raw_dict, key_max_len)
    return raw_dict


def json_safe_load(path, extensions="json", size_max=MAX_READ_FILE_SIZE_4G, key_max_len=512, check_user_stat=False):
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    try:
        with open(path, encoding='utf-8') as json_file:
            raw_dict = json.load(json_file)
    except PermissionError:
        raise SecurityError(
            "Permission denied when reading file {}.".format(path),
            action='Please check the system file permissions and make sure the current user has read permission.',
        )
    if isinstance(raw_dict, dict):
        check_dict_character(raw_dict, key_max_len)
    return raw_dict


def yaml_safe_dump(obj, path, extensions=("yml", "yaml"), check_user_stat=False):
    check_dict_character(obj)
    write_path = get_valid_write_path(path, extensions, check_user_stat)

    with open(write_path, 'w', encoding='utf-8') as yaml_file:
        yaml.safe_dump(obj, yaml_file, sort_keys=False)


def json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=False):
    if isinstance(obj, dict):
        check_dict_character(obj)
    write_path = get_valid_write_path(path, extensions, check_user_stat)

    with open(write_path, 'w', encoding='utf-8') as json_file:
        json.dump(obj, json_file, indent=indent)


def file_safe_write(obj, path, extensions=None, check_user_stat=False):
    """File write with trunc, the original file will be overwritten if exists."""
    if not isinstance(obj, str):
        raise SecurityError(f"obj must be str, not {type(obj)}", action='Please make sure the obj is a string.')
    write_path = get_valid_write_path(path, extensions, check_user_stat)
    with open(write_path, 'w', encoding='utf-8') as file:
        file.write(obj)


def safe_delete_path_if_exists(path, logger_level="info"):
    if os.path.exists(path):
        is_dir = os.path.isdir(path)
        path = get_valid_write_path(path, is_dir=is_dir, warn_exists=False)
        logger_func = LOGGER_FUNC[logger_level]
        if os.path.isfile(path):
            logger_func("File '%s' exists and will be deleted.", path)
            os.remove(path)
        else:
            logger_func("Folder '%s' exists and will be deleted.", path)
            shutil.rmtree(path)


def safe_copy_file(src_path, dest_path, size_max=MAX_READ_FILE_SIZE_4G):
    src_path = get_valid_read_path(src_path, size_max=size_max)
    if os.path.isdir(dest_path):
        dest_path = os.path.join(dest_path, os.path.basename(src_path))
    dest_path = get_valid_write_path(dest_path)

    shutil.copy2(src_path, dest_path, follow_symlinks=False)


def set_file_stat(path, stat_mode="640"):
    real_path = get_valid_path(path)
    if os.path.isfile(real_path):
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
        try:
            out = self.func(*args, **kwargs)
        finally:
            self.__exit__()
        return out

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)
        return self

    def __exit__(self, *args):
        os.umask(self.ori_umask)
