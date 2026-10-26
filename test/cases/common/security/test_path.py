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

import os
import stat
import json

import yaml
import pytest

from ascend_utils.common import security

TEST_DIR = "/tmp/a_test_path_for_testing_automl_common/"
TEST_READ_FILE_NAME = TEST_DIR + "testfile.testfile"
USER_NOT_PERMITTED_READ_FILE = TEST_DIR + "testfile_not_readable.testfile"
OTHERS_READABLE_READ_FILE = TEST_DIR + "testfile_others_readable.testfile"
OTHERS_WRITABLE_READ_FILE = TEST_DIR + "testfile_others_writable.testfile"
USER_NOT_PERMITTED_WRITE_FILE = TEST_DIR + "testfile_not_writable/foo"
JSON_FILE = TEST_DIR + "testfile.json"
YAML_FILE = TEST_DIR + "testfile.yaml"
TEST_FILE = TEST_DIR + "testfile.test"
ORI_DATA = {"a_long_key_name": 1, 12: "b", 3.14: "", "c": {"d": 3, "e": 4}, True: "true", False: "false", None: "null"}
OVER_WRITE_DATA = {"hello": "world"}


def setup_module():
    os.makedirs(TEST_DIR, mode=int('700', 8), exist_ok=True)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(
        os.open(TEST_READ_FILE_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w"
    ) as temp_file:
        temp_file.write("a_test_file_name_for_testing_automl_common")

    with os.fdopen(os.open(USER_NOT_PERMITTED_READ_FILE, os.O_CREAT, mode=000), "w"):
        pass

    with os.fdopen(os.open(OTHERS_READABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_READABLE_READ_FILE, int('755', 8))

    with os.fdopen(os.open(OTHERS_WRITABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_WRITABLE_READ_FILE, int('666', 8))

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.makedirs(dir_name, mode=int('500', 8), exist_ok=True)

    with os.fdopen(os.open(JSON_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(ORI_DATA, json_file)

    with os.fdopen(os.open(YAML_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as yaml_file:
        yaml.dump(ORI_DATA, yaml_file)


def teardown_module():
    os.remove(TEST_READ_FILE_NAME)
    os.chmod(USER_NOT_PERMITTED_READ_FILE, int('600', 8))
    os.remove(USER_NOT_PERMITTED_READ_FILE)
    os.remove(OTHERS_READABLE_READ_FILE)
    os.remove(OTHERS_WRITABLE_READ_FILE)

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.chmod(dir_name, int('700', 8))
    os.removedirs(dir_name)

    os.remove(JSON_FILE)
    os.remove(YAML_FILE)
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    os.removedirs(TEST_DIR)


def test_get_valid_path_pass_when_valid_path():
    security.get_valid_path('../anypath')
    security.get_valid_path('../anypath/a')


def test_get_valid_path_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        security.get_valid_path('../anypath*a')
    with pytest.raises(ValueError):
        security.get_valid_path('../anypath/\\a')
    with pytest.raises(ValueError):
        security.get_valid_path('../anypath/!a')


def test_get_valid_read_path_pass_when_readable_file():
    security.get_valid_read_path(TEST_READ_FILE_NAME)
    security.get_valid_read_path(TEST_READ_FILE_NAME, extensions='.testfile')
    security.get_valid_read_path(OTHERS_READABLE_READ_FILE)
    security.get_valid_read_path(OTHERS_WRITABLE_READ_FILE, check_user_stat=False)


def test_get_valid_read_path_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        security.get_valid_read_path('./not_exist')  # ValueError: The file ... doesn't exist or not a file.
    with pytest.raises(ValueError):
        # ValueError: The filename ... doesn't endswith ".json"
        security.get_valid_read_path(TEST_READ_FILE_NAME, extensions='.json')
    with pytest.raises(ValueError):
        # ValueError: The file ... exceeds size limitation of 1.
        security.get_valid_read_path(TEST_READ_FILE_NAME, size_max=1)
    with pytest.raises(ValueError):
        # ValueError: Current user doesn't have read permission to the file ....
        security.get_valid_read_path(USER_NOT_PERMITTED_READ_FILE)
    with pytest.raises(ValueError):
        # ValueError: The file ... has others writable permission.
        security.get_valid_read_path(OTHERS_WRITABLE_READ_FILE)


def test_check_write_directory_pass_when_valid_directory():
    security.check_write_directory(TEST_DIR)


def test_check_write_directory_given_invalid_when_any_then_error():
    with pytest.raises(ValueError):
        # ValueError: The file writen directory ... doesn't exist.
        security.check_write_directory('not_exists')


def test_get_valid_write_path_pass_when_writable_path():
    security.get_valid_write_path(TEST_READ_FILE_NAME, extensions='.testfile')
    # WARNING: root:... exists. The original file will be overwritten.


def test_get_valid_write_path_when_directory_not_exists():
    with pytest.raises(ValueError):
        # ValueError: The file writen directory ... doesn't exist.
        security.get_valid_write_path('not_exists/README.md', extensions='.md')


@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 0)() == 0,  # 直接判断：如果是 root 用户（UID=0）
    reason="root 用户跳过此用例",
)
def test_get_valid_write_path_when_no_write_permission():
    with pytest.raises(ValueError):
        # ValueError: Current user doesn't have writen permission to the file writen directory ....
        security.get_valid_write_path(USER_NOT_PERMITTED_WRITE_FILE)


def test_yaml_safe_load_pass_when_valid_yaml():
    security.yaml_safe_load(YAML_FILE)


def test_yaml_safe_load_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        # ValueError: The filename ... doesn't endswith "['.yml', '.yaml']".
        security.yaml_safe_load(TEST_READ_FILE_NAME)
    with pytest.raises(ValueError):
        # ValueError: Length of ... exceeds key limitation of 2.
        security.yaml_safe_load(YAML_FILE, key_max_len=2)


def test_json_safe_load_pass_when_valid_json():
    security.json_safe_load(JSON_FILE)


def test_json_safe_load_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        # ValueError: The filename ... doesn't endswith ".json"
        security.json_safe_load(YAML_FILE)
    with pytest.raises(ValueError):
        # ValueError: The file ... exceeds size limitation of 1.
        security.json_safe_load(JSON_FILE, size_max=1)


def test_file_safe_write_pass_when_valid_content():
    security.file_safe_write("hello world", TEST_FILE, ".test")


def test_file_safe_write_given_invalid_when_any_then_type_error():
    with pytest.raises(TypeError):
        # TypeError: obj must be str
        security.file_safe_write(ORI_DATA, TEST_FILE, ".test")


def test_yaml_safe_dump_yields_last_value_when_overwritten():
    security.yaml_safe_dump(ORI_DATA, YAML_FILE)
    security.yaml_safe_dump(OVER_WRITE_DATA, YAML_FILE)
    cur_dict = security.yaml_safe_load(YAML_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_json_safe_dump_yields_last_value_when_overwritten():
    security.json_safe_dump(ORI_DATA, JSON_FILE, indent=4)
    security.json_safe_dump(OVER_WRITE_DATA, JSON_FILE)
    cur_dict = security.json_safe_load(JSON_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_safe_copy_file_pass_when_copy_then_delete():
    dest_path = TEST_READ_FILE_NAME + "_copy"
    security.safe_copy_file(TEST_READ_FILE_NAME, dest_path)
    security.safe_delete_path_if_exists(dest_path)


def test_safe_write_umask_strips_group_other_bits_when_writing():
    security.safe_delete_path_if_exists(TEST_FILE)
    default_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fake_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP | stat.S_IWOTH | stat.S_IROTH  # 666
    with security.SafeWriteUmask(), os.fdopen(os.open(TEST_FILE, default_flags, mode=fake_mode), "w") as write_file:
        write_file.write("")
    assert os.stat(TEST_FILE).st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH) == 0
