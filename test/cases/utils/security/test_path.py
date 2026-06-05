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

msmodelslim.utils.security.path 模块的单元测试
"""

import os
import stat
import json
import sys
import yaml
from unittest.mock import patch

import pytest
from msmodelslim.utils.exception import SecurityError, SchemaValidateError

TEST_DIR = "/tmp/a_test_path_for_testing_automl_common/"
TEST_READ_FILE_NAME = TEST_DIR + "testfile.testfile"
USER_NOT_PERMITTED_READ_FILE = TEST_DIR + "testfile_not_readable.testfile"
OTHERS_READABLE_READ_FILE = TEST_DIR + "testfile_others_readable.testfile"
OTHERS_WRITABLE_READ_FILE = TEST_DIR + "testfile_others_writable.testfile"
USER_NOT_PERMITTED_WRITE_FILE = TEST_DIR + "testfile_not_writable/foo"
JSON_FILE = TEST_DIR + "testfile.json"
YAML_FILE = TEST_DIR + "testfile.yaml"
YAML_DUP_FILE = TEST_DIR + "dup.yaml"
TEST_FILE = TEST_DIR + "testfile.test"
ORI_DATA = {
    "a_long_key_name": 1,
    12: "b",
    3.14: "",
    "c": {"d": 3, "e": 4},
    True: "true",
    False: "false",
    None: "null",
}
OVER_WRITE_DATA = {"hello": "world"}


def setup_module():
    os.makedirs(TEST_DIR, mode=int("700", 8), exist_ok=True)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(
        os.open(
            TEST_READ_FILE_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            mode=default_mode,
        ),
        "w",
    ) as temp_file:
        temp_file.write("a_test_file_name_for_testing_automl_common")

    with os.fdopen(os.open(USER_NOT_PERMITTED_READ_FILE, os.O_CREAT, mode=000), "w"):
        pass

    with os.fdopen(os.open(OTHERS_READABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_READABLE_READ_FILE, int("755", 8))

    with os.fdopen(os.open(OTHERS_WRITABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_WRITABLE_READ_FILE, int("666", 8))

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.makedirs(dir_name, mode=int("500", 8), exist_ok=True)

    with os.fdopen(
        os.open(JSON_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode),
        "w",
    ) as json_file:
        json.dump(ORI_DATA, json_file)

    with os.fdopen(
        os.open(YAML_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode),
        "w",
    ) as yaml_file:
        yaml.dump(ORI_DATA, yaml_file)

    with os.fdopen(
        os.open(YAML_DUP_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode),
        "w",
    ) as yaml_dup_file:
        yaml_dup_file.write("a: 1\na: 2\n")

    module_name = "msmodelslim.utils.security.path"
    if module_name in sys.modules:
        del sys.modules[module_name]


def teardown_module():
    os.remove(TEST_READ_FILE_NAME)
    os.chmod(USER_NOT_PERMITTED_READ_FILE, int("600", 8))
    os.remove(USER_NOT_PERMITTED_READ_FILE)
    os.remove(OTHERS_READABLE_READ_FILE)
    os.remove(OTHERS_WRITABLE_READ_FILE)

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.chmod(dir_name, int("700", 8))
    os.removedirs(dir_name)

    os.remove(JSON_FILE)
    os.remove(YAML_FILE)
    if os.path.exists(YAML_DUP_FILE):
        os.remove(YAML_DUP_FILE)
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    os.removedirs(TEST_DIR)


def test_get_valid_path_given_valid_when_any_then_pass():
    """测试 get_valid_path：当路径合法时，函数应正常通过（不抛异常）"""
    from msmodelslim.utils.security.path import get_valid_path

    get_valid_path("../anypath")
    get_valid_path("../anypath/a")


def test_get_valid_path_given_invalid_when_any_then_value_error():
    """测试 get_valid_path：当路径包含非法字符时，应抛出 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_path

    with pytest.raises(SecurityError):
        get_valid_path("../anypath*a")
    with pytest.raises(SecurityError):
        get_valid_path("../anypath/\\a")
    with pytest.raises(SecurityError):
        get_valid_path("../anypath/!a")


def test_get_valid_read_path_given_valid_when_any_then_pass():
    """测试 get_valid_read_path：合法文件路径应成功通过安全性检查"""
    from msmodelslim.utils.security.path import get_valid_read_path

    get_valid_read_path(TEST_READ_FILE_NAME)
    get_valid_read_path(TEST_READ_FILE_NAME, extensions=".testfile")
    get_valid_read_path(OTHERS_READABLE_READ_FILE)
    get_valid_read_path(OTHERS_WRITABLE_READ_FILE, check_user_stat=False)


def test_get_valid_read_path_given_invalid_when_any_then_value_error():
    """测试 get_valid_read_path：非法或不符合要求的文件应抛 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_read_path

    with pytest.raises(SecurityError):
        get_valid_read_path("./not_exist")  # SecurityError: The file ... doesn't exist or not a file.
    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith ".json"
        get_valid_read_path(TEST_READ_FILE_NAME, extensions=".json")
    with pytest.raises(SecurityError):
        # SecurityError: The file ... exceeds size limitation of 1.
        get_valid_read_path(TEST_READ_FILE_NAME, size_max=1)
    with pytest.raises(SecurityError):
        # SecurityError: Current user doesn't have read permission to the file ....
        get_valid_read_path(USER_NOT_PERMITTED_READ_FILE)
    with pytest.raises(SecurityError):
        # SecurityError: The file ... has others writable permission.
        get_valid_read_path(OTHERS_WRITABLE_READ_FILE)


def test_check_write_directory_given_valid_when_any_then_pass():
    """测试 check_write_directory：合法目录应通过检查"""
    from msmodelslim.utils.security.path import check_write_directory

    check_write_directory(TEST_DIR)


def test_check_write_directory_given_invalid_when_any_then_error():
    """测试 check_write_directory：目录不存在时应抛 SecurityError"""
    from msmodelslim.utils.security.path import check_write_directory

    with pytest.raises(SecurityError):
        # SecurityError: The file writen directory ... doesn't exist.
        check_write_directory("not_exists")


def test_get_write_directory_given_valid_when_any_then_pass():
    """测试 get_write_directory：合法目录应被正确返回"""
    from msmodelslim.utils.security.path import get_write_directory

    get_write_directory(TEST_DIR)


def test_get_write_directory_given_invalid_when_any_then_error():
    """测试 get_write_directory：目录不存在时应抛异常"""
    from msmodelslim.utils.security.path import get_write_directory

    get_write_directory("not_exists_")


def test_get_valid_write_path_given_valid_when_any_then_pass():
    """测试 get_valid_write_path：合法文件路径应允许写入（可能覆盖旧文件）"""
    from msmodelslim.utils.security.path import get_valid_write_path

    get_valid_write_path(TEST_READ_FILE_NAME, extensions=".testfile")


def test_get_valid_write_path_when_directory_not_exists():
    """测试 get_valid_write_path：当目录不存在时应抛 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_write_path

    with pytest.raises(SecurityError):
        # SecurityError: The file writen directory ... doesn't exist.
        get_valid_write_path("not_exists/README.md", extensions=".md")


@pytest.mark.skipif(
    os.geteuid() == 0,  # 直接判断：如果是 root 用户（UID=0）
    reason="root 用户跳过此用例",
)
def test_get_valid_write_path_when_no_write_permission():
    """测试 get_valid_write_path：当前用户对目录无写权限时应抛 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_write_path

    with pytest.raises(SecurityError):
        # SecurityError: Current user doesn't have writen permission to the file writen directory ....
        get_valid_write_path(USER_NOT_PERMITTED_WRITE_FILE)


def test_yaml_safe_load_given_valid_when_any_then_pass():
    """测试 yaml_safe_load：合法 YAML 文件应成功加载"""
    from msmodelslim.utils.security.path import yaml_safe_load

    yaml_safe_load(YAML_FILE)


def test_yaml_safe_load_given_invalid_when_any_then_value_error():
    """测试 yaml_safe_load：非法或不符合格式的 YAML 文件应抛异常"""
    from msmodelslim.utils.security.path import yaml_safe_load

    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith "['.yml', '.yaml']".
        yaml_safe_load(TEST_READ_FILE_NAME)
    with pytest.raises(SchemaValidateError):
        # SecurityError: Length of ... exceeds key limitation of 2.
        yaml_safe_load(YAML_FILE, key_max_len=2)


def test_json_safe_load_given_valid_when_any_then_pass():
    """测试 json_safe_load：合法 JSON 文件应成功加载"""
    from msmodelslim.utils.security.path import json_safe_load

    json_safe_load(JSON_FILE)


def test_json_safe_load_given_invalid_when_any_then_value_error():
    """测试 json_safe_load：非法 JSON 文件应抛 SecurityError"""
    from msmodelslim.utils.security.path import json_safe_load

    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith ".json"
        json_safe_load(YAML_FILE)
    with pytest.raises(SecurityError):
        # SecurityError: The file ... exceeds size limitation of 1.
        json_safe_load(JSON_FILE, size_max=1)


def test_file_safe_write_given_valid_when_any_then_pass():
    """测试 file_safe_write：合法字符串内容应能安全写入文件"""
    from msmodelslim.utils.security.path import file_safe_write

    file_safe_write("hello world", TEST_FILE, ".test")


def test_file_safe_write_given_invalid_when_any_then_type_error():
    """测试 file_safe_write：写入非字符串对象应抛 SecurityError"""
    from msmodelslim.utils.security.path import file_safe_write

    with pytest.raises(SecurityError):
        # TypeError: obj must be str
        file_safe_write(ORI_DATA, TEST_FILE, ".test")


def test_yaml_safe_dump_given_valid_when_over_write_then_pass():
    """测试 yaml_safe_dump：可多次覆盖写入 YAML 文件并验证写入内容"""
    from msmodelslim.utils.security.path import yaml_safe_dump, yaml_safe_load

    yaml_safe_dump(ORI_DATA, YAML_FILE)
    yaml_safe_dump(OVER_WRITE_DATA, YAML_FILE)
    cur_dict = yaml_safe_load(YAML_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_json_safe_dump_given_valid_when_over_write_then_pass():
    """测试 json_safe_dump：JSON 文件多次写入覆盖应保持一致"""
    from msmodelslim.utils.security.path import json_safe_dump, json_safe_load

    json_safe_dump(ORI_DATA, JSON_FILE, indent=4)
    json_safe_dump(OVER_WRITE_DATA, JSON_FILE)
    cur_dict = json_safe_load(JSON_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_safe_copy_file_given_valid_when_over_write_then_pass():
    """测试 safe_copy_file：文件可安全复制并删除"""
    from msmodelslim.utils.security.path import (
        safe_copy_file,
        safe_delete_path_if_exists,
    )

    dest_path = TEST_READ_FILE_NAME + "_copy"
    safe_copy_file(TEST_READ_FILE_NAME, dest_path)
    safe_delete_path_if_exists(dest_path)


def test_set_file_stat_when_any_file_then_pass():
    """测试 set_file_stat：修改文件权限为安全模式后能正常执行"""
    from msmodelslim.utils.security.path import (
        safe_copy_file,
        set_file_stat,
        safe_delete_path_if_exists,
    )

    dest_path = TEST_READ_FILE_NAME + "_copy"
    safe_copy_file(TEST_READ_FILE_NAME, dest_path)
    set_file_stat(dest_path)
    safe_delete_path_if_exists(dest_path)


def test_safe_write_umask_given_valid_when_any_then_pass():
    """测试 SafeWriteUmask：确保创建文件时权限安全（无组/其他用户写权限）"""
    from msmodelslim.utils.security.path import (
        SafeWriteUmask,
        safe_delete_path_if_exists,
    )

    safe_delete_path_if_exists(TEST_FILE)
    default_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fake_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP | stat.S_IWOTH | stat.S_IROTH  # 666
    with SafeWriteUmask(), os.fdopen(os.open(TEST_FILE, default_flags, mode=fake_mode), "w") as write_file:
        write_file.write("")
    assert os.stat(TEST_FILE).st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH) == 0


# ------------------------------ 补充覆盖率测试 ------------------------------
class TestIsEndswithExtensions:
    """测试 is_endswith_extensions 扩展名匹配"""

    def test_is_endswith_extensions_return_true_when_tuple_match(self):
        """元组扩展名列表任一匹配时应返回 True"""
        from msmodelslim.utils.security.path import is_endswith_extensions

        assert is_endswith_extensions("a.json", (".json", ".yaml")) is True

    def test_is_endswith_extensions_return_true_when_str_match(self):
        """单个字符串扩展名匹配时应返回 True"""
        from msmodelslim.utils.security.path import is_endswith_extensions

        assert is_endswith_extensions("a.yaml", ".yaml") is True

    def test_is_endswith_extensions_return_false_when_no_match(self):
        """无匹配扩展名时应返回 False"""
        from msmodelslim.utils.security.path import is_endswith_extensions

        assert is_endswith_extensions("a.txt", ".json") is False


class TestGetValidPathEdgeCases:
    """测试 get_valid_path 边界与异常路径"""

    def test_get_valid_path_raise_security_error_when_empty(self):
        """空路径应抛出 SecurityError"""
        from msmodelslim.utils.security.path import get_valid_path

        with pytest.raises(SecurityError, match="cannot be empty"):
            get_valid_path("")

    def test_get_valid_path_raise_security_error_when_symlink(self, tmp_path):
        """软链接路径应抛出 SecurityError"""
        from msmodelslim.utils.security.path import get_valid_path

        target = tmp_path / "real.txt"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        with pytest.raises(SecurityError, match="soft link"):
            get_valid_path(str(link))

    def test_get_valid_path_raise_security_error_when_realpath_has_invalid_chars(self, tmp_path):
        """realpath 含非法字符时应抛出 SecurityError"""
        from msmodelslim.utils.security import path as path_mod

        short_file = tmp_path / "ok.txt"
        short_file.write_text("x", encoding="utf-8")
        short_real = str(short_file)
        bad_real = short_real + "*"
        with patch.object(path_mod.os.path, "realpath", return_value=bad_real):
            with patch.object(path_mod.os.path, "islink", return_value=False):
                with pytest.raises(SecurityError, match="invalid characters"):
                    path_mod.get_valid_path(short_real)


class TestCheckHelpers:
    """测试路径权限与属主相关辅助函数"""

    def test_check_others_not_writable_log_warning_when_world_writable(self, tmp_path):
        """目录对他人可写时应记录 warning"""
        from msmodelslim.utils.security.path import check_others_not_writable

        writable_dir = tmp_path / "writable"
        writable_dir.mkdir()
        os.chmod(writable_dir, 0o777)
        with patch("msmodelslim.utils.security.path.get_logger") as mock_logger:
            check_others_not_writable(str(writable_dir))
        assert mock_logger.return_value.warning.called

    def test_check_path_owner_consistent_log_warning_when_other_owner(self, tmp_path):
        """目录属主非当前用户时应记录 warning"""
        from msmodelslim.utils.security.path import check_path_owner_consistent

        other_dir = tmp_path / "other"
        other_dir.mkdir(mode=0o700)
        with patch("os.getuid", return_value=99999):
            with patch("msmodelslim.utils.security.path.get_logger") as mock_logger:
                check_path_owner_consistent(str(other_dir))
        mock_logger.return_value.warning.assert_called()

    def test_is_belong_to_user_or_group_return_true_when_same_uid(self):
        """当前用户拥有的文件应判定为属于用户或组"""
        from msmodelslim.utils.security.path import is_belong_to_user_or_group

        st = os.stat(TEST_READ_FILE_NAME)
        assert is_belong_to_user_or_group(st) is True


class TestGetValidReadPathExtended:
    """测试 get_valid_read_path / check_dirpath_before_read 扩展分支"""

    def test_get_valid_read_path_return_dir_when_is_dir_true(self):
        """is_dir=True 时应允许读取目录路径"""
        from msmodelslim.utils.security.path import get_valid_read_path

        result = get_valid_read_path(TEST_DIR, is_dir=True)
        assert os.path.isdir(result)

    def test_check_dirpath_before_read_raise_when_parent_missing(self):
        """父目录不存在时应抛出 SecurityError"""
        from msmodelslim.utils.security.path import check_dirpath_before_read

        with pytest.raises(SecurityError, match="doesn't exist"):
            check_dirpath_before_read("/nonexistent_parent_xyz/file.txt")

    def test_get_valid_read_path_warn_when_root_and_foreign_owner(self, tmp_path):
        """root 读取非属主文件时应记录 warning 并继续"""
        from msmodelslim.utils.security.path import get_valid_read_path

        if os.geteuid() != 0:
            pytest.skip("仅 root 用户可测 root 继续读取分支")
        foreign = str(tmp_path / "foreign.txt")
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(
            os.open(foreign, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=mode),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("x")
        with patch("os.geteuid", return_value=0):
            with patch(
                "msmodelslim.utils.security.path.is_belong_to_user_or_group",
                return_value=False,
            ):
                with patch("msmodelslim.utils.security.path.get_logger") as mock_logger:
                    get_valid_read_path(foreign, check_user_stat=True)
        mock_logger.return_value.warning.assert_called()


class TestGetValidWritePathExtended:
    """测试 get_valid_write_path / check_write_directory 扩展分支"""

    def setup_method(self):
        """确保TEST_DIR存在，因为teardown_module可能已删除"""
        os.makedirs(TEST_DIR, mode=int("700", 8), exist_ok=True)

    def test_get_valid_write_path_raise_when_target_is_directory(self):
        """写入目标是目录时应抛出 SecurityError"""
        from msmodelslim.utils.security.path import get_valid_write_path

        with pytest.raises(SecurityError, match="is a directory"):
            get_valid_write_path(TEST_DIR, is_dir=False)

    def test_get_valid_write_path_raise_when_not_owned_by_user(self, tmp_path):
        """已存在文件非当前用户属主时应抛出 SecurityError"""
        from msmodelslim.utils.security.path import get_valid_write_path

        if os.geteuid() == 0:
            pytest.skip("root 用户跳过属主检查")
        other_file = tmp_path / "other.txt"
        other_file.write_text("x", encoding="utf-8")
        # 仅patch path模块内的os.stat，避免影响os.path.exists等系统调用
        with patch("msmodelslim.utils.security.path.os.stat") as mock_stat:
            mock_stat.return_value.st_uid = os.getuid() + 999
            mock_stat.return_value.st_mode = stat.S_IWUSR | stat.S_IRUSR
            with pytest.raises(SecurityError, match="doesn't belong"):
                get_valid_write_path(str(other_file), check_user_stat=True)

    def test_check_write_directory_warn_when_root_and_foreign(self, tmp_path):
        """root 写入非属主目录时应记录 warning 并继续"""
        from msmodelslim.utils.security.path import check_write_directory

        if os.geteuid() != 0:
            pytest.skip("仅 root 可测")
        d = tmp_path / "wdir"
        d.mkdir(mode=0o700)
        with patch("os.geteuid", return_value=0):
            with patch(
                "msmodelslim.utils.security.path.is_belong_to_user_or_group",
                return_value=False,
            ):
                with patch("msmodelslim.utils.security.path.get_logger") as mock_logger:
                    check_write_directory(str(d), check_user_stat=True)
        mock_logger.return_value.warning.assert_called()


class TestYamlDuplicateKeys:
    """测试 yaml_safe_load 重复键检测"""

    def test_yaml_safe_load_raise_schema_error_when_duplicate_keys(self):
        """YAML 存在重复键时应抛出 SchemaValidateError"""
        from msmodelslim.utils.security.path import yaml_safe_load

        with pytest.raises(SchemaValidateError, match="Duplicate key"):
            yaml_safe_load(YAML_DUP_FILE)


class TestSafeDeleteAndCopy:
    """测试 safe_delete_path_if_exists / safe_copy_file"""

    def test_safe_delete_path_if_exists_remove_dir_when_directory(self, tmp_path):
        """目标为目录时应递归删除"""
        from msmodelslim.utils.security.path import safe_delete_path_if_exists

        sub = tmp_path / "to_delete"
        sub.mkdir()
        (sub / "inner.txt").write_text("x", encoding="utf-8")
        safe_delete_path_if_exists(str(sub), logger_level="debug")
        assert not sub.exists()

    def test_safe_copy_file_join_basename_when_dest_is_dir(self, tmp_path):
        """目标为目录时应复制到目录下同名文件"""
        from msmodelslim.utils.security.path import safe_copy_file

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir(mode=0o700)
        safe_copy_file(TEST_READ_FILE_NAME, str(dest_dir))
        copied = dest_dir / os.path.basename(TEST_READ_FILE_NAME)
        assert copied.exists()
        os.remove(copied)


class TestSafeWriteUmaskDecorator:
    """测试 SafeWriteUmask 装饰器用法"""

    def test_safe_write_umask_call_func_when_used_as_decorator(self):
        """作为装饰器使用时应正确设置并恢复 umask"""
        from msmodelslim.utils.security.path import SafeWriteUmask

        out = []

        @SafeWriteUmask
        def write_once():
            out.append("done")

        write_once()
        assert out == ["done"]


class TestSetFileStatEdge:
    """测试 set_file_stat 边界行为"""

    def test_set_file_stat_skip_chmod_when_not_owned(self, tmp_path):
        """文件非当前用户/组所有时不应 chmod"""
        from msmodelslim.utils.security.path import set_file_stat

        f = tmp_path / "nope.txt"
        f.write_text("x", encoding="utf-8")
        with patch(
            "msmodelslim.utils.security.path.is_belong_to_user_or_group",
            return_value=False,
        ):
            set_file_stat(str(f), "640")
        assert os.path.exists(f)
