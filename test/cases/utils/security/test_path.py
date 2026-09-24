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
        # SecurityError: The file ... doesn't exist or not a file.
        get_valid_read_path("./not_exist")
    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith ".json"
        get_valid_read_path(TEST_READ_FILE_NAME, extensions=".json")
    with pytest.raises(SecurityError):
        # SecurityError: The file ... exceeds size limitation of 1.
        get_valid_read_path(TEST_READ_FILE_NAME, size_max=1)


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
    getattr(os, 'geteuid', lambda: 0)() == 0,
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

    def test_get_valid_path_log_warning_when_symlink(self, tmp_path):
        """软链接路径应记录 warning 并继续使用真实路径"""
        from msmodelslim.utils.security.path import get_valid_path

        target = tmp_path / "real.txt"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        with patch("msmodelslim.utils.security.path.get_logger") as mock_logger:
            result = get_valid_path(str(link))
        assert os.path.isabs(result)
        assert mock_logger.return_value.warning.called

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

        if getattr(os, 'geteuid', lambda: 0)() != 0:
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

    def test_get_valid_write_path_raise_when_no_write_access(self, tmp_path):
        """已存在文件不可写时应抛出 SecurityError"""
        from msmodelslim.utils.security.path import get_valid_write_path

        no_write_file = tmp_path / "readonly.txt"
        no_write_file.write_text("x", encoding="utf-8")

        def _mock_access(path, mode):
            return path != str(no_write_file)

        with patch("msmodelslim.utils.security.path.os.access", side_effect=_mock_access):
            with pytest.raises(SecurityError, match="not writable"):
                get_valid_write_path(str(no_write_file))

    def test_check_write_directory_warn_when_root_and_foreign(self, tmp_path):
        """root 写入非属主目录时应记录 warning 并继续"""
        from msmodelslim.utils.security.path import check_write_directory

        if getattr(os, 'geteuid', lambda: 0)() != 0:
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


class TestSafeWriteUmask:
    """测试 msmodelslim.utils.security.SafeWriteUmask"""

    def test_context_manager_strips_group_other_write(self, tmp_path):
        """with 块内创建文件时应按 umask 去掉组/其他写权限"""
        from msmodelslim.utils.security.path import SafeWriteUmask

        target = tmp_path / "umask_ctx.txt"
        default_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fake_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP | stat.S_IWOTH | stat.S_IROTH  # 666
        with SafeWriteUmask(), os.fdopen(os.open(str(target), default_flags, mode=fake_mode), "w") as write_file:
            write_file.write("")
        assert os.stat(target).st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH) == 0

    def test_decorator_strips_group_other_write(self, tmp_path):
        """装饰器用法应同样生效"""
        from msmodelslim.utils.security.path import SafeWriteUmask

        target = tmp_path / "umask_dec.txt"

        @SafeWriteUmask
        def _write():
            default_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fake_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP | stat.S_IWOTH | stat.S_IROTH
            with os.fdopen(os.open(str(target), default_flags, mode=fake_mode), "w") as f:
                f.write("")

        _write()
        assert os.stat(target).st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH) == 0

    def test_umask_restored_after_exception(self):
        """异常退出后应恢复原 umask（with 与装饰器）"""
        from msmodelslim.utils.security.path import SafeWriteUmask

        before = os.umask(0)
        os.umask(before)
        try:
            with pytest.raises(RuntimeError):
                with SafeWriteUmask(umask=0o077):
                    raise RuntimeError("boom")
            after = os.umask(0)
            os.umask(after)
            assert after == before

            @SafeWriteUmask
            def _boom():
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError):
                _boom()
            after = os.umask(0)
            os.umask(after)
            assert after == before
        finally:
            os.umask(before)


# os.access always reports True for root, so permission checks can only be
# exercised on POSIX systems running as a non-root user.
_POSIX_NON_ROOT = pytest.mark.skipif(
    sys.platform.startswith("win") or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="requires POSIX permissions and a non-root user",
)


class TestCheckReadPermission:
    """check_read_permission: 仅校验读权限（R_OK）。"""

    @_POSIX_NON_ROOT
    def test_pass_when_file_readable(self, tmp_path):
        """可读文件应通过并返回 realpath"""
        from msmodelslim.utils.security.path import check_read_permission

        file_path = tmp_path / "readable.txt"
        file_path.write_text("data", encoding="utf-8")
        assert check_read_permission(str(file_path)) == os.path.realpath(str(file_path))

    @_POSIX_NON_ROOT
    def test_pass_when_path_not_exists(self, tmp_path):
        """路径不存在时应直接返回，不抛异常（存续性由 get_valid_read_path 负责）"""
        from msmodelslim.utils.security.path import check_read_permission

        check_read_permission(str(tmp_path / "no_such_path"))

    @_POSIX_NON_ROOT
    def test_pass_when_dir_readable_but_without_execute_bit(self, tmp_path):
        """444 目录具备读权限，不校验 x 位，应通过"""
        from msmodelslim.utils.security.path import check_read_permission

        dir_path = tmp_path / "readonly_dir"
        dir_path.mkdir()
        os.chmod(str(dir_path), 0o444)
        try:
            check_read_permission(str(dir_path))
        finally:
            os.chmod(str(dir_path), 0o700)

    @_POSIX_NON_ROOT
    def test_raise_when_file_not_readable(self, tmp_path):
        """000 文件应抛 SecurityError"""
        from msmodelslim.utils.security.path import check_read_permission

        file_path = tmp_path / "no_read.txt"
        file_path.write_text("data", encoding="utf-8")
        os.chmod(str(file_path), 0o000)
        try:
            with pytest.raises(SecurityError):
                check_read_permission(str(file_path))
        finally:
            os.chmod(str(file_path), 0o600)

    @_POSIX_NON_ROOT
    def test_raise_when_dir_not_readable(self, tmp_path):
        """000 目录应抛 SecurityError"""
        from msmodelslim.utils.security.path import check_read_permission

        dir_path = tmp_path / "no_read_dir"
        dir_path.mkdir()
        os.chmod(str(dir_path), 0o000)
        try:
            with pytest.raises(SecurityError):
                check_read_permission(str(dir_path))
        finally:
            os.chmod(str(dir_path), 0o700)


class TestCheckWritePermission:
    """check_write_permission: 校验输出路径的读+写权限。"""

    @_POSIX_NON_ROOT
    def test_pass_when_existing_dir_writable(self, tmp_path):
        """已存在且可写目录应通过"""
        from msmodelslim.utils.security.path import check_write_permission

        dir_path = tmp_path / "writable_dir"
        dir_path.mkdir()
        assert check_write_permission(str(dir_path)) == os.path.realpath(str(dir_path))

    @_POSIX_NON_ROOT
    def test_pass_when_target_missing_but_parent_writable(self, tmp_path):
        """目标不存在但父目录可写时应通过（允许后续创建）"""
        from msmodelslim.utils.security.path import check_write_permission

        check_write_permission(str(tmp_path / "new_dir" / "nested"))

    @_POSIX_NON_ROOT
    def test_raise_when_existing_path_not_writable(self, tmp_path):
        """已存在但只读（444）的路径应抛 SecurityError"""
        from msmodelslim.utils.security.path import check_write_permission

        file_path = tmp_path / "readonly.txt"
        file_path.write_text("data", encoding="utf-8")
        os.chmod(str(file_path), 0o444)
        try:
            with pytest.raises(SecurityError):
                check_write_permission(str(file_path))
        finally:
            os.chmod(str(file_path), 0o600)

    @_POSIX_NON_ROOT
    def test_raise_when_missing_target_under_readonly_parent(self, tmp_path):
        """目标不存在且最近存在祖先目录只读时应抛 SecurityError"""
        from msmodelslim.utils.security.path import check_write_permission

        parent = tmp_path / "readonly_parent"
        parent.mkdir()
        os.chmod(str(parent), 0o555)
        try:
            with pytest.raises(SecurityError):
                check_write_permission(str(parent / "save"))
        finally:
            os.chmod(str(parent), 0o700)


class TestPermissionChecksWithMockedPlatform:
    """用 mock 固定平台与 os.access 结果，使权限分支逻辑在任意平台都可验证。"""

    @staticmethod
    def _patch(access_result):
        """把平台置为 linux，并把 os.access 结果折叠成单个布尔值。"""
        return (
            patch("sys.platform", "linux"),
            patch("os.access", return_value=access_result),
        )

    def test_check_read_permission_raise_when_not_readable(self):
        from msmodelslim.utils.security.path import check_read_permission

        platform_patch, access_patch = self._patch(False)
        with platform_patch, access_patch, patch("os.path.exists", return_value=True):
            with pytest.raises(SecurityError):
                check_read_permission("/fake/dir/model")

    def test_check_read_permission_pass_when_readable(self):
        from msmodelslim.utils.security.path import check_read_permission

        platform_patch, access_patch = self._patch(True)
        target = "/fake/dir/model"
        with platform_patch, access_patch, patch("os.path.exists", return_value=True):
            assert check_read_permission(target) == os.path.realpath(target)

    def test_check_read_permission_skip_access_when_not_exists(self):
        from msmodelslim.utils.security.path import check_read_permission

        platform_patch, access_patch = self._patch(False)
        with platform_patch, access_patch as mock_access, patch("os.path.exists", return_value=False):
            check_read_permission("/fake/dir/no_such_model")
        mock_access.assert_not_called()

    def test_check_read_permission_skip_when_windows(self):
        from msmodelslim.utils.security.path import check_read_permission

        with (
            patch("sys.platform", "win32"),
            patch("os.access", return_value=False),
            patch("os.path.exists", return_value=True),
        ):
            check_read_permission("/fake/dir/model")

    def test_check_write_permission_raise_when_existing_path_not_accessible(self):
        """既不可读也不可写时应抛 SecurityError。"""
        from msmodelslim.utils.security.path import check_write_permission

        platform_patch, access_patch = self._patch(False)
        with platform_patch, access_patch, patch("os.path.exists", return_value=True):
            with pytest.raises(SecurityError):
                check_write_permission("/fake/dir/save")

    def test_check_write_permission_raise_when_existing_path_readable_only(self):
        """可读但不可写（如 444 文件）也应抛 SecurityError，需同时具备读写权限。"""
        from msmodelslim.utils.security.path import check_write_permission

        with (
            patch("sys.platform", "linux"),
            patch("os.access", side_effect=lambda path, mode: mode == os.R_OK),
            patch("os.path.exists", return_value=True),
        ):
            with pytest.raises(SecurityError) as cm:
                check_write_permission("/fake/dir/save")

        assert "read and write permission" in str(cm.value)

    def test_check_write_permission_checks_read_and_write_together(self):
        """读写权限必须以合并掩码一次校验。"""
        from msmodelslim.utils.security.path import check_write_permission

        checked_modes = []

        def _access(path, mode):
            checked_modes.append(mode)
            return True

        with (
            patch("sys.platform", "linux"),
            patch("os.access", side_effect=_access),
            patch("os.path.exists", return_value=True),
        ):
            check_write_permission("/fake/dir/save")

        assert checked_modes == [os.R_OK | os.W_OK]

    def test_check_write_permission_pass_when_existing_path_rw(self):
        from msmodelslim.utils.security.path import check_write_permission

        target = "/fake/dir/save"
        platform_patch, access_patch = self._patch(True)
        with platform_patch, access_patch, patch("os.path.exists", return_value=True):
            assert check_write_permission(target) == os.path.realpath(target)

    def test_check_write_permission_probe_ancestor_when_target_missing(self):
        """目标不存在时应向上找到最近存在的祖先目录并校验其可写性。"""
        from msmodelslim.utils.security.path import check_write_permission

        target = "/fake/not_exist_dir/nested/save"
        real = os.path.realpath(target)
        ancestor = os.path.dirname(os.path.dirname(os.path.realpath(target)))

        checked_paths = []

        def _access(path, mode):
            checked_paths.append(path)
            return path == ancestor

        with (
            patch("sys.platform", "linux"),
            patch("os.access", side_effect=_access),
            patch("os.path.exists", side_effect=lambda p: p == ancestor),
        ):
            assert check_write_permission(target) == real

        assert checked_paths == [ancestor]

    def test_check_write_permission_raise_when_ancestor_not_writable(self):
        from msmodelslim.utils.security.path import check_write_permission

        target = "/fake/not_exist_dir/save"
        root = os.path.dirname(os.path.realpath(target))

        with (
            patch("sys.platform", "linux"),
            patch("os.access", return_value=False),
            patch("os.path.exists", side_effect=lambda p: p == root),
        ):
            with pytest.raises(SecurityError):
                check_write_permission(target)

    def test_check_write_permission_skip_when_windows(self):
        from msmodelslim.utils.security.path import check_write_permission

        with (
            patch("sys.platform", "win32"),
            patch("os.access", return_value=False),
            patch("os.path.exists", return_value=False),
        ):
            check_write_permission("/fake/dir/save")
