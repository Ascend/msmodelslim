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

from datetime import timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.validation.conversion import (
    convert_to_bool,
    convert_to_timedelta,
    convert_to_readable_dir,
    convert_to_writable_dir,
    convert_to_readable_file,
)


class TestConvertToBool(TestCase):
    """测试 convert_to_bool 函数"""

    def test_convert_to_bool_return_true_when_input_is_true_bool(self):
        result = convert_to_bool(True)
        self.assertIs(result, True)

    def test_convert_to_bool_return_false_when_input_is_false_bool(self):
        result = convert_to_bool(False)
        self.assertIs(result, False)

    def test_convert_to_bool_return_true_when_input_is_true_string(self):
        result = convert_to_bool("True")
        self.assertIs(result, True)

    def test_convert_to_bool_return_false_when_input_is_false_string(self):
        result = convert_to_bool("False")
        self.assertIs(result, False)

    def test_convert_to_bool_raise_schema_validate_error_when_input_is_other_string(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_bool("yes")
        msg = str(cm.exception)
        self.assertIn("value must be a string or bool", msg)
        self.assertIn("Please ensure the input is literally True or False", msg)

    def test_convert_to_bool_raise_schema_validate_error_when_input_is_invalid_type(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_bool(1)  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("value must be a string or bool", msg)
        self.assertIn("<class 'int'>", msg)


class TestConvertToTimedelta(TestCase):
    """测试 convert_to_timedelta 函数"""

    def test_convert_to_timedelta_return_same_when_input_is_timedelta_instance(self):
        td = timedelta(days=1, hours=2, minutes=3, seconds=4)
        result = convert_to_timedelta(td)
        self.assertIs(result, td)

    def test_convert_to_timedelta_return_expected_when_input_is_full_pattern_string(self):
        result = convert_to_timedelta("1D2H30M15S")
        self.assertEqual(result, timedelta(days=1, hours=2, minutes=30, seconds=15))

    def test_convert_to_timedelta_return_expected_when_input_is_minutes_only_string(self):
        result = convert_to_timedelta("30M")
        self.assertEqual(result, timedelta(minutes=30))

    def test_convert_to_timedelta_return_expected_when_input_is_seconds_only_string(self):
        result = convert_to_timedelta("45S")
        self.assertEqual(result, timedelta(seconds=45))

    def test_convert_to_timedelta_return_expected_when_input_is_hours_only_string(self):
        result = convert_to_timedelta("2H")
        self.assertEqual(result, timedelta(hours=2))

    def test_convert_to_timedelta_return_expected_when_input_is_days_only_string(self):
        result = convert_to_timedelta("3D")
        self.assertEqual(result, timedelta(days=3))

    def test_convert_to_timedelta_raise_schema_validate_error_when_input_is_invalid_format_string(self):
        for invalid in ("", "abc", "1X", "1D2X", "DH", "1D-2H"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SchemaValidateError) as cm:
                    convert_to_timedelta(invalid)
                msg = str(cm.exception)
                self.assertIn("value has invalid timedelta format", msg)

    def test_convert_to_timedelta_raise_schema_validate_error_when_input_is_invalid_type(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_timedelta(123)  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("value must be a string or timedelta", msg)
        self.assertIn("<class 'int'>", msg)


class TestConvertToReadableDir(TestCase):
    """测试 convert_to_readable_dir 函数"""

    @patch("msmodelslim.utils.validation.conversion.get_valid_read_path")
    def test_convert_to_readable_dir_return_path_when_input_is_valid_string(self, mock_get_valid_read_path):
        mock_get_valid_read_path.return_value = "/test/dir"
        result = convert_to_readable_dir("/test/dir")

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/dir"))
        mock_get_valid_read_path.assert_called_once_with("/test/dir", is_dir=True)

    @patch("msmodelslim.utils.validation.conversion.get_valid_read_path")
    def test_convert_to_readable_dir_return_path_when_input_is_path_object(self, mock_get_valid_read_path):
        mock_get_valid_read_path.return_value = "/test/dir"
        path_obj = Path("/test/dir")
        result = convert_to_readable_dir(path_obj)

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/dir"))
        mock_get_valid_read_path.assert_called_once_with(str(path_obj), is_dir=True)

    def test_convert_to_readable_dir_raise_schema_validate_error_when_input_is_invalid_type(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_readable_dir(123)  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("A readable dir must be a string or Path", msg)
        self.assertIn("<class 'int'>", msg)


class TestConvertToWritableDir(TestCase):
    """测试 convert_to_writable_dir 函数"""

    @patch("msmodelslim.utils.validation.conversion.get_write_directory")
    def test_convert_to_writable_dir_return_path_when_input_is_valid_string(self, mock_get_write_directory):
        mock_get_write_directory.return_value = "/test/dir"
        result = convert_to_writable_dir("/test/dir")

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/dir"))
        mock_get_write_directory.assert_called_once_with("/test/dir")

    @patch("msmodelslim.utils.validation.conversion.get_write_directory")
    def test_convert_to_writable_dir_return_path_when_input_is_path_object(self, mock_get_write_directory):
        mock_get_write_directory.return_value = "/test/dir"
        path_obj = Path("/test/dir")
        result = convert_to_writable_dir(path_obj)

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/dir"))
        mock_get_write_directory.assert_called_once_with(str(path_obj))

    def test_convert_to_writable_dir_raise_schema_validate_error_when_input_is_invalid_type(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_writable_dir(456)  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("A writable dir must be a string or Path", msg)
        self.assertIn("<class 'int'>", msg)


class TestConvertToReadableFile(TestCase):
    """测试 convert_to_readable_file 函数"""

    @patch("msmodelslim.utils.validation.conversion.get_valid_read_path")
    def test_convert_to_readable_file_return_path_when_input_is_valid_string(self, mock_get_valid_read_path):
        mock_get_valid_read_path.return_value = "/test/file.txt"
        result = convert_to_readable_file("/test/file.txt")

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/file.txt"))
        mock_get_valid_read_path.assert_called_once_with("/test/file.txt", is_dir=False)

    @patch("msmodelslim.utils.validation.conversion.get_valid_read_path")
    def test_convert_to_readable_file_return_path_when_input_is_path_object(self, mock_get_valid_read_path):
        mock_get_valid_read_path.return_value = "/test/file.txt"
        path_obj = Path("/test/file.txt")
        result = convert_to_readable_file(path_obj)

        self.assertIsInstance(result, Path)
        self.assertTrue(result.as_posix().endswith("test/file.txt"))
        mock_get_valid_read_path.assert_called_once_with(str(path_obj), is_dir=False)

    def test_convert_to_readable_file_raise_schema_validate_error_when_input_is_invalid_type(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_readable_file(789)  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("A readable file must be a string or Path", msg)
        self.assertIn("<class 'int'>", msg)

    def test_convert_to_readable_file_raise_schema_validate_error_when_input_is_dict(self):
        with self.assertRaises(SchemaValidateError) as cm:
            convert_to_readable_file({"path": "/test/file.txt"})  # type: ignore[arg-type]
        msg = str(cm.exception)
        self.assertIn("A readable file must be a string or Path", msg)
        self.assertIn("<class 'dict'>", msg)
