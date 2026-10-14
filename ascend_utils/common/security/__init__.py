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

from ascend_utils.common.security.type import (
    check_type,  # noqa: F401
    check_number,  # noqa: F401
    check_int,  # noqa: F401
    check_element_type,  # noqa: F401
    check_character,  # noqa: F401
    check_dict_character,  # noqa: F401
    check_dict_element,  # noqa: F401
)
from ascend_utils.common.security.path import (
    MAX_READ_FILE_SIZE_4G,  # noqa: F401
    MAX_READ_FILE_SIZE_32G,  # noqa: F401
    MAX_READ_FILE_SIZE_512G,  # noqa: F401
    get_valid_path,  # noqa: F401
    get_valid_read_path,  # noqa: F401
    get_valid_write_path,  # noqa: F401
    check_write_directory,  # noqa: F401
    get_write_directory,  # noqa: F401
    json_safe_load,  # noqa: F401
    json_safe_dump,  # noqa: F401
    yaml_safe_load,  # noqa: F401
    yaml_safe_dump,  # noqa: F401
    file_safe_write,  # noqa: F401
    safe_delete_path_if_exists,  # noqa: F401
    safe_copy_file,  # noqa: F401
    SafeWriteUmask,  # noqa: F401
    set_file_stat,  # noqa: F401
)
