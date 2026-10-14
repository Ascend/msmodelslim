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

import hashlib
import json

from pydantic import BaseModel


def calculate_md5(obj: BaseModel) -> str:
    """
    Calculate MD5 hash of a Pydantic BaseModel object.

    Args:
        obj: BaseModel object (e.g., PracticeConfig, EvaluateServiceConfig)

    Returns:
        str: MD5 hash string
    """
    # Use json mode so Decimal/datetime and similar values are JSON-serializable.
    obj_dict = obj.model_dump(mode='json')
    obj_json = json.dumps(obj_dict, sort_keys=True)
    return hashlib.md5(obj_json.encode('utf-8')).hexdigest()  # nosec B324
