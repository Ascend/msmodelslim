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

Filesystem capacity helpers. They only measure; they never decide pass/fail.
"""

from __future__ import annotations

import os
import shutil
import stat
from numbers import Integral
from pathlib import Path
from typing import Union


def get_dir_size(root: Union[str, os.PathLike]) -> int:
    """Return total bytes of files under ``root``.

    Symbolic links are followed for files and directories; directories are
    pruned by inode so symlink cycles terminate. Any stat or walk error returns
    ``-1``, as does an empty string, invalid root, or a path that is not a
    directory.
    """
    if isinstance(root, str) and not root.strip():
        return -1
    try:
        root_path = Path(root)
        root_stat = os.stat(str(root_path))
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return -1
    if not stat.S_ISDIR(root_stat.st_mode):
        return -1

    total = 0
    seen_dirs = set()
    seen_files = set()

    def _onerror(error):
        raise error

    try:
        for dir_path, dir_names, file_names in os.walk(str(root_path), onerror=_onerror, followlinks=True):
            try:
                dir_stat = os.stat(dir_path)
            except OSError:
                return -1
            dir_identity = (
                (dir_stat.st_dev, dir_stat.st_ino)
                if dir_stat.st_ino
                else ("path", os.path.normcase(os.path.realpath(dir_path)))
            )
            if dir_identity in seen_dirs:
                dir_names[:] = []
                continue
            seen_dirs.add(dir_identity)
            for name in file_names:
                full_path = os.path.join(dir_path, name)
                try:
                    file_stat = os.stat(full_path)
                except OSError:
                    return -1
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                identity = (
                    (file_stat.st_dev, file_stat.st_ino)
                    if file_stat.st_ino
                    else ("path", os.path.normcase(os.path.realpath(full_path)))
                )
                if identity in seen_files:
                    continue
                seen_files.add(identity)
                total += file_stat.st_size
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return -1
    return total


def get_free_space(path: Union[str, os.PathLike]) -> int:
    """Return free bytes of the filesystem containing ``path``.

    If ``path`` does not exist, walk up to the nearest existing ancestor.
    Invalid paths and filesystem-query failures return ``-1``.
    """
    if isinstance(path, str) and not path.strip():
        return -1
    try:
        target = Path(path)
        while True:
            try:
                target.stat()
                break
            except FileNotFoundError:
                parent = target.parent
                if parent == target:
                    return -1
                target = parent
        free = shutil.disk_usage(str(target)).free
        if isinstance(free, bool) or not isinstance(free, Integral):
            return -1
        free_bytes = int(free)
        return free_bytes if free_bytes >= 0 else -1
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return -1
