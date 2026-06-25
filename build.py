#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import argparse
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BuildManager:
    """
    统一构建管理：依赖拉取 → 编译 → 安装 / 测试。

    用法:
        python build.py                             完整构建（拉取依赖 + Release 编译）
        python build.py local                       本地构建（跳过依赖拉取, Release 编译）
        python build.py test                        单元测试（拉取依赖 + Debug 编译 + 执行测试）
        python build.py test local                  单元测试（跳过依赖拉取, Debug 编译 + 执行测试）
        python build.py --version/-v <version>      指定构建版本号（用于 run/exe/dmg 包）
        python build.py --extra/-e KEY=VALUE        指定额外构建选项，可多次使用

    参数说明:
        - 参数: command    : 构建动作: 为空时为全构建, local 为跳过依赖下载, test 为运行单元测试。
        - 参数: --version  : 构建版本号，不传时默认 1.0.0。
        - 参数: --extra    : 额外构建选项，格式为 KEY=VALUE，可多次指定。
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        ap = argparse.ArgumentParser(description='Build the project and optionally run tests.')
        ap.add_argument(
            'command',
            nargs='*',
            default=[],
            choices=[[], 'local', 'test'],
            help='Build action: omit for full build, "local" to skip dependency download, "test" to run unit tests',
        )
        ap.add_argument(
            '-v', '--version', type=str, default='1.0.0', help='Build version for run/exe/dmg packages (default: 1.0.0)'
        )
        ap.add_argument(
            '-e',
            '--extra',
            metavar='KEY=VALUE',
            action='append',
            default=[],
            help='Extra build options in KEY=VALUE format, can be specified multiple times',
        )
        self.args = ap.parse_args()

    def _execute_command(self, cmd, timeout_seconds=36000, cwd=None, env=None):
        logging.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, timeout=timeout_seconds, check=True, cwd=cwd, env=env)

    def _archive_artifacts(self):
        artifact_patterns = ("*.whl",)
        dist_dir = self.project_root / "dist"
        artifacts_dir = self.project_root / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for pattern in artifact_patterns:
            for artifact in dist_dir.glob(pattern):
                destination = artifacts_dir / artifact.name
                logging.info("Archiving: %s -> %s", artifact, destination)
                shutil.copy2(artifact, destination)

    def _prepare_runtime_dependencies(self):
        """安装运行时依赖（来源 requirements.txt，由 install.sh 完成）。"""
        self._execute_command(["bash", "install.sh"], cwd=self.project_root)

    def _prepare_test_dependencies(self):
        """安装测试专用依赖（在运行时依赖之上额外安装，来源 test/requirements.txt）。"""
        self._execute_command(
            ["pip", "install", "-r", str(self.project_root / "test" / "requirements.txt")],
            cwd=self.project_root,
        )

    def _check_torch_npu_conflict(self):
        """检测环境是否安装了 torch_npu，存在则告警。

        UT 基于“纯 torch + mock torch_npu”运行，若环境真实安装了 torch_npu，
        测试用例中的 mock 兜底逻辑会被跳过，torch_npu 指向真实 NPU 后端，
        在无 NPU 硬件的 UT 环境下可能导致用例失败或行为异常。
        此处仅告警，不卸载，避免隐式修改系统级环境。
        """
        if importlib.util.find_spec("torch_npu") is not None:
            logging.warning(
                "torch_npu is installed in the current environment. "
                "UT runs on pure torch with mocked torch.npu; "
                "a real torch_npu may cause test failures."
            )

    def run(self):
        os.chdir(self.project_root)
        is_local = 'local' in self.args.command
        is_test = 'test' in self.args.command

        # 在非 local 场景下按需更新依赖；在 local 场景下仅使用本地已有代码，不更新依赖。
        if not is_local:
            if is_test:
                # 测试：先安装测试专用依赖，再安装运行时依赖，可减少一些重复安装
                self._prepare_test_dependencies()
                self._prepare_runtime_dependencies()
            else:
                # 构建：setup.py bdist_wheel 直接读取 requirements.txt 作为 install_requires，无需预先安装
                pass

        if is_test:
            # -------------------- 单元测试 --------------------
            self._check_torch_npu_conflict()
            self._execute_command(["bash", "run_ut.sh", "--modelslim_v1"], cwd=self.project_root / "test")
        else:
            # -------------------- 产品构建 --------------------
            logging.info("--version: %s", self.args.version)
            for opt in self.args.extra:
                key, _, val = opt.partition('=')
                logging.info("--extra: %s = %s", key, val)

            build_env = os.environ.copy()
            build_env["BUILD_VERSION"] = self.args.version
            build_env["WHL_VERSION"] = self.args.version
            self._execute_command(["python3", "setup.py", "bdist_wheel"], cwd=self.project_root, env=build_env)
            self._archive_artifacts()


if __name__ == "__main__":
    try:
        BuildManager().run()
    except Exception:
        logging.error("Unexpected error: %s", traceback.format_exc())
        sys.exit(1)
