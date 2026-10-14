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
# pylint: disable=no-member,pointless-string-statement

"""
测试用例执行脚本（与 Shell 联动版）
核心功能：
1. 双日志输出（控制台 + 时间戳文件），适配 Shell 脚本解析日志
2. 用例发现（按目录/模块筛选）、权限设置、顺序执行（并行预留接口）
3. 结构化输出（模块名/用例名/执行状态），便于 Shell 提取关键结果
4. 防异常处理（超时、编码错误、权限问题等），保证脚本稳定性

使用场景：
- 配合 Shell 脚本（如 run_st.sh）完成自动化测试流程
- 单独执行：python3 run_st.py -d ./test_cases -p "**/*.sh" -t 300
"""

import os  # noqa: E402
import sys  # noqa: E402
import subprocess  # noqa: E402
import argparse  # noqa: E402
import time  # noqa: E402
import signal  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Dict, Tuple  # noqa: E402
from typing import Optional  # noqa: E402, F401


# -------------------------- 工具函数：特殊字符过滤（适配 Shell/grep 解析） --------------------------
def sanitize_all(s: str, enable: bool = False) -> str:
    """
    过滤可能导致 Shell/grep 解析报错的特殊字符（如括号、管道符、通配符）

    Args:
        s: 需要过滤的字符串（如日志内容、文件路径）
        enable: 是否启用过滤（默认 False，因多数场景无需过滤，避免丢失原始信息）

    Returns:
        过滤后的字符串（空字符串或非字符串输入返回原内容）

    敏感字符说明：
    - ()[]{}：grep 正则元字符，可能导致匹配异常
    - |*?+^$\\：Shell 管道/通配符，可能触发意外命令执行
    - &<>!@#%~`"';:=：Shell 特殊符号，可能破坏命令语法
    """
    if not enable or not isinstance(s, str):
        return s

    # 定义需过滤的敏感字符集（按风险类型分组，便于维护）
    sensitive_chars = (
        r'()[]{}'  # 正则分组/范围符
        r'|*?+^$\\'  # 管道/通配符/正则量词
        r'&<>!@#%~`"'
        ';:='  # Shell 特殊操作符
    )

    # 替换所有敏感字符为空（也可替换为下划线，根据需求调整）
    for char in sensitive_chars:
        s = s.replace(char, '')

    return s.strip()  # 去除首尾空白，避免空行干扰


# -------------------------- 信号处理：防止 Shell 管道中断导致脚本崩溃 --------------------------
def handle_sigpipe():
    """
    忽略 SIGPIPE 信号（解决「Broken Pipe」错误）
    场景：当 Shell 用管道截取脚本输出时（如 `python3 run_st.py | head`），
    管道提前关闭会触发 SIGPIPE，默认处理方式是脚本崩溃，此处改为「忽略」以保证执行完成。
    """
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # SIG_DFL：使用系统默认处理（终止信号改为忽略）


# -------------------------- 核心类：测试用例执行器（适配 Shell 联动） --------------------------
class TestCaseExecutor:
    """
    测试用例执行器（与 Shell 脚本联动的核心类）
    核心特性：
    - 日志结构化：便于 Shell 提取日志路径、用例状态等关键信息
    - 异常全覆盖：处理权限、超时、编码、文件缺失等场景
    - 可扩展性：预留并行执行接口，支持模块筛选、权限自定义

    Attributes:
        base_dir: 用例基础目录（绝对路径）
        timeout: 单个用例超时时间（秒）
        log_file: 日志文件路径（时间戳命名，如 test_task_20241001123456.log）
        results: 用例执行结果列表（字典格式，含 idx/module/name/success/time）
    """

    def __init__(self, base_dir: str = ".", timeout: int = 300):
        """
        初始化执行器（检查目录有效性 + 创建日志文件 + 初始化配置）

        Args:
            base_dir: 用例基础目录（相对路径会转为绝对路径）
            timeout: 单个用例超时时间（默认 300 秒，即 5 分钟）

        Raises:
            NotADirectoryError: 若 base_dir 不存在或不是目录
        """
        # 处理基础目录：转为绝对路径 + 验证存在性
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.is_dir():
            raise NotADirectoryError(f"用例基础目录不存在或不是目录：{self.base_dir}")

        # 验证超时时间有效性
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError(f"超时时间必须为正数（当前：{self.timeout} 秒）")

        # 初始化结果存储
        self.results: List[Dict] = []

        # 创建日志文件（时间戳命名，避免覆盖；设置权限 644，允许其他用户读取）
        log_filename = f"test_task_{time.strftime('%Y%m%d%H%M%S')}.log"
        self.log_file = Path(log_filename).resolve()
        self.log_file.touch(exist_ok=True, mode=0o644)  # 0o644：所有者读写，其他只读

        # 初始化日志（输出关键配置，便于 Shell 解析）
        self.log_output("=" * 60)
        self.log_output("=== 测试任务初始化完成 ===")
        self.log_output(f"用例基础目录：{sanitize_all(str(self.base_dir))}")
        self.log_output(f"单个用例超时时间：{self.timeout} 秒")
        self.log_output(f"日志文件路径：{sanitize_all(str(self.log_file))}")
        self.log_output("日志特性：控制台与文件实时同步输出")
        self.log_output("=" * 60)

    def log_output(self, msg: str, end: str = "\n") -> None:
        """
        双日志输出：同时打印到控制台和日志文件（保证信息不丢失）

        Args:
            msg: 日志内容（支持任意可字符串化的对象）
            end: 行结尾符（默认换行，适配多行输出场景）

        编码说明：
        - 日志文件使用 UTF-8 编码，避免中文乱码
        - 若 msg 非字符串，自动转为字符串（如数字、异常对象）
        - 写入文件时忽略编码错误（用 'replace' 替换无法编码的字符）
        """
        # 统一转为字符串（处理非字符串输入，如异常对象）
        msg_str = str(msg) if not isinstance(msg, str) else msg

        # 1. 控制台输出（保持原格式）
        print(msg_str, end=end, flush=True)  # flush=True：确保实时输出，不缓存

        # 2. 文件日志输出（追加模式，UTF-8 编码）
        try:
            with open(self.log_file, "a", encoding="utf-8", errors="replace") as f:
                f.write(msg_str + end)
        except OSError as e:
            # 日志写入失败时，仅控制台告警（避免脚本崩溃）
            print(f"\n[WARNING] 日志文件写入失败：{e}（日志仅控制台可见）", flush=True)

    def _validate_perm_mode(self, mode: str) -> bool:
        """
        私有工具：验证权限模式是否合法（如 750、644，3 位数字且每位 0-7）

        Args:
            mode: 权限模式字符串（如 "750"）

        Returns:
            True：合法；False：非法
        """
        if len(mode) != 3:
            return False
        for c in mode:
            if not c.isdigit() or int(c) < 0 or int(c) > 7:
                return False
        return True

    def set_base_dir_permissions(self, mode: str = "750") -> None:
        """
        递归设置用例基础目录的权限（确保用例可执行、目录可访问）

        Args:
            mode: 权限模式（默认 750：所有者读写执行，组用户读执行，其他无权限）

        注意：
        - 仅当目录存在且权限模式合法时执行 chmod
        - 若 chmod 执行失败，仅日志告警（不中断后续流程）
        """
        # 1. 验证权限模式合法性
        if not self._validate_perm_mode(mode):
            self.log_output(f"[WARNING] 权限模式 {mode} 非法（需 3 位 0-7 数字），使用默认 750")
            mode = "750"

        # 2. 执行权限设置（调用系统 chmod 命令）
        safe_dir = sanitize_all(str(self.base_dir))
        self.log_output(f"\n[权限设置] 开始递归设置目录权限：{safe_dir} -> {mode}")

        try:
            # subprocess.run 捕获输出，避免干扰正常日志
            result = subprocess.run(  # noqa: F841
                ["chmod", "-R", mode, str(self.base_dir)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,  # 若 chmod 返回非 0 码，触发 CalledProcessError
            )
            self.log_output(f"[权限设置] 成功：{safe_dir} 权限已更新为 {mode}")
        except subprocess.CalledProcessError as e:
            # chmod 执行失败（如权限不足）
            err_msg = sanitize_all(e.stderr.strip())
            self.log_output(f"[权限设置] 失败（返回码：{e.returncode}）：{err_msg}")
            self.log_output(f"[权限设置] 建议：手动执行 `chmod -R {mode} {safe_dir}` 后重试")
        except FileNotFoundError:
            # 极端场景：系统无 chmod 命令（几乎不可能）
            self.log_output("[权限设置] 异常：未找到 chmod 命令（系统工具缺失）")
        except Exception as e:
            # 其他未知异常（如目录被删除）
            err_msg = sanitize_all(str(e))
            self.log_output(f"[权限设置] 异常：{err_msg}")

        self.log_output("-" * 50)

    def discover_test_cases(self, pattern: str = "**/*.sh") -> List[Path]:
        """
        从基础目录中发现可执行测试用例（按 glob 模式匹配）

        Args:
            pattern: 用例匹配模式（默认 **/*.sh：递归匹配所有 .sh 脚本）

        Returns:
            可执行用例路径列表（按「父目录名 -> 文件名」排序，保证执行顺序稳定）

        筛选规则：
        1. 符合 pattern 匹配规则
        2. 是文件（非目录）
        3. 拥有执行权限（os.X_OK）
        4. 不是符号链接（等效原 follow_symlinks=False 的效果）
        """
        safe_pattern = sanitize_all(pattern)
        safe_dir = sanitize_all(str(self.base_dir))
        self.log_output(f"\n[用例发现] 扫描范围：{safe_dir}，匹配模式：{safe_pattern}")

        test_cases: List[Path] = []
        try:
            # 先 glob 匹配，再过滤符号链接
            for file_path in self.base_dir.glob(pattern):
                # 排除utils目录
                if file_path.parent.resolve() == (self.base_dir / "utils").resolve():
                    continue
                # 筛选规则：
                # 1. 是文件（非目录）
                # 2. 不是符号链接
                # 3. 拥有执行权限
                if (
                    file_path.is_file()
                    and not file_path.is_symlink()  # 过滤符号链接
                    and os.access(file_path, os.X_OK)
                ):
                    test_cases.append(file_path)
        except OSError as e:
            # 扫描异常（如目录无访问权限）
            err_msg = sanitize_all(str(e))
            self.log_output(f"[用例发现] 扫描异常：{err_msg}（可能无目录访问权限）")
            return test_cases

        # 排序：按父目录名 -> 文件名升序，保证每次执行顺序一致（便于回归测试）
        test_cases.sort(key=lambda x: (x.parent.name.lower(), x.name.lower()))

        # 输出发现结果
        if not test_cases:
            self.log_output("[用例发现] 未找到符合条件的可执行用例（需 .sh 脚本、非符号链接且有执行权限）")
        else:
            self.log_output(f"[用例发现] 共找到 {len(test_cases)} 个可执行用例：")
            for idx, case in enumerate(test_cases, 1):
                # 输出相对路径（更简洁，便于定位）
                relative_path = case.relative_to(self.base_dir)
                safe_path = sanitize_all(str(relative_path))
                self.log_output(f"  {idx:2d}. {safe_path}")

        self.log_output("-" * 50)
        return test_cases

    def execute_single_case(self, test_case: Path, case_idx: int) -> Tuple[bool, float]:
        """
        执行单个测试用例，返回执行结果和耗时

        Args:
            test_case: 用例文件路径（绝对路径）
            case_idx: 用例序号（用于日志标识）

        Returns:
            Tuple[bool, float]：(执行成功与否, 耗时秒数)

        执行逻辑：
        1. 输出结构化用例信息（模块名/用例名/路径，便于 Shell 解析）
        2. 调用 subprocess 执行用例，捕获 stdout/stderr
        3. 处理异常：超时、执行失败、编码错误等
        4. 输出结构化执行结果（状态：成功/失败/超时/异常）
        """
        # 提取结构化信息（模块 = 用例所在父目录名）
        module_name = test_case.parent.name
        case_name = test_case.name
        safe_module = sanitize_all(module_name)
        safe_name = sanitize_all(case_name)
        safe_path = sanitize_all(str(test_case.resolve()))

        # 1. 输出用例基本信息（Shell 可通过 grep "模块名：" 提取关键信息）
        self.log_output(f"\n=== 用例 {case_idx} 执行详情 ===")
        self.log_output(f"模块名：{safe_module}")
        self.log_output(f"用例名：{safe_name}")
        self.log_output(f"用例路径：{safe_path}")
        self.log_output(f"开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()
        success = False  # 默认执行失败

        try:
            # 2. 执行用例（子进程运行，捕获输出，设置超时）
            result = subprocess.run(
                [str(test_case)],  # 用例命令（如 ./test_case.sh）
                capture_output=True,  # 捕获 stdout/stderr
                text=True,  # 输出转为字符串（而非字节流）
                timeout=self.timeout,  # 单个用例超时
                cwd=test_case.parent,  # 切换到用例所在目录（避免相对路径问题）
                encoding="utf-8",  # 输出编码
                errors="replace",  # 编码错误时替换为 �
                check=False,  # 不自动抛出 CalledProcessError（手动判断返回码）
            )

            # 3. 计算耗时，处理输出
            exec_time = round(time.time() - start_time, 2)
            stdout = sanitize_all(result.stdout.strip()) if result.stdout else "无"
            stderr = sanitize_all(result.stderr.strip()) if result.stderr else "无"

            # 4. 输出执行详情
            self.log_output(f"耗时：{exec_time} 秒")
            self.log_output(f"返回码：{result.returncode}  # 0=成功，非0=失败（Unix 命令规范）")
            self.log_output(f"标准输出：\n{stdout}")
            self.log_output(f"标准错误：\n{stderr}")

            # 5. 判断执行结果（返回码 0 视为成功）
            if result.returncode == 0:
                self.log_output("[执行结果] 状态：成功 ✅")
                success = True
            else:
                self.log_output("[执行结果] 状态：失败 ❌")

        except subprocess.TimeoutExpired:
            # 用例超时（subprocess 会自动杀死子进程）
            exec_time = round(time.time() - start_time, 2)
            self.log_output(f"耗时：{exec_time} 秒")
            self.log_output("[执行结果] 状态：超时 ⏰")
            self.log_output(f"超时原因：超过 {self.timeout} 秒限制")

        except FileNotFoundError:
            # 用例文件不存在（执行中被删除）
            exec_time = round(time.time() - start_time, 2)
            self.log_output(f"耗时：{exec_time} 秒")
            self.log_output("[执行结果] 状态：异常 ⚠")
            self.log_output("异常原因：用例文件不存在（可能被删除）")

        except Exception as e:
            # 其他未知异常（如权限突然被回收）
            exec_time = round(time.time() - start_time, 2)
            err_msg = sanitize_all(str(e))
            self.log_output(f"耗时：{exec_time} 秒")
            self.log_output("[执行结果] 状态：异常 ⚠")
            self.log_output(f"异常原因：{err_msg}")

        # 用例分隔线（便于 Shell 按行分割不同用例的日志）
        self.log_output("=== 用例分隔线 ===")
        return success, exec_time

    def execute_sequentially(self, test_cases: List[Path]) -> bool:
        """
        顺序执行所有测试用例（失败不中断，保证完整统计）

        Args:
            test_cases: 用例路径列表（由 discover_test_cases 生成）

        Returns:
            bool：所有用例执行成功返回 True，否则返回 False
        """
        total = len(test_cases)
        self.log_output("\n=== 开始顺序执行用例 ===")
        self.log_output(f"总用例数：{total}，执行策略：失败不中断（便于完整统计）")
        self.results.clear()  # 清空历史结果

        all_passed = True  # 默认所有用例通过

        for case_idx, test_case in enumerate(test_cases, 1):
            # 执行单个用例
            success, exec_time = self.execute_single_case(test_case, case_idx)

            # 记录结果（便于后续统计）
            self.results.append(
                {
                    "idx": case_idx,
                    "module": sanitize_all(test_case.parent.name),
                    "name": sanitize_all(test_case.name),
                    "success": success,
                    "time": exec_time,
                }
            )

            # 若有任意用例失败，整体结果设为 False
            if not success:
                all_passed = False

        # 输出执行汇总（便于 Shell 提取总结果）
        self.log_output("\n=== 测试执行汇总 ===")
        passed = sum(1 for res in self.results if res["success"])
        failed = total - passed
        total_time = round(sum(res["time"] for res in self.results), 2)
        avg_time = round(total_time / total, 2) if total > 0 else 0.0

        self.log_output(f"总用例数：{total} | 通过：{passed} | 失败：{failed}")
        self.log_output(f"总耗时：{total_time} 秒 | 平均耗时：{avg_time} 秒/用例")
        self.log_output("=" * 60)

        return all_passed

    def execute_in_parallel(self, test_cases: List[Path]) -> bool:
        """
        并行执行测试用例（预留接口，当前未实现，自动降级为顺序执行）

        Args:
            test_cases: 用例路径列表

        Returns:
            bool：所有用例执行成功返回 True，否则返回 False

        1. 使用 concurrent.futures.ProcessPoolExecutor（避免 GIL 限制）
        2. 支持设置最大并行数（--max-workers 参数）
        3. 并行日志需加锁，避免输出混乱
        """
        self.log_output("\n[WARNING] 并行执行功能暂未实现（预留接口），自动切换为顺序执行")
        return self.execute_sequentially(test_cases)


# -------------------------- 主函数：参数解析 + 执行流程入口 --------------------------
def main():
    # 1. 初始化信号处理（防止管道中断）
    handle_sigpipe()

    # -------------------------- 动态计算默认st_pr路径 --------------------------
    # 获取当前脚本（run_st.py）的绝对路径（不受执行目录影响）
    script_abs_path = Path(__file__).resolve()
    # 获取脚本所在目录（即 test 目录，因为 run_st.py 在 test 下）
    test_dir = script_abs_path.parent
    # 动态拼接 st_pr 目录的绝对路径（test/st_pr）
    default_base_dir = test_dir / "st_pr"
    # --------------------------------------------------------------------------------

    # 2. 解析命令行参数（适配 Shell 传参）
    parser = argparse.ArgumentParser(
        description="测试用例执行脚本（与 Shell 联动版）",
        formatter_class=argparse.RawTextHelpFormatter,  # 保留帮助信息的换行
    )
    parser.add_argument(
        '--base-dir', '-d', default=str(default_base_dir), help='用例基础目录（默认：st_pr，相对路径会转为绝对路径）'
    )
    parser.add_argument(
        '--pattern', '-p', default='**/*.sh', help='用例匹配模式（默认：**/*.sh，递归匹配所有 .sh 脚本）'
    )
    parser.add_argument('--timeout', '-t', type=int, default=600, help='单个用例超时时间（默认：600 秒，必须为正数）')
    parser.add_argument('--parallel', '-par', action='store_true', help='启用并行执行（暂未实现，默认顺序执行）')
    parser.add_argument(
        '--modules', '-m', nargs='+', help='指定执行的模块列表（模块 = 用例所在父目录名，如：-m module1 module2）'
    )
    parser.add_argument(
        '--perm-mode', '-per', default='750', help='用例目录递归权限模式（默认：750，需 3 位 0-7 数字，如 644、777）'
    )
    args = parser.parse_args()

    # 3. 核心执行流程（加异常捕获，避免脚本崩溃无日志）
    try:
        # 初始化执行器
        executor = TestCaseExecutor(base_dir=args.base_dir, timeout=args.timeout)

        # 设置目录权限（可选，默认执行）
        executor.set_base_dir_permissions(mode=args.perm_mode)

        # 发现用例
        test_cases = executor.discover_test_cases(pattern=args.pattern)
        if not test_cases:
            executor.log_output("\n[ERROR] 未发现可执行用例，脚本退出")
            sys.exit(1)

        # 按模块筛选用例（若指定 --modules）
        if args.modules:
            safe_modules = [sanitize_all(m) for m in args.modules]
            executor.log_output(f"\n[模块筛选] 指定执行模块：{safe_modules}")

            # 筛选规则：用例的父目录名在指定模块列表中
            filtered_cases = [case for case in test_cases if sanitize_all(case.parent.name) in safe_modules]

            if not filtered_cases:
                executor.log_output(f"[ERROR] 指定模块 {safe_modules} 无可用用例，脚本退出")
                sys.exit(1)

            test_cases = filtered_cases
            executor.log_output(f"[模块筛选] 筛选后剩余 {len(test_cases)} 个用例")

        # 执行用例（并行/顺序）
        if args.parallel:
            all_passed = executor.execute_in_parallel(test_cases)
        else:
            all_passed = executor.execute_sequentially(test_cases)

        # 4. 退出脚本（返回码：0=成功，1=失败，适配 Shell 判断）
        sys.exit(0 if all_passed else 1)

    except (ValueError, NotADirectoryError) as e:
        # 初始化阶段的已知异常（参数错误、目录不存在）
        print(f"\n[ERROR] 初始化失败：{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # 未知异常（兜底处理，输出日志后退出）
        print(f"\n[ERROR] 脚本异常退出：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
