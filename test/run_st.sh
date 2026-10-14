#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

# 严格模式：遇到错误立即退出、未定义变量报错、管道命令失败则整体失败
set -euo pipefail

##############################################################################
# @Description  一键执行测试用例（纯功能精简版）
# @Core Logic   核心流程：必要环境检查 → 前置依赖安装 → 调用Python执行测试
# @Usage        ./run_st.sh [Python脚本参数]
#               示例：./run_st.sh -d ./cases -m module1 （指定用例目录和模块）
# @Required Env 必须提前设置的环境变量（无默认值，缺失则报错）
#               1. MODEL_RESOURCE_PATH：模型资源文件存放路径
#               2. OUTPUT_RESOURCE_PATH：测试结果输出路径
# @Exit Codes   0：所有测试用例执行成功
#               1：脚本初始化/依赖检查/安装失败，或Python测试执行失败
##############################################################################

# ============================== 全局配置 ==============================
# 终端颜色定义（统一视觉提示，避免混乱）
declare -r RED='\033[0;31m'                                                     # 错误信息
declare -r GREEN='\033[0;32m'                                                   # 成功信息
declare -r YELLOW='\033[1;33m'                                                  # 提示/步骤信息
declare -r NC='\033[0m'                                                         # 重置终端颜色（避免影响后续命令）

# 核心文件路径（脚本运行依赖的关键文件，路径基于当前脚本所在目录计算）
declare -r SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd) # 当前脚本所在绝对路径
declare -r CORE_PY_SCRIPT="${SCRIPT_DIR}/run_st.py"                             # 测试执行核心Python脚本
declare -r PRE_INSTALL_SCRIPT="${SCRIPT_DIR}/../install.sh"                     # 前置依赖安装脚本

# ============================== 工具函数（模块化，便于复用和维护） ==============================
##############################################################################
# @Function     check_required_env
# @Description  检查必要环境变量是否已定义（无默认值，缺失则报错退出）
# @Parameters   $1: 环境变量名（如 MODEL_RESOURCE_PATH）
#               $2: 环境变量的用途描述（如 "模型资源路径"）
# @Exit         1：环境变量未定义或为空；0：环境变量正常
##############################################################################
check_required_env() {
  local env_name="$1"
  local env_desc="$2"

  # ${!env_name}：间接引用环境变量（通过变量名获取变量值）；-z 检查值是否为空
  if [[ -z "${!env_name:-}" ]]; then
    echo -e "\n${RED}[ERROR] 必要环境变量「${env_name}」未定义（用途：${env_desc}）！${NC}"
    echo -e "${YELLOW}  请先执行以下命令设置（替换为实际路径）：${NC}"
    echo -e "    export ${env_name}=/your/actual/path"
    exit 1
  fi

  # 导出环境变量，确保子进程（如Python脚本、install.sh）可访问
  export "${env_name}"
  echo -e "${GREEN}[OK] 环境变量「${env_name}」已就绪（值：${!env_name}）${NC}"
}

##############################################################################
# @Function     check_command
# @Description  检查核心命令是否存在（确保执行环境具备必要工具）
# @Parameters   $1: 命令名（如 python3、bash）
#               $2: 命令的用途描述（如 "Python 3解释器"）
# @Exit         1：命令不存在；0：命令正常
##############################################################################
check_command() {
  local cmd="$1"
  local cmd_desc="$2"

  # command -v：检查命令是否在PATH中，&>/dev/null 屏蔽无关输出
  if ! command -v "${cmd}" &>/dev/null; then
    echo -e "\n${RED}[ERROR] 缺失必要工具「${cmd_desc}」（命令：${cmd}）${NC}"
    echo -e "${YELLOW}  建议安装方式：${NC}"
    case "${cmd}" in
    python3) echo -e "    Ubuntu/Debian: sudo apt install python3; CentOS/RHEL: sudo yum install python3" ;;
    bash) echo -e "    Ubuntu/Debian: sudo apt install bash; CentOS/RHEL: sudo yum install bash" ;;
    chmod) echo -e "    该命令属于coreutils，通常已预装，若缺失请安装：sudo apt install coreutils 或 sudo yum install coreutils" ;;
    *) echo -e "    请通过系统包管理器安装「${cmd}」" ;;
    esac
    exit 1
  fi
}

##############################################################################
# @Function     check_file_exists
# @Description  检查核心文件是否存在（确保脚本依赖的文件未丢失）
# @Parameters   $1: 文件路径（绝对路径或相对路径）
#               $2: 文件的用途描述（如 "Python测试执行脚本"）
# @Exit         1：文件不存在；0：文件正常
##############################################################################
check_file_exists() {
  local file_path="$1"
  local file_desc="$2"

  if [[ ! -f "${file_path}" ]]; then
    echo -e "\n${RED}[ERROR] 核心文件缺失（${file_desc}）：${file_path}${NC}"
    echo -e "${YELLOW}  请确认文件路径正确，或重新获取该文件${NC}"
    exit 1
  fi
}

# ============================== 核心执行流程 ==============================
# ------------------------------------------------------------------------------
# 步骤1：初始化检查（必要环境变量 + 核心命令）
# 目的：提前排除基础环境问题，避免后续步骤白执行
# ------------------------------------------------------------------------------
echo -e "${YELLOW}\n================================ 步骤1/3：初始化环境检查 ================================${NC}"
# 检查必须的环境变量（模型路径 + 输出路径）
check_required_env "MODEL_RESOURCE_PATH" "模型资源文件存放路径"
check_required_env "OUTPUT_RESOURCE_PATH" "测试结果输出路径"

# 检查必须的执行工具（确保Shell、Python、权限工具可用）
echo -e "\n${YELLOW}▶ 检查核心执行工具...${NC}"
check_command "bash" "Bash解释器（脚本运行基础）"
check_command "python3" "Python 3解释器（执行测试脚本）"
check_command "chmod" "权限设置工具（确保安装脚本可执行）"
echo -e "${GREEN}\n[OK] 初始化环境检查通过${NC}"

# ------------------------------------------------------------------------------
# 步骤2：前置依赖安装
# 目的：自动安装Python依赖、系统库等，避免测试执行时因依赖缺失失败
# 逻辑：找到install.sh则执行，无则提示跳过（需手动处理依赖）
# ------------------------------------------------------------------------------
echo -e "${YELLOW}\n================================ 步骤2/3：前置依赖安装 ================================${NC}"
# 检查安装脚本是否存在
if [[ -f "${PRE_INSTALL_SCRIPT}" ]]; then
  echo -e "${YELLOW}▶ 发现前置安装脚本：${PRE_INSTALL_SCRIPT}${NC}"

  # 保存当前工作目录
  original_dir=$(pwd)
  # 安装脚本所在目录
  install_dir=$(dirname "${PRE_INSTALL_SCRIPT}")

  # 切换到安装目录（若失败则恢复原目录并退出）
  echo -e "${YELLOW}▶ 切换到安装目录：${install_dir}${NC}"
  if ! cd "${install_dir}"; then
    echo -e "${RED}[ERROR] 无法切换到安装目录：${install_dir}${NC}"
    cd "${original_dir}" || exit 1
    exit 1
  fi

  # 给安装脚本添加执行权限（避免因权限不足无法运行）
  if [[ ! -x "install.sh" ]]; then
    echo -e "${YELLOW}▶ 自动添加执行权限：chmod +x install.sh${NC}"
    if ! chmod +x "install.sh"; then
      echo -e "${RED}[ERROR] 无法给install.sh添加执行权限${NC}"
      echo -e "${YELLOW}  请手动执行安装：bash ${PRE_INSTALL_SCRIPT}${NC}"
      cd "${original_dir}" || exit 1
      exit 1
    fi
  fi

  # 执行安装脚本（若失败则恢复原目录并退出）
  echo -e "${YELLOW}▶ 开始执行依赖安装...${NC}"
  if ! bash "install.sh"; then
    echo -e "${RED}[ERROR] 前置依赖安装失败，无法继续执行测试${NC}"
    cd "${original_dir}" || exit 1
    exit 1
  fi

  # 切回原工作目录（不影响用户后续操作）
  echo -e "${YELLOW}▶ 切换回原目录：${original_dir}${NC}"
  if ! cd "${original_dir}"; then
    echo -e "${RED}[ERROR] 无法切换回原目录：${original_dir}${NC}"
    exit 1
  fi

  echo -e "${GREEN}[OK] 前置依赖安装完成${NC}"
else
  echo -e "${YELLOW}▶ 未找到前置安装脚本（${PRE_INSTALL_SCRIPT}），跳过安装${NC}"
  echo -e "${YELLOW}  提示：若后续测试执行失败，请手动安装Python依赖（如 pip install -r requirements.txt）${NC}"
fi

# ------------------------------------------------------------------------------
# 步骤3：执行测试用例（核心功能）
# 目的：调用Python脚本执行测试，传递用户参数，保留执行结果码
# ------------------------------------------------------------------------------
echo -e "${YELLOW}\n================================ 步骤3/3：执行测试用例 ================================${NC}"
# 检查Python核心脚本是否存在
check_file_exists "${CORE_PY_SCRIPT}" "Python测试执行脚本"

# 打印执行命令（方便用户排查参数传递问题）
echo -e "${YELLOW}▶ 执行命令：python3 ${CORE_PY_SCRIPT} $@${NC}\n"

# 调用Python脚本执行测试（$@ 保留用户输入的所有参数，支持含空格的参数）
python3 "${CORE_PY_SCRIPT}" "$@"
# 保存Python脚本的退出码（0=成功，非0=失败）
test_exit_code=$?

# ============================== 结果汇总 ==============================
echo -e "\n${YELLOW}================================ 测试执行结果 ================================${NC}"
if [[ "${test_exit_code}" -eq 0 ]]; then
  echo -e "${GREEN}[SUCCESS] 所有测试用例执行通过！${NC}"
else
  echo -e "${RED}[FAIL] 部分/全部测试用例执行失败（详情请查看Python脚本输出）${NC}"
fi
echo -e "${YELLOW}执行结果码：${test_exit_code}（0=成功，非0=失败）${NC}"

# 传递退出码给外部脚本（方便CI/CD等外部系统判断执行结果）
exit "${test_exit_code}"
