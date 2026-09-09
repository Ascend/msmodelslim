#!/usr/bin/env bash
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
# post-create.sh - devcontainer 首次创建后的幂等初始化（登录 shell、非 root）。
# 经 init.sh 调用；run_step 分步输出 [STEP n/N] 与耗时，单项失败仅告警继续。
# 可重复执行；重装包 POST_CREATE_REINSTALL=1；日志 /tmp/msmodelslim-post-create.log。
# -------------------------------------------------------------------------
set -u

PROJ="${MSMODELSLIM_PROJ:-/workspace}"
VENV="${PROJ}/.venv"
LOG=/tmp/msmodelslim-post-create.log
TOTAL_STEPS=14
exec 2>>"${LOG}"

info() { printf '[post-create] %s\n' "$*"; }
warn() { printf '[post-create][WARN] %s\n' "$*"; }
has()  { command -v "$1" >/dev/null 2>&1; }

# run_step - 以统一格式执行单步初始化：打印序号/状态/耗时；失败仅告警并继续。
run_step() {
    local no="$1" title="$2" fn="$3" label exit_code started=$SECONDS
    printf -v label '%02d/%02d' "$no" "$TOTAL_STEPS"
    info "== [STEP ${label}] START | ${title}"
    if "$fn"; then
        info "   [STEP ${label}] DONE  | ${title} ($((SECONDS - started))s)"
    else
        exit_code=$?
        warn "   [STEP ${label}] FAILED | ${title} ($((SECONDS - started))s, exit=${exit_code}) —— continue"
    fi
    return 0
}

# openEuler 源切华为云镜像（仅当 repo 指向 openeuler/huaweicloud 时替换；带备份、幂等）。
configure_yum_mirror() {
    local repo_dir="/etc/yum.repos.d"
    local mirror_base="http://mirrors.huaweicloud.com/openeuler"
    local repo_files=("${repo_dir}"/*.repo)
    local repo_file backup_file changed=0
    if [ ! -e "${repo_files[0]}" ]; then
        info "no yum repo files in ${repo_dir}; skip mirror config"
        return 0
    fi
    for repo_file in "${repo_files[@]}"; do
        if ! grep -Eq 'https?://repo\.openeuler\.org|https://mirrors\.huaweicloud\.com/openeuler|https?://repo\.huaweicloud\.com/openeuler' "${repo_file}"; then
            continue
        fi
        backup_file="${repo_file}.post-create.bak"
        if [ ! -e "${backup_file}" ]; then
            sudo cp -a "${repo_file}" "${backup_file}" 2>/dev/null || { warn "backup yum repo failed: ${repo_file}"; return 1; }
        fi
        sudo sed -E -i \
            -e "s#https?://repo\.openeuler\.org#${mirror_base}#g" \
            -e "s#https://mirrors\.huaweicloud\.com/openeuler#${mirror_base}#g" \
            -e "s#https?://repo\.huaweicloud\.com/openeuler#${mirror_base}#g" \
            "${repo_file}" 2>/dev/null || { warn "update yum repo failed: ${repo_file}"; return 1; }
        changed=1
    done
    if ! grep -Eq '^[[:space:]]*baseurl=http://mirrors\.huaweicloud\.com/openeuler/' "${repo_files[@]}" 2>/dev/null; then
        warn "huaweicloud yum mirror not in active repo config"
        return 1
    fi
    if [ "${changed}" -eq 1 ]; then
        if has dnf; then
            sudo dnf clean all >/dev/null 2>&1 || warn "dnf clean all failed"
        elif has yum; then
            sudo yum clean all >/dev/null 2>&1 || warn "yum clean all failed"
        fi
        info "yum mirror switched to ${mirror_base}"
    else
        info "yum mirror already ${mirror_base}"
    fi
}

# 尽力 source 镜像登录 profile/工具集的 python/CANN 环境，供 venv 复用（缺失不阻塞）。
source_image_python_env() {
    for f in /etc/profile.d/*ascend* /etc/profile.d/*mindstudio* \
             /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        [ -f "${f}" ] && { info "sourcing image env: ${f}"; . "${f}" 2>/dev/null || true; }
    done
    if ! has python3; then
        warn "no python3 found in image"
        return 1
    fi
    info "python3 = $(command -v python3)  $(python3 --version 2>&1)"
}

# 提升 inotify 监听上限，避免 VS Code 在大仓（lab_calib 等）文件监听报错。
fix_file_watcher_limit() {
    local cur
    cur=$(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null || echo "")
    [ -z "${cur}" ] && { warn "cannot read inotify limit; skip"; return 0; }
    if [ "${cur:-0}" -ge 524288 ]; then
        info "inotify max_user_watches already ${cur}"
        return 0
    fi
    local cmd="sysctl -w fs.inotify.max_user_watches=524288"
    if [ "$(id -u)" = 0 ]; then
        $cmd >/dev/null 2>&1 && info "raised inotify limit to 524288" || warn "raise inotify failed"
    elif has sudo; then
        sudo $cmd >/dev/null 2>&1 && info "raised inotify limit to 524288" || warn "raise inotify failed (sudo)"
    else
        warn "not root & no sudo; inotify limit left ${cur}"
    fi
}

# 在工作区创建 .venv（bind mount 持久）；精简 UT 锁 torch==2.1.0 无 3.12 wheel，故对 3.12+ 告警。
ensure_venv() {
    if [ -x "${VENV}/bin/python" ]; then
        info "venv already present: ${VENV}"
    else
        info "creating venv ${VENV}"
        python3 -m venv "${VENV}" || { warn "venv creation failed"; return 1; }
    fi
    local pyver
    pyver=$("${VENV}/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "unknown")
    case "${pyver}" in
        3.8|3.9|3.10|3.11)
            info "venv python ${pyver} OK (reduced UT torch==2.1.0 supports it)"
            ;;
        3.12|3.13|3.14|3.15)
            warn "venv python ${pyver}: 精简 UT 的 test/requirements.txt 锁 torch==2.1.0（无此版本 wheel）。"
            warn "  请改用 'Install test deps (full UT)'（torch==2.7.1），或在镜像内使用 3.10/3.11 的 python3 重建 .venv。"
            ;;
        *)
            warn "venv python ${pyver} —— 请与 test/requirements.txt 的版本约束核对"
            ;;
    esac
    "${VENV}/bin/pip" install --upgrade pip >/dev/null 2>&1 \
        && info "pip upgraded" \
        || warn "pip upgrade failed (ignored)"
}

# 把仓库根 config/lab_calib/lab_practice 真实拷贝到 msmodelslim/（均 gitignore）。
# 刻意不用 editable 的符号链接：run_ut.sh 的 cp/chmod 穿透符号链接会报错或把
# 被跟踪的 config/config.ini 改成 100640 弄脏 git；chmod 仅作用于拷贝内。
ensure_data_dirs() {
    local d dst
    for d in config lab_calib lab_practice; do
        dst="${PROJ}/msmodelslim/${d}"
        if [ -L "${dst}" ]; then
            info "removing editable symlink ${dst}"
            rm -f "${dst}"
        fi
        if [ ! -d "${dst}" ]; then
            cp -r "${PROJ}/${d}" "${dst}" 2>/dev/null \
                && info "populated ${dst}" \
                || warn "failed to populate ${dst} (verify ${PROJ}/${d} exists)"
        fi
    done
    chmod 640 "${PROJ}/msmodelslim/config/config.ini" \
              "${PROJ}/msmodelslim/config/__init__.py" 2>/dev/null || true
}

# 非 editable 装进 venv（等价 install.sh）；时间戳标记幂等，setup.py/config.ini 更新后重装。
install_msmodelslim() {
    local marker
    marker="${VENV}/.msmodelslim-installed"
    if [ -f "${marker}" ] && [ "${POST_CREATE_REINSTALL:-0}" != 1 ] \
       && "${VENV}/bin/pip" show -q msmodelslim >/dev/null 2>&1 \
       && [ "${marker}" -nt "${PROJ}/setup.py" ] \
       && [ "${marker}" -nt "${PROJ}/config/config.ini" ]; then
        info "msmodelslim already installed (force reinstall with POST_CREATE_REINSTALL=1)"
        return 0
    fi
    info "installing msmodelslim into venv (pip install ., --no-cache-dir)"
    if (cd "${PROJ}" && umask 027 && "${VENV}/bin/pip" install . --no-cache-dir); then
        touch "${marker}"
        info "msmodelslim installed"
    else
        warn "pip install . failed —— 请重跑任务 'msmodelslim: Reinstall package'"
        return 1
    fi
}

# 建 ~/.local/bin 并把 PATH 写进 ~/.bashrc / ~/.profile（幂等，容器外 shell 也生效）。
configure_user_bin() {
    mkdir -p "${HOME}/.local/bin"
    local rc line
    line='export PATH="$HOME/.local/bin:$PATH"'
    for rc in "${HOME}/.bashrc" "${HOME}/.profile"; do
        touch "${rc}"
        if ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' "${rc}"; then
            printf '\n# msmodelslim devcontainer (post-create)\n%s\n' "${line}" >> "${rc}"
            info "appended PATH to ${rc}"
        fi
    done
    export PATH="${HOME}/.local/bin:${PATH}"
}

# pre-commit 的 local 钩子无二进制会 exit 1；尽力从华为 OBS 镜像站下载 gitleaks 到
# ~/.local/bin（on PATH），失败仅告警并提示 SKIP=gitleaks-offline-scan（勿把二进制放进仓库根）。
ensure_gitleaks() {
    if has gitleaks || [ -x "${PROJ}/gitleaks" ] || [ -x "${PROJ}/gitleaks.exe" ]; then
        info "gitleaks already available"
        return 0
    fi
    local arch url target="${HOME}/.local/bin/gitleaks"
    case "$(uname -m)" in
        x86_64|amd64)  arch=x86_64 ;;
        aarch64|arm64) arch=aarch64 ;;
        *)             arch=x86_64 ;;
    esac
    url="https://inst.obs.cn-north-4.myhuaweicloud.com/env/mirror/${arch}/gitleaks"
    info "downloading gitleaks (${arch}) from OBS -> ${target}"
    if mkdir -p "${HOME}/.local/bin" && \
       { (has wget && wget --no-check-certificate -q -O "${target}" "${url}" 2>/dev/null) || \
         (has curl && curl -fsSL "${url}" -o "${target}" 2>/dev/null); }; then
        chmod +x "${target}"
        info "gitleaks installed: $("${target}" version 2>&1 | head -1 || true)"
    else
        rm -f "${target}"
        warn "gitleaks download failed (offline); 使用 SKIP=gitleaks-offline-scan pre-commit run --all-files，或手动放入 PATH/仓库根"
    fi
}

install_pre_commit_hook() {
    if ! "${VENV}/bin/pip" show -q pre-commit >/dev/null 2>&1; then
        info "installing pre-commit (>=4.0.0) into venv"
        "${VENV}/bin/pip" install "pre-commit>=4.0.0" >/dev/null 2>&1 \
            || { warn "pre-commit install failed —— 编码规范要求 pre-commit"; return 1; }
    fi
    if (cd "${PROJ}" && "${VENV}/bin/pre-commit" install >/dev/null 2>&1); then
        info "pre-commit hook installed"
    else
        warn "pre-commit hook install failed —— 可手动执行任务 'msmodelslim: Lint (pre-commit all files)'"
    fi
}

# 后台预热 pre-commit Hook 环境（install-hooks，不执行检查）。缓存目录经 PRE_COMMIT_HOME
# 指向宿主 bind 缓存，重建后复用；用 nice/ionice 降优先级、flock 防重复，失败不影响使用。
warmup_pre_commit_async() {
    local pcb
    if [ -x "${VENV}/bin/pre-commit" ]; then
        pcb="${VENV}/bin/pre-commit"
    elif has pre-commit; then
        pcb="pre-commit"
    else
        warn "pre-commit not installed yet; skip warmup"
        return 0
    fi
    [ -f "${PROJ}/.pre-commit-config.yaml" ] || { warn "no pre-commit config; skip warmup"; return 0; }
    (cd "${PROJ}" && git rev-parse --show-toplevel >/dev/null 2>&1) || { warn "workspace not a git repo; skip warmup"; return 0; }
    local warm_dir="${HOME}/.cache" log_file
    warm_dir="${HOME}/.cache"; log_file="${warm_dir}/pre-commit-warmup.log"
    mkdir -p "${warm_dir}"
    (
        trap '' HUP
        cd "${PROJ}" || exit 0
        if has flock; then
            exec 9>"${warm_dir}/pre-commit-warmup.lock"
            flock -n 9 || { info "pre-commit warmup already running"; exit 0; }
        fi
        has ionice && ionice -c 3 -p "$$" 2>/dev/null || true
        export PATH="${HOME}/.local/bin:${PATH}"
        info "pre-commit environment warmup started (background)"
        nice -n 10 "${pcb}" install-hooks \
            && info "pre-commit environment warmup succeeded" \
            || warn "pre-commit environment warmup failed (见 ${log_file})"
    ) </dev/null >>"${log_file}" 2>&1 &
    info "pre-commit warmup launched in background (pid: $!, log: ${log_file})"
}

# 把宿主侧 .host-gitconfig（user.name/email 快照）并入容器内 git 身份；已有身份则跳过。
sync_git_identity() {
    local src name email
    src="${PROJ}/.devcontainer/.host-gitconfig"
    [ -f "${src}" ] || { info "no .host-gitconfig (host has no git user) —— skip"; return 0; }
    chmod 600 "${src}" 2>/dev/null || true
    if [ -n "$(git config --get user.email 2>/dev/null)" ]; then
        info "container git identity already set —— skip"
        return 0
    fi
    name=$(git config --file "${src}" --get user.name 2>/dev/null || true)
    email=$(git config --file "${src}" --get user.email 2>/dev/null || true)
    [ -n "${name}" ]  && git config --global user.name  "${name}"
    [ -n "${email}" ] && git config --global user.email "${email}"
    info "synced git identity ${name:+name=$name }${email:+email=$email}"
}

append_dev_hint_once() {
    grep -qF '# msmodelslim devcontainer hint' "${HOME}/.bashrc" && return 0
    cat >> "${HOME}/.bashrc" <<EOF

# msmodelslim devcontainer hint
alias msl-env='source ${PROJ}/.venv/bin/activate'
msl-ut() { cd ${PROJ} && source .venv/bin/activate && DEVICE_ID=0 python3 build.py test local; }
echo "msModelSlim dev: venv=${PROJ}/.venv  (alias msl-env, fn msl-ut)"
EOF
    info "dev hint appended to ${HOME}/.bashrc"
}

# 对随仓库提交的 .vscode/settings.json 设 skip-worktree，避免个人化本地改动污染 git status。
ignore_vscode_settings() {
    if (cd "${PROJ}" && git update-index --skip-worktree .vscode/settings.json 2>/dev/null); then
        info ".vscode/settings.json -> skip-worktree (取消: git update-index --no-skip-worktree .vscode/settings.json)"
    else
        warn "skip-worktree failed (文件未被 git 跟踪？)"
    fi
}

# 注册别名 git sync-up：临时放开 settings.json 的 skip-worktree 后拉取再恢复，规避远端更新冲突。
install_git_safe_pull_alias() {
    if (cd "${PROJ}" && git config --global alias.sync-up \
        '!f() { git update-index --no-skip-worktree .vscode/settings.json 2>/dev/null; git pull "$@"; git update-index --skip-worktree .vscode/settings.json 2>/dev/null; }; f' \
        2>/dev/null); then
        info "git alias sync-up registered (safe pull with skip-worktree settings)"
    else
        warn "register git alias sync-up failed"
    fi
}

print_ready_banner() {
    printf '\n%s\n' '========================================================'
    printf '  msModelSlim Dev Container    STATUS: READY\n'
    printf '  workspace = %s    log = %s\n' "${PROJ}" "${LOG}"
    printf '%s\n' '========================================================'
}

main() {
    info "== post-create start (user=$(id -un 2>/dev/null || echo '?'), HOME=${HOME:-?}, steps=${TOTAL_STEPS}) =="
    run_step 1  "Source image python/CANN env"          source_image_python_env
    run_step 2  "Switch to Huawei Cloud yum mirror"     configure_yum_mirror
    run_step 3  "Raise inotify file-watcher limit"      fix_file_watcher_limit
    run_step 4  "Ensure workspace venv"                 ensure_venv
    run_step 5  "Ensure data dirs under msmodelslim"    ensure_data_dirs
    run_step 6  "Install msmodelslim (non-editable)"    install_msmodelslim
    run_step 7  "Configure user bin PATH"               configure_user_bin
    run_step 8  "Ensure gitleaks binary"                ensure_gitleaks
    run_step 9  "Install pre-commit git hook"           install_pre_commit_hook
    run_step 10 "Warm up pre-commit env (background)"   warmup_pre_commit_async
    run_step 11 "Sync git identity"                     sync_git_identity
    run_step 12 "Append dev hints"                      append_dev_hint_once
    run_step 13 "Skip-worktree .vscode/settings.json"   ignore_vscode_settings
    run_step 14 "Register git alias sync-up"            install_git_safe_pull_alias
    info "== post-create finished (log ${LOG}) =="
    print_ready_banner
}

main "$@"
