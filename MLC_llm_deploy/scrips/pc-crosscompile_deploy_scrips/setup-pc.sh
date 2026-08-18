#!/usr/bin/env bash
# ============================================================================
# setup-pc.sh — PC 交叉编译前置条件：检查 + 安装（crosscompile.sh 的前置工具）
#
# 用法:
#   ./setup-pc.sh check    # 只检查，不改系统，输出 ✅/❌ 清单和修复建议
#   ./setup-pc.sh install  # 检查 + 自动安装缺失项（uv/git-lfs/mlc_llm，sudo 步骤会提示）
#   ./setup-pc.sh build    # 构建 mlc-llm-jetson:cu126 镜像（约 2-4 小时）
#   ./setup-pc.sh all      # check → install → build 全流程
#
# 检查项: NVIDIA GPU / Docker / docker 组 / GPU 容器透传 / uvx /
#         git-lfs / 交叉编译镜像 / 磁盘 / 内存
# （mlc_llm 不装宿主机——转换与编译都在镜像内跑，保证与 v0.20.0 同源）
#
# 版本锚点: MLC-LLM v0.20.0 / CUDA 12.6 / sm_87（Orin）/ LLVM 20（镜像内）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="mlc-llm-jetson:cu126"
GPU_TEST_IMAGE="nvidia/cuda:12.6.3-base-ubuntu22.04"

# 检查结果记录: 名称=状态(ok/fail/warn)|详情|修复命令
RESULTS=()

record() { # record <名称> <ok|fail|warn> <详情> <修复命令或 "-">
    RESULTS+=("${1}|${2}|${3}|${4}")
}

fix_hint() { # 取记录里的修复命令，去掉 "-" 占位
    local fix="$1"
    [ "${fix}" = "-" ] && echo "" || echo "${fix}"
}

# ---------------------------------------------------------------------------
# 各检查函数（纯只读，不改动系统）
# ---------------------------------------------------------------------------

check_gpu() {
    if out="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"; then
        local vram
        vram="$(echo "${out}" | grep -oE '[0-9]+' | tail -1)"
        if [ "${vram:-0}" -ge 8000 ]; then
            record "NVIDIA GPU" ok "${out}" "-"
        else
            record "NVIDIA GPU" warn "${out}（显存 < 8GB，交叉编译可能吃紧）" "-"
        fi
    else
        record "NVIDIA GPU" fail "nvidia-smi 不可用" "安装 NVIDIA 驱动（sudo ubuntu-drivers autoinstall 或官网驱动）"
    fi
}

check_docker() {
    if command -v docker >/dev/null 2>&1; then
        record "Docker" ok "$(docker --version)" "-"
    else
        record "Docker" fail "未安装" "curl -fsSL https://get.docker.com | sudo sh"
    fi
}

check_docker_group() {
    command -v docker >/dev/null 2>&1 || { record "docker 组" warn "Docker 未安装，跳过" "-"; return; }
    if id -nG "$(id -un)" 2>/dev/null | tr ' ' '\n' | grep -qw docker; then
        record "docker 组" ok "当前用户可直接运行 docker（无需 sudo）" "-"
    else
        record "docker 组" warn "docker 命令需要 sudo" "sudo usermod -aG docker \$USER && 重新登录"
    fi
}

# docker 命令前缀：用户不在 docker 组则加 sudo
docker_cmd() {
    if id -nG "$(id -un)" 2>/dev/null | tr ' ' '\n' | grep -qw docker; then
        echo "docker"
    else
        echo "sudo docker"
    fi
}

check_gpu_passthrough() {
    command -v docker >/dev/null 2>&1 || { record "GPU 容器透传" warn "Docker 未安装，跳过" "-"; return; }
    local DK; DK="$(docker_cmd)"
    echo "   （拉取/运行 ${GPU_TEST_IMAGE} 测试中，首次约 200MB，请稍候...）" >&2
    local err
    if err="$(timeout 120 ${DK} run --rm --gpus all ${GPU_TEST_IMAGE} nvidia-smi 2>&1)"; then
        record "GPU 容器透传" ok "--gpus all 容器内可见 GPU" "-"
    else
        # 区分: runtime 缺失 vs 网络拉镜像失败
        if echo "${err}" | grep -qiE "could not select device driver|unknown flag"; then
            record "GPU 容器透传" fail "缺少 nvidia-container-toolkit（--gpus 参数不被识别）" "sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker（或直接: ./setup-pc.sh install）"
        elif echo "${err}" | grep -qiE "connection refused|timeout|no route|TLS handshake"; then
            record "GPU 容器透传" fail "测试镜像拉取失败（网络问题）" "检查网络/配置 docker 镜像加速器或代理（daemon.json 的 registry-mirrors / proxies）"
        else
            record "GPU 容器透传" fail "测试容器运行失败: $(echo "${err}" | head -1 | cut -c1-60)" "docker run --rm --gpus all ${GPU_TEST_IMAGE} nvidia-smi 手动复现查看完整报错"
        fi
    fi
}

check_uvx() {
    if command -v uvx >/dev/null 2>&1; then
        record "uvx" ok "$(uvx --version)" "-"
    else
        record "uvx" fail "未安装" "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc"
    fi
}

check_gitlfs() {
    if command -v git-lfs >/dev/null 2>&1; then
        record "git-lfs" ok "$(git-lfs version | cut -d' ' -f1-2)" "-"
    else
        record "git-lfs" fail "未安装" "sudo apt-get install -y git-lfs && git lfs install"
    fi
}

check_image() {
    command -v docker >/dev/null 2>&1 || { record "交叉编译镜像" warn "Docker 未安装，跳过" "-"; return; }
    local DK; DK="$(docker_cmd)"
    if [ -n "$(${DK} images -q ${IMAGE} 2>/dev/null)" ]; then
        record "交叉编译镜像" ok "${IMAGE} 已构建" "-"
    else
        record "交叉编译镜像" fail "${IMAGE} 未构建（crosscompile.sh 第③步依赖）" "./setup-pc.sh build（约 2-4 小时，约 20GB 磁盘）"
    fi
}

check_disk() {
    local avail_gb
    avail_gb="$(df -BG --output=avail "${SCRIPT_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [ "${avail_gb:-0}" -ge 50 ]; then
        record "磁盘余量" ok "${avail_gb}GB（≥ 50GB）" "-"
    else
        record "磁盘余量" fail "${avail_gb}GB < 50GB（镜像 ~20GB + 模型权重 + 编译产物）" "清理磁盘或换工作目录"
    fi
}

check_mem() {
    local total_gb
    total_gb="$(free -g | awk '/^Mem:/{print $2}')"
    if [ "${total_gb:-0}" -ge 16 ]; then
        record "系统内存" ok "${total_gb}GB（≥ 16GB）" "-"
    else
        record "系统内存" warn "${total_gb}GB < 16GB（构建 TVM/MLC 时可能吃紧，可加 swap）" "-"
    fi
}

do_check() {
    echo "== PC 交叉编译环境检查（只读，不改系统）=="
    check_gpu
    check_docker
    check_docker_group
    check_gpu_passthrough
    check_uvx
    check_gitlfs
    check_image
    check_disk
    check_mem

    echo ""
    echo "== 检查结果汇总 =="
    local fails=0 warns=0 name status detail fix
    printf "  %-4s %-16s %-8s %s\n" "" "检查项" "状态" "详情"
    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r name status detail fix <<< "${entry}"
        local mark
        case "${status}" in
            ok)   mark="✅"; ;;
            warn) mark="⚠️ "; warns=$((warns+1)) ;;
            fail) mark="❌"; fails=$((fails+1)) ;;
        esac
        printf "  %-4s %-16s %-8s %s\n" "${mark}" "${name}" "${status}" "${detail}"
        if [ "${status}" != "ok" ]; then
            local hint; hint="$(fix_hint "${fix}")"
            [ -n "${hint}" ] && printf "  %-4s %s\n" "" "→ 修复: ${hint}"
        fi
    done
    echo ""
    if [ "${fails}" -eq 0 ] && [ "${warns}" -eq 0 ]; then
        echo "  全部通过 ✅  可以运行: ./crosscompile.sh"
    elif [ "${fails}" -eq 0 ]; then
        echo "  无缺失项（${warns} 个警告）✅  基本可以运行: ./crosscompile.sh"
    else
        echo "  ${fails} 个缺失项 ❌  建议先运行: ./setup-pc.sh install"
    fi
    echo "== 检查完成 =="
}

# ---------------------------------------------------------------------------
# install 子命令：只装缺失项（幂等）
# ---------------------------------------------------------------------------

install_uv() {
    command -v uvx >/dev/null 2>&1 && { echo "-- uvx 已安装，跳过"; return; }
    echo "-- 安装 uv（用户级，无需 sudo）"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uvx >/dev/null 2>&1 || { echo "   [错误] uv 安装后仍不可用，请重新打开终端或 source ~/.bashrc"; return 1; }
}

install_gitlfs() {
    command -v git-lfs >/dev/null 2>&1 && { echo "-- git-lfs 已安装，跳过"; return; }
    if command -v apt-get >/dev/null 2>&1; then
        echo "-- 安装 git-lfs（需要 sudo）"
        sudo apt-get update -qq && sudo apt-get install -y git-lfs
        git lfs install
    else
        echo "   [提示] 非 apt 系统，请手动安装 git-lfs: https://git-lfs.com"
    fi
}

install_nvidia_toolkit() {
    # 先实测透传是否正常，正常则不动系统
    local DK; DK="$(docker_cmd)"
    if timeout 120 ${DK} run --rm --gpus all ${GPU_TEST_IMAGE} nvidia-smi >/dev/null 2>&1; then
        echo "-- GPU 容器透传正常，跳过 nvidia-container-toolkit 安装"
        return
    fi
    echo "-- 安装 nvidia-container-toolkit（需要 sudo）"
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "   [提示] 非 apt 系统，请参考 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit 手动安装"
        return
    fi
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    # 复测
    if timeout 120 ${DK} run --rm --gpus all ${GPU_TEST_IMAGE} nvidia-smi >/dev/null 2>&1; then
        echo "   ✅ 安装后透传测试通过"
    else
        echo "   [警告] 安装后仍不通过，请检查: docker info | grep -i runtime"
    fi
}

do_install() {
    echo "== 安装缺失的前置依赖（已装则跳过）=="
    command -v docker >/dev/null 2>&1 \
        || { echo "   [错误] Docker 未安装。先运行: curl -fsSL https://get.docker.com | sudo sh"; exit 1; }
    install_uv
    install_gitlfs
    install_nvidia_toolkit
    echo ""
    echo "== 安装完成，复核检查项 =="
    RESULTS=()
    do_check
}

# ---------------------------------------------------------------------------
# build 子命令
# ---------------------------------------------------------------------------

do_build() {
    echo "== 构建 ${IMAGE}（约 2-4 小时，约 20GB 磁盘）=="
    echo "   内容: CUDA 12.6 基础镜像 + LLVM 20 + Rust + Bootlin aarch64 工具链"
    echo "        + TVM（双架构 sm_61;sm_87）+ MLC-LLM v0.20.0 源码编译"
    echo "   建议: nohup ./setup-pc.sh build > build.log 2>&1 & 后台跑，tail -f build.log 看进度"
    echo "   提示: 中断后重跑会利用 Docker layer 缓存，不会从头再来"
    read -r -p "   确认开始构建? [y/N] " ans
    case "${ans}" in
        y|Y)
            local DK; DK="$(docker_cmd)"
            ${DK} build --progress=plain -t "${IMAGE}" "${SCRIPT_DIR}"
            echo ""
            echo "== 构建完成，复核检查项 =="
            RESULTS=()
            do_check
            ;;
        *)
            echo "已取消"
            ;;
    esac
}

do_all() {
    do_install
    do_build
}

cmd="${1:-help}"
case "${cmd}" in
    check)   do_check ;;
    install) do_install ;;
    build)   do_build ;;
    all)     do_all ;;
    *)       grep '^#   ' "$0" | sed 's/^#   //'; exit 1 ;;
esac
