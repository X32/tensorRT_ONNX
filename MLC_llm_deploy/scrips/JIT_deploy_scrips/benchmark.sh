#!/usr/bin/env bash
# ============================================================================
# benchmark.sh — MLC-LLM 标准化性能测量（TTFT / 吞吐 / 峰值内存）
#
# 用法: ./benchmark.sh [jetson-ip]（默认 localhost）
# 前置: serve 已启动（jetson-run.sh serve）；建议 sudo nvpmodel -m 0（MAXN）
#
# 环境变量:
#   ROUNDS=5     # 正式测量轮数（默认 5，报 P50 中位数 + P90）
#   WARMUP=2     # 预热轮数（默认 2，结果丢弃——剔除冷启动/CUDA graph 捕获/缓存未热）
#   INTERVAL=2   # 轮间隔秒（默认 2，防热节流累积）
#   PROMPT=...   # 覆盖测试 prompt（默认固定 32-token 问句；对比测试必须用同一 prompt）
#
# 方法论（四条硬标准）:
#   ① 引擎常驻: 前置要求 serve 已启动（模型只加载一次），测的是 HTTP 推理延迟，
#      不含模型加载/引擎初始化——每次运行重新拉起引擎测到的 ~1-2s 是冷启动，不是 TTFT
#   ② 先 warmup 再测: WARMUP 轮丢弃（CUDA context / 显存分配 / 缓存预热）
#   ③ 固定 prompt: TTFT≈prefill 时间，与输入长度强相关；跨模型/跨配置对比必须
#      同一 prompt（prompt_tokens 由 API 实测并写入报告，杜绝"空 prompt"数据）
#   ④ 多轮取 P50 + 报 P90: 单次抖动大，中位数为主指标，P90 看尾部
#
# 测量口径（与 2026-08-15 基线一致的继承项）:
#   - TTFT = 请求发出 → 首个含 content 的 SSE delta（流式）
#   - 吞吐 = (chunks-1)/(total-ttft)，不含首 token（chunks≈生成 token 数）
#   - 零依赖: bash + python3 标准库
# 注意: 新增 warmup 与多轮统计后数值会略优于旧单发口径；
#       A/B 对比时两组都用本脚本重测，勿与旧文档数值直接混比。
#
# 输出: 终端报告 + benchmark-result.txt（追加）
# ============================================================================
set -euo pipefail

HOST="${1:-localhost}"
PORT=8000
ROUNDS="${ROUNDS:-5}"
WARMUP="${WARMUP:-2}"
INTERVAL="${INTERVAL:-2}"
PROMPT="${PROMPT:-请用中文介绍一下你自己，包括你的能力和你擅长的任务。}"

# 从 /v1/models 自动探测实际服务的模型(避免标签与真实模型不符——
# 单模型 serve 对错误的 model 名也会照常响应,拼出来的名字不可信)
MODEL="$(curl -s -m 5 "http://${HOST}:${PORT}/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null)" \
  || MODEL=""
if [ -z "${MODEL}" ]; then
    echo "错误: 无法从 http://${HOST}:${PORT}/v1/models 获取模型,serve 是否已启动?" >&2
    exit 1
fi
OUT="benchmark-result.txt"

MAX_TOKENS=256

# 探测投机解码状态（A/B 时区分引擎配置；拿不到不阻断）
SPEC_LABEL="$(sudo docker logs mlc-serve 2>&1 | grep -ioE "small_draft|eagle|speculative[ _-]mode[^ ]*" | sort -u | paste -sd, - || true)"
[ -n "${SPEC_LABEL}" ] || SPEC_LABEL="disable/unknown"

log() { echo "$@" | tee -a "${OUT}"; }

log "== MLC-LLM 标准化性能测量 $(date '+%F %T') =="
log "目标: http://${HOST}:${PORT}  模型: ${MODEL}"
log "前提: 引擎常驻（serve 已启动，模型仅加载一次；测 HTTP 推理延迟，不含引擎初始化）"
log "投机解码: ${SPEC_LABEL}   轮数: ${WARMUP} 预热 + ${ROUNDS} 测量（间隔 ${INTERVAL}s）"
nvpmodel -q 2>/dev/null | tee -a "${OUT}" || true

# --- tegrastats 后台采样（内存/功耗） ---
if command -v tegrastats >/dev/null 2>&1; then
    sudo tegrastats --interval 1000 --logfile "$(pwd)/tegrastats.log" &
    TEGRA_PID=$!
    trap 'sudo kill ${TEGRA_PID} 2>/dev/null || true' EXIT
    echo "tegrastats 采样中（PID ${TEGRA_PID}）..."
fi

# --- 测量主循环：非流式拿 prompt_tokens → 预热 → 多轮流式 → 统计 ---
python3 - "$HOST" "$PORT" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$ROUNDS" "$WARMUP" "$INTERVAL" <<'EOF' | tee -a "${OUT}"
import sys, time, json, statistics, urllib.request

host, port, model, prompt = sys.argv[1:5]
max_tokens, rounds, warmup, interval = (int(x) for x in sys.argv[5:9])
url = f"http://{host}:{port}/v1/chat/completions"

def make_req(stream):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
    }).encode()
    return urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})

# ① 非流式请求拿精确 prompt_tokens（顺带验证服务可用）
with urllib.request.urlopen(make_req(False), timeout=300) as resp:
    usage = json.load(resp).get("usage", {})
prompt_tokens = usage.get("prompt_tokens", -1)
print(f"\nprompt_tokens（API 实测）: {prompt_tokens}")

def one_round():
    """一次流式请求: 返回 (ttft, total, chunks)；收不到内容抛异常"""
    t0 = time.time()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(make_req(True), timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
    total = time.time() - t0
    if ttft is None or chunks == 0:
        raise RuntimeError("本轮未收到任何内容，结果无效")
    return ttft, total, chunks

# ② 预热轮（结果丢弃——剔除引擎冷启动 / CUDA graph 捕获 / 缓存未热）
for i in range(1, warmup + 1):
    _, _, c = one_round()
    print(f"预热 {i}/{warmup} 完成（{c} tokens，不计入统计）")
    if i < warmup + rounds:
        time.sleep(interval)

# ③ 正式测量轮
ttfts, totals, chunkss, tokss = [], [], [], []
for i in range(1, rounds + 1):
    ttft, total, chunks = one_round()
    toks = (chunks - 1) / (total - ttft) if chunks > 1 else 0.0
    ttfts.append(ttft); totals.append(total)
    chunkss.append(chunks); tokss.append(toks)
    print(f"第 {i}/{rounds} 轮: TTFT {ttft:.3f}s  生成 {chunks} tok  "
          f"总时延 {total:.3f}s  {toks:.2f} tok/s")
    if i < rounds:
        time.sleep(interval)

# ④ 统计（P50 中位数为主指标，附 P90 看尾部）
def pctl(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]
def med(xs): return statistics.median(xs)
print(f"\n== 汇总（{rounds} 轮）==")
print(f"TTFT   P50: {med(ttfts):.3f} s   P90: {pctl(ttfts, 90):.3f} s   "
      f"(min {min(ttfts):.3f} / max {max(ttfts):.3f})")
print(f"吞吐   P50: {med(tokss):.2f} tok/s  P90(慢尾): {pctl(tokss, 10):.2f} tok/s  "
      f"(min {min(tokss):.2f} / max {max(tokss):.2f})")
print(f"生成 token 数/轮: 中位数 {med(chunkss):.0f}（max_tokens={max_tokens}）")
EOF

# --- tegrastats 汇总 ---
if [ -f "$(pwd)/tegrastats.log" ] && command -v tegrastats >/dev/null 2>&1; then
    sudo kill ${TEGRA_PID} 2>/dev/null || true
    sleep 1
    echo "== 内存/功耗摘要 ==" | tee -a "${OUT}"
    grep -oP 'RAM [0-9]+/[0-9]+MiB' "$(pwd)/tegrastats.log" \
        | awk '{print $2}' | cut -d/ -f1 | sort -n | tail -1 \
        | xargs -I{} echo "峰值已用内存: {} MiB" | tee -a "${OUT}"
    grep -oP 'POM_5V_GPU_PWR [0-9]+' "$(pwd)/tegrastats.log" \
        | awk '{print $2}' | sort -n | tail -1 \
        | xargs -I{} echo "峰值 GPU 功耗: {} mW" | tee -a "${OUT}" || true
fi

echo "完整报告已存: ${OUT}"
