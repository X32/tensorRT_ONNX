#!/usr/bin/env bash
# ============================================================================
# benchmark.sh — MLC-LLM 性能基线采集（在 Jetson 上、serve 已启动时运行）
#
# 用法: ./benchmark.sh [jetson-ip]（默认 localhost）
# 前置: ./jetson-run.sh serve 已启动；建议先 sudo nvpmodel -m 0（25W MAXN）
#
# 采集项: TTFT / 总时延 / tok/s / 峰值内存（tegrastats）
# 输出:   终端报告 + benchmark-result.txt
# ============================================================================
set -euo pipefail

HOST="${1:-localhost}"
PORT=8000
MODEL="Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
OUT="benchmark-result.txt"

PROMPT="请用中文介绍一下你自己，包括你的能力和你擅长的任务。"
MAX_TOKENS=256

echo "== MLC-LLM 性能基线 $(date '+%F %T') ==" | tee "${OUT}"
echo "目标: http://${HOST}:${PORT}  模型: ${MODEL}" | tee -a "${OUT}"
nvpmodel -q 2>/dev/null | tee -a "${OUT}" || true

# --- tegrastats 后台采样（内存/功耗） ---
if command -v tegrastats >/dev/null 2>&1; then
    sudo tegrastats --interval 1000 --logfile "$(pwd)/tegrastats.log" &
    TEGRA_PID=$!
    trap 'sudo kill ${TEGRA_PID} 2>/dev/null || true' EXIT
    echo "tegrastats 采样中（PID ${TEGRA_PID}）..."
fi

# --- 流式请求，测量 TTFT 与总时延 ---
python3 - "$HOST" "$PORT" "$MODEL" "$PROMPT" "$MAX_TOKENS" <<'EOF' | tee -a "${OUT}"
import sys, time, json, urllib.request

host, port, model, prompt, max_tokens = sys.argv[1:6]
url = f"http://{host}:{port}/v1/chat/completions"
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True,
    "max_tokens": int(max_tokens),
}).encode()
req = urllib.request.Request(url, data=payload,
                              headers={"Content-Type": "application/json"})

t0 = time.time()
ttft = None
chunks = 0
content_len = 0
with urllib.request.urlopen(req, timeout=300) as resp:
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
            content_len += len(delta)

total = time.time() - t0
print(f"\n== 结果 ==")
print(f"TTFT（首 token） : {ttft:.3f} s" if ttft else "TTFT: 未收到内容")
print(f"生成 token 数     : {chunks}")
print(f"总时延            : {total:.3f} s")
if ttft and chunks > 1:
    print(f"生成速度          : {(chunks - 1) / (total - ttft):.2f} tok/s（不含首 token）")
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
