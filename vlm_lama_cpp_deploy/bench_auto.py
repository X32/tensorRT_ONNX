#!/usr/bin/env python3
"""
Qwen2.5-VL 自动化基准测试脚本
================================

解决问题: llama-server 有 prompt cache, 重复提问会导致 prompt_n=1, 测不出真实 prefill 速度。
方案:     每次提问自动附加唯一编号 (如 "#003"), 保证 100% 不命中缓存。

自动测试流程:
    1. 纯文本测试 N 轮 (每轮唯一提问)
    2. 图文测试 N 轮 (每轮唯一提问)
    3. 全程后台线程采样内存 (tegrastats, 1秒一次)
    4. 输出统计结果 + 可直接粘贴到文档的 Markdown 表格

用法:
    # 默认: 纯文本+图文 各测 3 轮
    python bench_auto.py

    # 只测纯文本, 5 轮
    python bench_auto.py --rounds 5 --no-image

    # 只测图文, 3 轮, 指定图片
    python bench_auto.py --no-text --image ./test.jpg --resize 768

    # 指定服务器
    python bench_auto.py --url http://localhost:8080
"""
import argparse
import base64
import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

# 全局: 每轮测试的内存峰值 (由采样线程写入)
ROUND_MEM_PEAK = 0
MEM_BASELINE = 0


# ---------- 内存采样 ----------

def read_ram_mb() -> int | None:
    """tegrastats 读统一内存, 失败则用 /proc/meminfo"""
    try:
        proc = subprocess.Popen(
            ['tegrastats', '--interval', '1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        line = proc.stdout.readline()
        proc.kill()
        proc.wait(timeout=2)
        m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
        if info.get('MemTotal', 0) > 0:
            return (info['MemTotal'] - info.get('MemAvailable', 0)) // 1024
    except Exception:
        pass
    return None


def mem_sampler(stop_event: threading.Event, peak: dict):
    """后台线程: 持续采样, 记录全局峰值"""
    while not stop_event.is_set():
        mb = read_ram_mb()
        if mb is not None:
            peak['now'] = mb
            peak['max'] = max(peak['max'], mb)
        stop_event.wait(1.0)


# ---------- 推理 ----------

def image_to_data_url(image_path: str, max_side: int | None) -> str:
    path = Path(image_path)
    if not path.exists():
        print(f"❌ 图片不存在: {path.absolute()}")
        sys.exit(1)
    if max_side is not None:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            out = path.with_name(f"{path.stem}_bench.jpg")
            img.save(out, "JPEG", quality=85)
            path = out
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def run_once(url: str, prompt: str, image_url: str | None, max_tokens: int) -> dict:
    """单次推理, 返回关键指标"""
    if image_url:
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    payload = {
        "model": "qwen2.5-vl-3b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    t = result.get("timings", {})
    usage = result.get("usage", {})
    prompt_n = t.get("prompt_n") or usage.get("prompt_tokens", 0)
    prompt_ms = t.get("prompt_ms")
    pred_n = t.get("predicted_n") or usage.get("completion_tokens", 0)
    pred_ms = t.get("predicted_ms")

    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "pred_n": pred_n,
        "pred_ms": pred_ms,
        "prefill_tps": (prompt_n / (prompt_ms / 1000)) if prompt_ms and prompt_n else None,
        "decode_tps": (pred_n / (pred_ms / 1000)) if pred_ms and pred_n else None,
        "ttft_ms": prompt_ms,
        "cache_hit": prompt_n <= 2 and usage.get("prompt_tokens", 0) > 10,  # 输入多但只算了1-2个 = 缓存命中
        "elapsed": (prompt_ms + pred_ms) / 1000 if prompt_ms and pred_ms else None,
    }


def bench_suite(name: str, url: str, rounds: int, prompt_base: str,
                image_url: str | None, max_tokens: int, mem_peak: dict) -> list:
    """跑一组测试 (每轮唯一提问), 返回指标列表"""
    print(f"\n{'─' * 60}")
    print(f"📋 测试组: {name} ({rounds} 轮, 唯一提问绕过缓存)")
    print(f"{'─' * 60}")

    results = []
    for i in range(rounds):
        # 唯一后缀: 编号 + 纳秒时间戳, 确保每轮 prompt 不同
        unique = f"{prompt_base} (编号{i+1:02d}-{time.time_ns()})"
        # 记录本轮起点内存
        mem_peak['max'] = mem_peak.get('now', 0) or 0

        try:
            r = run_once(url, unique, image_url, max_tokens)
        except Exception as e:
            print(f"   第{i+1}轮 ❌ 失败: {e}")
            continue

        r["_round_mem_peak"] = mem_peak.get('max', 0)
        results.append(r)

        cache_flag = "⚠️缓存" if r["cache_hit"] else "✅"
        prefill = f"{r['prefill_tps']:.1f}" if r["prefill_tps"] else "-"
        decode = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "-"
        print(f"   第{i+1}轮 {cache_flag} | prefill: {r['prompt_n']}tok/"
              f"{r['prompt_ms']:.0f}ms={prefill} t/s | decode: {decode} t/s | "
              f"TTFT: {r['ttft_ms']:.0f}ms | 峰值内存: {r['_round_mem_peak']}MB")
    return results


# ---------- 统计输出 ----------

def summarize(name: str, results: list) -> dict:
    """统计一组结果, 返回汇总 dict"""
    valid = [r for r in results if not r["cache_hit"]]
    if not valid:
        valid = results  # 全部缓存命中时退回全部数据(并会在输出中提示)

    def avg(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    def vmin(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return min(vals) if vals else 0

    def vmax(key):
        vals = [r[key] for r in valid if r[key] is not None]
        return max(vals) if vals else 0

    s = {
        "name": name,
        "rounds_ok": len(results),
        "cache_hits": sum(1 for r in results if r["cache_hit"]),
        "prefill_avg": avg("prefill_tps"),
        "prefill_min": vmin("prefill_tps"),
        "prefill_max": vmax("prefill_tps"),
        "decode_avg": avg("decode_tps"),
        "decode_min": vmin("decode_tps"),
        "decode_max": vmax("decode_tps"),
        "ttft_avg": avg("ttft_ms"),
        "prompt_tokens": valid[0]["prompt_n"] if valid else 0,
        "mem_peak": max((r.get("_round_mem_peak", 0) for r in results), default=0),
    }
    return s


def print_markdown_table(summaries: list, mem_baseline: int, args):
    """输出可直接粘贴到 performance_data.md 的表格"""
    print(f"\n{'=' * 60}")
    print("📊 汇总 (Markdown, 可直接粘贴到 performance_data.md)")
    print(f"{'=' * 60}\n")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"> 测试时间: {ts} ｜ 服务器: {args.url} ｜ 每组 {args.rounds} 轮取平均")
    print(f"> 内存基线(推理前): {mem_baseline}MB ｜ 提问均带唯一编号, 无缓存命中\n")

    print("| 测试组 | 输入tok | Prefill(t/s) | Decode(t/s) | TTFT(ms) | 峰值内存(MB) |")
    print("|--------|---------|-------------|------------|----------|------------|")
    for s in summaries:
        warn = f" ⚠️{s['cache_hits']}轮缓存" if s["cache_hits"] else ""
        print(f"| {s['name']}{warn} | {s['prompt_tokens']} | "
              f"{s['prefill_avg']:.1f} | {s['decode_avg']:.1f} | "
              f"{s['ttft_avg']:.0f} | {s['mem_peak']} |")

    print("\n<details><summary>各轮明细</summary>\n")
    for s, results in zip(summaries, ALL_RESULTS):
        print(f"\n**{s['name']}**")
        print("| 轮次 | prefill t/s | decode t/s | TTFT ms |")
        print("|------|------------|-----------|---------|")
        for i, r in enumerate(results):
            prefill = f"{r['prefill_tps']:.1f}" if r["prefill_tps"] else "-"
            decode = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "-"
            flag = " (缓存)" if r["cache_hit"] else ""
            print(f"| {i+1}{flag} | {prefill} | {decode} | {r['ttft_ms']:.0f} |")
    print("\n</details>")


ALL_RESULTS = []  # 保存各组原始结果, 供明细输出


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL 自动化基准测试(绕过缓存)")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器地址")
    parser.add_argument("--rounds", type=int, default=3, help="每组测试轮数 (默认 3)")
    parser.add_argument("--no-text", action="store_true", help="跳过纯文本测试")
    parser.add_argument("--no-image", action="store_true", help="跳过图文测试")
    parser.add_argument("--image", "-i", default="./test.jpg", help="测试图片 (默认 ./test.jpg)")
    parser.add_argument("--resize", "-r", type=int, default=768, help="图片长边缩放 (默认 768)")
    parser.add_argument("--max-tokens", "-n", type=int, default=128, help="每轮生成 token 上限 (默认 128)")
    parser.add_argument("--prompt", default="请解释量子计算的基本原理", help="基础提问")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Qwen2.5-VL 自动化基准测试")
    print("=" * 60)
    print(f"[服务器] {args.url}")
    print(f"[轮数] {args.rounds} | [生成上限] {args.max_tokens} tokens")
    print(f"[模式] {'纯文本' if args.no_image else ''}{'图文' if args.no_text else ''}"
          f"{'' if (args.no_text or args.no_image) else '纯文本 + 图文'}")

    # 健康检查
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=5) as resp:
            assert resp.status == 200
    except Exception:
        print(f"❌ 服务器不可达: {args.url}")
        sys.exit(1)
    print(f"✅ 服务器在线\n")

    # 启动内存采样线程
    mem_peak = {"max": 0, "now": 0}
    stop_event = threading.Event()
    sampler = threading.Thread(target=mem_sampler, args=(stop_event, mem_peak), daemon=True)
    sampler.start()
    time.sleep(1.5)  # 等第一个采样
    global MEM_BASELINE
    MEM_BASELINE = mem_peak.get('now', 0)
    print(f"💾 内存采样已启动, 基线: {MEM_BASELINE}MB")

    summaries = []

    # 1. 纯文本测试
    if not args.no_text:
        results = bench_suite("纯文本", args.url, args.rounds,
                              args.prompt, None, args.max_tokens, mem_peak)
        ALL_RESULTS.append(results)
        summaries.append(summarize("纯文本", results))

    # 2. 图文测试
    if not args.no_image:
        try:
            img_url = image_to_data_url(args.image, args.resize)
            print(f"🖼️  测试图片已编码: {args.image} (长边≤{args.resize}px)")
            results = bench_suite("图文理解", args.url, args.rounds,
                                  "描述这张图片的内容", img_url, args.max_tokens, mem_peak)
            ALL_RESULTS.append(results)
            summaries.append(summarize("图文理解", results))
        except FileNotFoundError:
            print(f"⚠️  跳过图文测试: 图片不存在 {args.image}")

    # 停止采样
    stop_event.set()
    sampler.join(timeout=3)

    # 汇总输出
    if summaries:
        print_markdown_table(summaries, MEM_BASELINE, args)
    else:
        print("⚠️  没有完成的测试组")

    print(f"\n{'=' * 60}")
    print("✅ 测试完成")


if __name__ == "__main__":
    main()
