#!/usr/bin/env python3
"""
Qwen2.5-VL llama.cpp 服务测试程序
用法:
    # 视觉测试（图片 + 提问）
    python test_vlm.py --image /path/to/image.jpg --prompt "描述这张图片的内容"

    # 纯文本测试
    python test_vlm.py --prompt "你好，请介绍一下自己"

    # 指定服务器地址
    python test_vlm.py --url http://192.168.1.100:8080 --image ./test.jpg
"""
import argparse
import base64
import json
import sys
import time
import mimetypes
from pathlib import Path

import urllib.request
import urllib.error


def image_to_data_url(image_path: str, max_side: int | None = None) -> str:
    """把本地图片转成 data URL (base64 编码)；max_side 不为 None 时先缩小长边"""
    path = Path(image_path)
    if not path.exists():
        print(f"❌ 图片不存在: {path.absolute()}")
        sys.exit(1)

    if max_side is not None:
        try:
            from PIL import Image
        except ImportError:
            print("❌ 需要 Pillow 才能缩放图片: pip install Pillow")
            sys.exit(1)

        img = Image.open(path)
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            # JPEG 质量按需调整；质量 85 对 VLM 理解足够
            out = path.with_name(f"{path.stem}_resized.jpg")
            img.save(out, "JPEG", quality=85)
            path = out
            print(f"📐 已缩小到 {img.size[0]}x{img.size[1]} (长边 {max_side}px)")
        else:
            print(f"📐 图片长边 {max(w, h)}px ≤ {max_side}px，无需缩小")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"  # 默认按 jpeg 处理

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"🖼️  图片: {path.name} ({mime_type}, {size_mb:.2f} MB)")
    return f"data:{mime_type};base64,{b64}"


def build_messages(prompt: str, image_path: str | None, max_side: int | None = None) -> list:
    """构造 OpenAI 格式的 messages"""
    if image_path:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path, max_side)}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def chat(url: str, messages: list, max_tokens: int = 256) -> dict:
    """调用 llama-server 的 /v1/chat/completions 接口"""
    payload = {
        "model": "qwen2.5-vl-3b",  # llama-server 会忽略此字段，使用已加载的模型
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 打印服务器返回的具体错误信息，方便排查
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "(无法读取错误响应体)"
        print(f"\n❌ 服务器返回 HTTP {e.code}: {e.reason}")
        print(f"📄 错误详情: {err_body}")
        if "image" in err_body.lower() or "mmproj" in err_body.lower() or "multimodal" in err_body.lower():
            print("\n💡 提示: 多模态模型需要加载 mmproj 视觉投影文件才能处理图片，")
            print("   启动 llama-server 时加上 --mmproj 参数，例如:")
            print("   ./build/bin/llama-server -m <model.gguf> --mmproj <mmproj.gguf> --port 8080")
        sys.exit(1)
    elapsed = time.time() - start

    result["_elapsed"] = elapsed
    return result


def health_check(url: str) -> bool:
    """检查服务器是否在线"""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL llama.cpp 服务测试")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器地址 (默认: http://localhost:8080)")
    parser.add_argument("--image", "-i", default=None, help="本地图片路径")
    parser.add_argument("--resize", "-r", type=int, default=None,
                        help="发送前把图片长边缩到此像素数（如 768），减少图片 token 数")
    parser.add_argument("--prompt", "-p", default="描述这张图片的内容", help="提问内容")
    parser.add_argument("--max-tokens", "-n", type=int, default=256, help="最大生成 token 数 (默认: 256)")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Qwen2.5-VL llama.cpp 服务测试")
    print("=" * 60)

    # 1. 健康检查
    if not health_check(args.url):
        print(f"❌ 服务器不可达: {args.url}")
        print("   请确认 llama-server 已启动，例如:")
        print("   ./build/bin/llama-server -m <model.gguf> --host 0.0.0.0 --port 8080 --ctx-size 2048")
        sys.exit(1)
    print(f"✅ 服务器在线: {args.url}\n")

    # 2. 构造请求
    mode = "视觉理解" if args.image else "纯文本"
    print(f"[模式] {mode}")
    print(f"[提问] {args.prompt}")
    print("-" * 60)

    messages = build_messages(args.prompt, args.image, args.resize)
    result = chat(args.url, messages, args.max_tokens)

    # 3. 输出结果
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    elapsed = result["_elapsed"]
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0

    print(f"\n💬 回答:\n{content}\n")
    print("-" * 60)
    print(f"⏱️  总耗时: {elapsed:.2f}s")
    print(f"📊 生成 token: {completion_tokens} | 速度: {tps:.1f} tok/s")
    print(f"📊 完整 usage: {usage}")


if __name__ == "__main__":
    main()
