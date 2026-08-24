#!/usr/bin/env python3
"""
Qwen2.5-VL OCR 能力测试脚本
测试不同类型的文字识别任务
"""
import sys
import os

# 自动添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from test_vlm import chat, build_messages, get_memory_usage, format_memory

def test_ocr(image_path: str, prompt: str = "请提取这张图片中的所有文字内容，保持原有格式"):
    """测试 OCR 文字识别"""
    messages = build_messages(prompt, image_path)
    result = chat("http://localhost:8080", messages, max_tokens=512)
    content = result["choices"][0]["message"]["content"]
    return content

def test_ocr_table(image_path: str):
    """测试表格 OCR"""
    prompt = "这是一个表格图片，请识别并输出为 Markdown 格式，保持表格结构和数据"
    messages = build_messages(prompt, image_path)
    result = chat("http://localhost:8080", messages, max_tokens=1024)
    content = result["choices"][0]["message"]["content"]
    return content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-VL OCR 测试")
    parser.add_argument("--image", "-i", required=True, help="图片路径")
    parser.add_argument("--type", "-t", choices=["text", "table", "document"],
                       default="text", help="OCR 类型：text=普通文字, table=表格, document=文档")
    parser.add_argument("--no-memory", action="store_true", help="不显示内存使用情况")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器地址")
    args = parser.parse_args()

    # 检查内存监控
    show_memory = not args.no_memory
    if show_memory:
        try:
            import psutil
            mem_initial = get_memory_usage()
            print(f"💾 初始内存: {format_memory(mem_initial)}")
        except ImportError:
            print("⚠️  内存监控需要 psutil: pip install psutil\n")
            show_memory = False

    print("=" * 60)
    print(f"🔍 OCR 测试: {args.image}")
    print(f"📝 类型: {args.type}")
    print("=" * 60)

    start_time = __import__('time').time()

    if args.type == "text":
        result = test_ocr(args.image)
    elif args.type == "table":
        result = test_ocr_table(args.image)
    else:
        result = test_ocr(args.image, "请识别这个文档的所有内容，包括标题、正文和格式")

    elapsed = __import__('time').time() - start_time

    print("\n📄 识别结果:")
    print(result)
    print("-" * 60)
    print(f"⏱️  总耗时: {elapsed:.2f}s")

    # 显示内存信息
    if show_memory:
        mem_final = get_memory_usage()
        mem_growth = mem_final["rss_mb"] - (mem_initial.get("rss_mb", 0) if 'mem_initial' in locals() else mem_final["rss_mb"])
        print(f"💾 内存使用: {format_memory(mem_final)} (增长 {mem_growth:.1f}MB)")

    print("=" * 60)
