#!/usr/bin/env python3
"""
Qwen2.5-VL OCR 能力测试脚本
测试不同类型的文字识别任务
"""
import sys
sys.path.insert(0, '/home/zy/Desktop/workplace/tensorRT/vlm_lama_cpp')

from test_vlm import chat, build_messages

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
    args = parser.parse_args()

    print("=" * 60)
    print(f"🔍 OCR 测试: {args.image}")
    print(f"📝 类型: {args.type}")
    print("=" * 60)

    if args.type == "text":
        result = test_ocr(args.image)
    elif args.type == "table":
        result = test_ocr_table(args.image)
    else:
        result = test_ocr(args.image, "请识别这个文档的所有内容，包括标题、正文和格式")

    print("\n📄 识别结果:")
    print(result)
    print("=" * 60)
