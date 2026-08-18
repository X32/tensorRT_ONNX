#!/usr/bin/env python3
"""
TensorRT-Edge-LLM 增强客户端测试脚本

支持多轮对话、会话管理、错误处理、性能统计等功能
"""

import argparse
import json
import time
from typing import List, Dict, Any
from openai import OpenAI


class LLMClient:
    """TensorRT-Edge-LLM 客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", api_key: str = "not-needed"):
        """初始化客户端

        Args:
            base_url: 服务基础 URL
            api_key: API 密钥 (默认不需要)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self.model = "qwen2.5-0.5b"  # 默认模型

    def check_service(self) -> bool:
        """检查服务可用性"""
        try:
            models = self.client.models.list()
            if models.data:
                self.model = models.data[0].id
                print(f"✓ 服务可用，模型: {self.model}")
                return True
        except Exception as e:
            print(f"✗ 服务不可用: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """对话补全

        Args:
            messages: 对话消息列表
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top-p 参数
            stream: 是否流式输出

        Returns:
            响应结果字典
        """
        start_time = time.time()

        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True
                )

                # 收集流式响应
                content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="", flush=True)

                print()  # 换行

                end_time = time.time()

                return {
                    "content": content,
                    "model": self.model,
                    "time": end_time - start_time,
                    "tokens": len(content) // 2,  # 粗略估计
                    "stream": True
                }
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )

                end_time = time.time()

                content = response.choices[0].message.content

                return {
                    "content": content,
                    "model": response.model,
                    "time": end_time - start_time,
                    "tokens": response.usage.total_tokens if hasattr(response, 'usage') else len(content) // 2,
                    "stream": False
                }

        except Exception as e:
            return {
                "error": str(e),
                "time": time.time() - start_time
            }

    def simple_chat(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """简单对话

        Args:
            prompt: 用户输入
            **kwargs: 其他参数传递给 chat_completion

        Returns:
            响应结果
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, **kwargs)


class InteractiveSession:
    """交互式会话"""

    def __init__(self, client: LLMClient):
        """初始化会话

        Args:
            client: LLM 客户端
        """
        self.client = client
        self.messages = []
        self.system_prompt = "你是一个有用的AI助手。"

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def chat(self, user_input: str) -> str:
        """进行对话

        Args:
            user_input: 用户输入

        Returns:
            AI 响应
        """
        self.messages.append({"role": "user", "content": user_input})

        result = self.client.chat_completion(self.messages)

        if "error" in result:
            return f"错误: {result['error']}"

        assistant_message = result["content"]
        self.messages.append({"role": "assistant", "content": assistant_message})

        # 显示性能统计
        print(f"\n[时延: {result['time']:.2f}s, Token: {result['tokens']}]")

        return assistant_message

    def reset(self):
        """重置会话"""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def run(self):
        """运行交互式会话"""
        print("=== TensorRT-Edge-LLM 交互式对话 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'reset' 重置会话")
        print("输入 'history' 查看对话历史")
        print("输入 'system <prompt>' 设置系统提示词")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break

                if user_input.lower() == 'reset':
                    self.reset()
                    print("✓ 会话已重置")
                    continue

                if user_input.lower() == 'history':
                    print("\n对话历史:")
                    for i, msg in enumerate(self.messages):
                        role = msg['role']
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"{i+1}. [{role}]: {content}")
                    continue

                if user_input.lower().startswith('system '):
                    system_prompt = user_input[8:].strip()
                    self.set_system_prompt(system_prompt)
                    print(f"✓ 系统提示词已设置: {system_prompt[:50]}...")
                    continue

                # 进行对话
                print("AI: ", end="", flush=True)
                response = self.chat(user_input)
                print()

            except KeyboardInterrupt:
                print("\n\n会话中断")
                break
            except EOFError:
                print("\n\n会话结束")
                break


def run_basic_test(client: LLMClient):
    """运行基础测试"""
    print("=== 基础功能测试 ===")

    test_cases = [
        {"name": "简单问候", "prompt": "你好"},
        {"name": "自我介绍", "prompt": "请用一句话介绍你自己"},
        {"name": "知识问答", "prompt": "什么是TensorRT?"},
        {"name": "代码生成", "prompt": "用Python写一个计算斐波那契数列的函数"},
    ]

    results = []

    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        print(f"问题: {test['prompt']}")

        result = client.simple_chat(test['prompt'], max_tokens=128)

        if "error" in result:
            print(f"✗ 失败: {result['error']}")
            results.append({"name": test['name'], "success": False})
        else:
            print(f"回答: {result['content'][:100]}...")
            print(f"[时延: {result['time']:.2f}s]")
            results.append({
                "name": test['name'],
                "success": True,
                "time": result['time'],
                "tokens": result['tokens']
            })

    # 总结
    print("\n=== 测试总结 ===")
    success_count = sum(1 for r in results if r['success'])
    print(f"成功: {success_count}/{len(results)}")

    if success_count > 0:
        avg_time = sum(r['time'] for r in results if r['success']) / success_count
        print(f"平均时延: {avg_time:.2f}s")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TensorRT-Edge-LLM 客户端测试工具")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1", help="服务 URL")
    parser.add_argument("--mode", choices=["interactive", "test", "single"], default="interactive", help="运行模式")
    parser.add_argument("--prompt", help="单次模式下的提示词")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p 参数")
    parser.add_argument("--stream", action="store_true", help="流式输出")

    args = parser.parse_args()

    # 创建客户端
    client = LLMClient(base_url=args.url)

    # 检查服务
    if not client.check_service():
        print("错误: 服务不可用，请检查 URL 和服务状态")
        return 1

    # 根据模式运行
    if args.mode == "interactive":
        session = InteractiveSession(client)
        session.run()

    elif args.mode == "test":
        run_basic_test(client)

    elif args.mode == "single":
        if not args.prompt:
            print("错误: 单次模式需要提供 --prompt 参数")
            return 1

        result = client.simple_chat(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=args.stream
        )

        if "error" in result:
            print(f"错误: {result['error']}")
            return 1
        else:
            print(f"回答: {result['content']}")
            print(f"\n[时延: {result['time']:.2f}s, Token: {result['tokens']}]")
            return 0

    return 0


if __name__ == "__main__":
    exit(main())