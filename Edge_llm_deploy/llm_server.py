#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容 HTTP Server for TensorRT-Edge-LLM v0.6.0 (llm_inference backend)
============================================================
在 Jetson Orin 上把已构建好的 TRT 引擎包装成 OpenAI /v1/chat/completions API，
供局域网内的应用 / OpenAI SDK 直接调用。

用法示例:
  python3 llm_server.py \
    --engine-dir    /home/zy/Desktop/workplace/engineMake_0_5b/engine_new \
    --llm-inference /home/zy/Desktop/workplace/engineMake/trt_build/examples/llm/llm_inference \
    --plugin        /home/zy/Desktop/workplace/engineMake/trt_build/libNvInfer_edgellm_plugin.so \
    --host 0.0.0.0 --port 8000

测试:
  curl http://<jetson-ip>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"你好"}]}'

OpenAI SDK 对接:
  from openai import OpenAI
  client = OpenAI(base_url="http://<jetson-ip>:8000/v1", api_key="not-needed")
  resp = client.chat.completions.create(model="qwen2.5-0.5b",
                                        messages=[{"role":"user","content":"你好"}])

注意:
  * 每次请求会启动一次 llm_inference 进程（重新加载引擎 ~1-2s），
    这是本版（路线 A）的取舍；请求是串行处理的（全局锁），
    并发请求会排队等待。
  * 引擎无流式输出能力（v0.6.0 runtime 限制），stream=True 会降级为
    一次性返回。
"""
import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn

app = FastAPI(title="TensorRT-Edge-LLM OpenAI Server")

# ---------------- 全局配置（由 main 设置） ----------------
CFG = {
    "engine_dir": None,
    "llm_inference": None,
    "plugin": None,
    "default_max_tokens": 128,
}
_lock = asyncio.Lock()  # llm_inference 进程串行化


# ---------------- OpenAI 请求模型 ----------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict], None] = None


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-0.5b"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    stream: bool = False

    class Config:
        populate_by_name = True


@app.get("/v1/models")
async def list_models():
    """让 OpenAI SDK / 客户端能发现模型。"""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-0.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tensorrt-edgellm",
            }
        ],
    }


def _normalize_messages(messages: List[dict]) -> List[dict]:
    """OpenAI messages -> llm_inference 需要的 messages 格式。

    llm_inference 支持 role: system/user/assistant，content 为字符串即可。
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            # 多模态内容（image/text 数组）: 只取文本部分
            texts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = "\n".join(t for t in texts if t) or ""
        out.append({"role": role, "content": content or ""})
    return out


def _run_llm_inference(input_json: dict, timeout: int = 600) -> dict:
    """调用一次 llm_inference，返回解析后的 output.json。"""
    env = os.environ.copy()
    if CFG["plugin"]:
        env["EDGELLM_PLUGIN_PATH"] = CFG["plugin"]

    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "input.json")
        output_file = os.path.join(tmp, "output.json")
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_json, f, ensure_ascii=False)

        cmd = [
            CFG["llm_inference"],
            "--engineDir", CFG["engine_dir"],
            "--inputFile", input_file,
            "--outputFile", output_file,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="推理超时")

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"llm_inference 失败 (rc={proc.returncode}): "
                       f"{proc.stderr[-2000:]}",
            )
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)


def _to_openai_response(text: str, model: str, req_id: str) -> dict:
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # 组装 llm_inference 输入
    input_json = {
        "batch_size": 1,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k if req.top_k is not None else 50,
        "max_generate_length": req.max_tokens or CFG["default_max_tokens"],
        "requests": [
            {"messages": _normalize_messages([m.dict() for m in req.messages])}
        ],
    }

    async with _lock:  # 串行化，防止并发启动多个引擎进程
        try:
            result = await asyncio.to_thread(_run_llm_inference, input_json)
        except HTTPException:
            raise

    responses = result.get("responses", [])
    if not responses:
        raise HTTPException(status_code=502, detail="推理返回为空")
    text = responses[0].get("output_text", "")

    if req.stream:
        # v0.6.0 无流式能力，降级为一次性返回（客户端仍能收到完整内容）
        return JSONResponse(_to_openai_response(text, req.model, req_id))
    return _to_openai_response(text, req.model, req_id)


def parse_args():
    p = argparse.ArgumentParser(description="TensorRT-Edge-LLM OpenAI 兼容服务")
    p.add_argument("--engine-dir", required=True, help="engine_new 目录（含 llm.engine/config.json/tokenizer.json）")
    p.add_argument("--llm-inference", required=True, help="llm_inference 可执行文件路径")
    p.add_argument("--plugin", default=None, help="libNvInfer_edgellm_plugin.so 路径（设置 EDGELLM_PLUGIN_PATH）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--default-max-tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    CFG["engine_dir"] = args.engine_dir
    CFG["llm_inference"] = args.llm_inference
    CFG["plugin"] = args.plugin
    CFG["default_max_tokens"] = args.default_max_tokens

    if not os.path.isdir(args.engine_dir):
        raise SystemExit(f"engine-dir 不存在: {args.engine_dir}")
    if not os.path.isfile(args.llm_inference):
        raise SystemExit(f"llm_inference 不存在: {args.llm_inference}")
    if args.plugin and not os.path.isfile(args.plugin):
        raise SystemExit(f"plugin 不存在: {args.plugin}")

    print(f"引擎目录    : {args.engine_dir}")
    print(f"llm_inference: {args.llm_inference}")
    print(f"服务地址    : http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
