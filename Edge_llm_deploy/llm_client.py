from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "用一句话介绍你自己"}]
)
print(resp.choices[0].message.content)
