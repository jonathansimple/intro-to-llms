from openai import OpenAI

client = OpenAI(
    base_url="http://210.61.209.139:45276/v1",
    api_key="123"
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[
        {"role": "user", "content": "Hi"},
    ],
    max_tokens=8192,
    temperature=0.7,
    extra_body={"top_k": 50},
)

print(resp.choices[0].message.content)