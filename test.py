import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:53878/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Altın oran nedir, kısaca anlat."}],
    stream=True
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)