# Using the OpenAI SDK

crane-serve implements the OpenAI API shape, so the official `openai` Python
package (or any OpenAI-compatible client) works against it unmodified. Point
`base_url` at your running server and use any API key string; it's only
checked if `--api-key`/`--api-key-file` is configured.

## Chat

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Tool calling

The standard OpenAI agentic loop works unmodified:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

def get_weather(city):
    return {"temperature_c": 18, "conditions": "cloudy"}

messages = [{"role": "user", "content": "What is the weather in Paris?"}]

while True:
    reply = client.chat.completions.create(
        model="qwen3.8-27b", messages=messages, tools=TOOLS,
    ).choices[0].message

    if not reply.tool_calls:
        print(reply.content)
        break

    # Append the assistant turn including tool_calls. The template replays
    # it into the transcript; dropping it makes the model call again.
    messages.append(reply.model_dump(exclude_none=True))
    for call in reply.tool_calls:
        args = json.loads(call.function.arguments)   # arguments is a JSON string
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": json.dumps(get_weather(**args)),
        })
```

Reasoning models additionally expose the scratchpad on the message:

```python
reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    reasoning_effort="low",                       # low | medium | xhigh
    # or: extra_body={"chat_template_kwargs": {"enable_thinking": False}},
).choices[0].message

print(reply.content)                              # "4"
print(getattr(reply, "reasoning_content", None))  # the <think> scratchpad
```

## Text-to-Speech

See [Audio](audio.md) for setup and voice lists. The examples below assume a
server already running with the matching TTS model loaded.

### Voxtral TTS

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# German female voice
response = client.audio.speech.create(
    model="voxtral",
    voice="de_female",
    input="Hallo, wie geht es Ihnen heute?",
    extra_body={"language": "german", "response_format": "wav"},
)
response.stream_to_file("output.wav")

# English neutral voice
response = client.audio.speech.create(
    model="voxtral",
    voice="neutral_female",
    input="Hello, this is a test of the Voxtral TTS engine.",
    extra_body={"response_format": "wav"},
)
response.stream_to_file("english.wav")
```

### Qwen3-TTS

Start crane-serve with a Qwen3-TTS checkpoint (`--model-type qwen3_tts`).

**CustomVoice model (predefined speakers):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Chelsie",
    input="今天天气真好，我们去公园吧！",
    extra_body={"language": "chinese"},
)
response.stream_to_file("output.wav")

response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Ethan",
    input="Hello, this is a test of the Crane TTS engine.",
    extra_body={"language": "english", "temperature": 0.7},
)
response.stream_to_file("english.wav")
```

**Base model (voice cloning):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# voice field is ignored in voice-clone mode
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="clone",
    input="そんな何もない今日が 少しだけでもいい日になったと思えたら",
    extra_body={
        "language": "japanese",
        "reference_audio": "data/audio/kinsenka_3.wav",
        "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
    },
)
response.stream_to_file("voice_clone.wav")
```

### Without the OpenAI SDK

Any HTTP client works, since `/v1/audio/speech` is a plain POST endpoint:

```python
import requests, pathlib

r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "voxtral",
        "input": "Hello, how are you today?",
        "voice": "neutral_female",
        "response_format": "wav",
    },
)
r.raise_for_status()
pathlib.Path("speech.wav").write_bytes(r.content)
```
