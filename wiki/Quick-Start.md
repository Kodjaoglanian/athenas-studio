# Quick Start

## 1. Start the TUI

```bash
athenas
```

The TUI provides:
- **F1** — Chat panel (interactive chat with streaming)
- **F2** — Models panel (load/unload local models)
- **F3** — Model browser (search & download from HuggingFace)
- **F4** — Settings (inference, server, HuggingFace config)
- **F5** — Server panel (start/stop API server, manage models)
- **F6** — Logs (real-time application logs)

## 2. Download a Model

Press **F3** to open the model browser, search for a model, and download it. Or use the CLI:

```bash
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf
```

## 3. Chat in the TUI

Press **F2** to select a model, then **F1** to chat. Type your message and press **Enter**.

## 4. Start the API Server

From the TUI server panel (F5), or via CLI:

```bash
athenas serve ~/.athenas/models/llama-2-7b-chat.Q4_K_M.gguf --port 8080
```

## 5. Use the API

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Or with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## 6. One-Shot Inference

```bash
athenas run model.gguf "What is the meaning of life?"
```
