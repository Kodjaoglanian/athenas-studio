# TUI Guide

The Athenas Studio TUI provides a full-featured terminal interface for LLM inference, model management, and server control.

## Key Bindings

### Global

| Key | Action |
|-----|--------|
| **F1** | Chat panel |
| **F2** | Models panel |
| **F3** | Model browser (HuggingFace) |
| **F4** | Settings |
| **F5** | Server panel |
| **F6** | Logs |
| **Ctrl+C** | Quit |

### Chat Panel (F1)

| Key | Action |
|-----|--------|
| **Enter** | Send message |
| **Shift+Enter** | Newline |
| **Up/Down** | Scroll chat history |
| **Tab** | Switch between input and chat |
| **Esc** | Cancel generation |
| `/help` | Show chat commands |
| `/clear` | Clear chat history |
| `/model` | Show current model info |

### Models Panel (F2)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate model list |
| **Enter** | Load selected model |
| **Left/Right** | Cycle models |
| **Delete** | Unload model |

### Model Browser (F3)

| Key | Action |
|-----|--------|
| **Enter** | Search / Download selected model |
| **Up/Down** | Navigate results |
| **Esc** | Clear search |

### Settings (F4)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate settings |
| **Left/Right** | Change value |
| **Enter** | Edit text field |
| **Esc** | Cancel edit |

### Server Panel (F5)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate fields |
| **Left/Right** | Change values / select models |
| **Enter** | Start server / Load additional model / Unload / Set default |
| **Esc** | Stop server |

### Logs (F6)

| Key | Action |
|-----|--------|
| **Up/Down** | Scroll logs |
| **C** | Clear logs |
| **Auto-scroll** | Follows latest log entries |

## Server Panel — Multi-Model Management

When the server is running, you can:

1. Use **Left/Right** on the **Model** field to select a different model
2. Navigate to **▶ Load Additional Model** and press **Enter** to load it alongside the existing model
3. Use **■ Unload** (Left/Right to select, Enter to unload) to remove a model from memory
4. Use **★ Default** (Left/Right to select, Enter to set) to choose which model handles requests without a `model` field
5. The **LOADED MODELS** section shows all active models with their IDs, backends, and default status (★)

## Chat Integration with Server

When the server is running with a loaded model, the TUI chat automatically uses the server's loaded model. This means:

- **No duplicate loading** — The chat reuses the server's model backend
- **Shared context** — Chat and API requests share the same model instance
- **Real-time sync** — Loading/unloading models in the server panel updates the chat state

If you also load a model locally via F2, the local model takes priority for chat. The server model is used as a fallback when no local model is loaded.

## Reasoning/Thinking Display

For models that support reasoning (Qwen3.5, DeepSeek R1, etc.), the TUI shows:

- **Reasoning content** in a collapsible section above the response
- **Toggle expand/collapse** with Enter on the reasoning section
- **Token-per-second** counter during generation
- If the model produces only reasoning and no response, a helpful message is shown
