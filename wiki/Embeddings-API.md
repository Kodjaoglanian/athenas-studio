# Embeddings API

The `/v1/embeddings` endpoint provides OpenAI-compatible text embeddings for semantic search, clustering, and retrieval-augmented generation (RAG).

## Endpoint

```
POST /v1/embeddings
```

## Request

```json
{
  "model": "all-MiniLM-L6-v2",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | default model | Embedding model ID |
| `input` | string or array | required | Text to embed (single string or array of strings) |
| `encoding_format` | string | `"float"` | Encoding format: `float` or `base64` |

## Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0091, 0.0152, ...],
      "index": 0
    }
  ],
  "model": "all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

## Batch Embeddings

Send an array of strings to embed multiple texts in one request:

```json
{
  "model": "all-MiniLM-L6-v2",
  "input": ["text one", "text two", "text three"]
}
```

Response includes one embedding per input, indexed by position.

## Usage Examples

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.embeddings.create(
    model="all-MiniLM-L6-v2",
    input="The quick brown fox"
)
print(response.data[0].embedding[:5])  # first 5 dimensions
```

### cURL

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2",
    "input": "Hello world"
  }'
```

### Semantic Search with FAISS

```python
from openai import OpenAI
import faiss
import numpy as np

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

# Build index
documents = ["doc 1 text", "doc 2 text", "doc 3 text"]
embeddings = []
for doc in documents:
    resp = client.embeddings.create(model="all-MiniLM-L6-v2", input=doc)
    embeddings.append(resp.data[0].embedding)

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Search
query = "search text"
query_emb = client.embeddings.create(model="all-MiniLM-L6-v2", input=query)
D, I = index.search(np.array([query_emb.data[0].embedding], dtype=np.float32), k=2)
print([documents[i] for i in I[0]])
```

## Supported Embedding Models

Any GGUF embedding model supported by llama.cpp can be used, including:

- `all-MiniLM-L6-v2` (384 dims)
- `all-MiniLM-L12-v2` (384 dims)
- `bge-small-en-v1.5` (384 dims)
- `bge-base-en-v1.5` (768 dims)
- `bge-large-en-v1.5` (1024 dims)
- `e5-small-v2` (384 dims)
- `e5-base-v2` (768 dims)
- `nomic-embed-text-v1.5` (768 dims)

Download embedding models the same way as chat models:

```bash
athenas models pull nomic-ai/nomic-embed-text-v1.5-GGUF
```

## Loading Embedding Models

Embedding models are loaded separately from chat models. The server detects embedding models automatically based on the model architecture.

```bash
# Start server with an embedding model
athenas serve all-MiniLM-L6-v2.Q4_K_M.gguf --port 8080

# Or load at runtime
curl http://127.0.0.1:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/embedding.gguf"}'
```

## Implementation Details

The embeddings endpoint is implemented in the `athenas-server` crate and delegates to the active backend's `/v1/embeddings` endpoint (llama.cpp server). The backend automatically routes embedding requests to the correct model.

- **Streaming**: Not supported (embeddings are returned in full)
- **Rate limiting**: Subject to the same rate limits as other endpoints
- **Metrics**: Tracked under `athenas_requests_total{path="/v1/embeddings"}`
- **Auth**: Requires API key if configured
