## Embeddings: OpenAI and Local Models

Embedding models convert text into a list of numbers. These are like a map of text in numerical form. Each number represents a feature, and similar texts will have numbers close to each other. So, if the numbers are similar, the text they represent mean something similar.

This is useful because text similarity is important in many common problems:

1. **Search**. Find similar documents to a query.
2. **Classification**. Classify text into categories.
3. **Clustering**. Group similar items into clusters.
4. **Anomaly Detection**. Find an unusual piece of text.

You can run embedding models locally or using an API. Local models are better for privacy and cost. APIs are better for scale and quality.

| Feature     | Local Models               | API                       |
| ----------- | -------------------------- | ------------------------- |
| **Privacy** | High                       | Dependent on provider     |
| **Cost**    | High setup, low after that | Pay-as-you-go             |
| **Scale**   | Limited by local resources | Easily scales with demand |
| **Quality** | Varies by model            | Typically high            |

The [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard) provides comprehensive comparisons of embedding models. These models are compared on several parameters, but here are some key ones to look at:

1. **Rank**. Higher ranked models have higher quality.
2. **Memory Usage**. Lower is better (for similar ranks). It costs less and is faster to run.
3. **Embedding Dimensions**. Lower is better. This is the number of numbers in the array. Smaller dimensions are cheaper to store.
4. **Max Tokens**. Higher is better. This is the number of input tokens (words) the model can take in a _single_ input.
5. Look for higher scores in the columns for Classification, Clustering, Summarization, etc. based on your needs.

### Local Embeddings

[Guide to Local Embeddings with Sentence Transformers: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a screenshot from a video or online presentation. It features a split screen with text overlay and a person\'s face.\n\n**Left Side:**\n* A pink rectangular background dominates the left side.\n* White text reads "Text embeddings & semantic search."\n* Below that, in smaller font, is "with Lewis" alongside a smiling face emoji.\n* There\'s a partially visible horizontal bar near the top with the words "Input IDs."\n\n**Right Side:**\n* A close-up of a man\'s face is visible.\n* He has dark curly hair and a beard.\n* He is looking directly at the camera with a slight smile.\n* A slightly blurred, digital-looking background is visible behind his head.\n\n**Overall:** The image suggests the content is related to a presentation or tutorial about text embeddings and semantic search, likely presented by a person named Lewis.](https://youtu.be_OATCgQtNX2o)

Here's a minimal example using a local embedding model:

```python
# /// script
# requires-python = "==3.12"
# dependencies = [
#   "sentence-transformers",
#   "httpx",
#   "numpy",
# ]
# ///

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('BAAI/bge-base-en-v1.5')  # A small, high quality model

async def embed(text: str) -> list[float]:
    """Get embedding vector for text using local model."""
    return model.encode(text).tolist()

async def get_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts."""
    emb1 = np.array(await embed(text1))
    emb2 = np.array(await embed(text2))
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

async def main():
    print(await get_similarity("Apple", "Orange"))
    print(await get_similarity("Apple", "Lightning"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Note the `get_similarity` function. It uses a [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to calculate the similarity between two embeddings.

### OpenAI Embeddings

For comparison, here's how to use OpenAI's API with direct HTTP calls. Replace the `embed` function in the earlier script:

```python
import os
import httpx

async def embed(text: str) -> list[float]:
    """Get embedding vector for text using OpenAI's API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        return response.json()["data"][0]["embedding"]
```

**NOTE**: You need to set the [`OPENAI_API_KEY`](https://platform.openai.com/api-keys) environment variable for this to work.
