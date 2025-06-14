## Local LLM Runner: Ollama

[`ollama`](https://github.com/ollama/ollama) is a command-line tool for running open-source large language models entirely on your own machine—no API keys, no vendor lock-in, full control over models and performance.

[Run AI Models Locally: Ollama Tutorial (Step-by-Step Guide + WebUI): Here\'s a detailed description of the image:\n\n**Overall:** The image is a promotional graphic with a dark background, designed to advertise AI software. \n\n**Text:**\n* Large, white text reads "FREE AI ON YOUR DESKTOP". \n* The text is bold and prominently displayed on the left side of the image.\n\n**Imagery:**\n* On the right, there is a futuristic, robotic llama. It is silver with vibrant, rainbow-colored sections on its body.\n* The llama appears to be constructed from metallic parts and has a complex mechanical design.\n* A simple, cartoon-style pig icon with a screen for a face is present below the text. An arrow points from the pig to the robotic llama, suggesting a transformation or upgrade.\n\n**Color Palette:** The image employs a dark background with bright, contrasting colors in the text and the robotic llama. The rainbow colors on the llama and the white text stand out.\n\n**Style:** The graphic combines elements of technology, artificial intelligence, and whimsical imagery, likely targeting an audience interested in cutting-edge software.](https://youtu.be_Lb5D892-2HY)

### Basic Usage

[Download Ollama for macOS, Linux, or Windows](https://ollama.com/) and add the binary to your `PATH`. See the full [Docs ↗](https://ollama.com/docs) for installation details and troubleshooting.

```bash
# List installed and available models
ollama list

# Download/pin a specific model version
ollama pull gemma3:1b-it-qat

# Run a one-off prompt
ollama run gemma3:1b-it-qat 'Write a haiku about data visualization'

# Launch a persistent HTTP API on port 11434
ollama serve

# Interact programmatically over HTTP
curl -X POST http://localhost:11434/api/chat \
     -H 'Content-Type: application/json' \
     -d '{"model":"gemma3:1b-it-qat","prompt":"Hello, world!"}'
```

### Key Features

- **Model management**: `list`/`pull` — Install and switch among Llama 3.3, DeepSeek-R1, Gemma 3, Mistral, Phi-4, and more.
- **Local inference**: `run` — Execute prompts entirely on-device for privacy and zero latency beyond hardware limits.
- **Persistent server**: `serve` — Expose a local REST API for multi-session chats and integration into scripts or apps.
- **Version pinning**: `pull model:tag` — Pin exact model versions for reproducible demos and experiments.
- **Resource control**: `--threads` / `--context` — Tune CPU/GPU usage and maximum context window for performance and memory management.

### Real-World Use Cases

- **Quick prototyping**. Brainstorm slide decks or blog outlines offline, without worrying about API quotas: `ollama run gemma-3 'Outline a slide deck on Agile best practices'`
- **Data privacy**. Summarize sensitive documents on-device, retaining full control of your data: `cat financial_report.pdf | ollama run phi-4 'Summarize the key findings'`
- **CI/CD integration**. Validate PR descriptions or test YAML configurations in your pipeline without incurring API costs: `git diff origin/main | ollama run llama2 'Check for style and clarity issues'`
- **Local app embedding**. Power a desktop or web app via the local REST API for instant LLM features: `curl -X POST http://localhost:11434/api/chat -d '{"model":"mistral","prompt":"Translate to German"}'`

Read the full [Ollama docs ↗](https://github.com/ollama/ollama/tree/main/docs) for advanced topics like custom model hosting, GPU tuning, and integrating with your development workflows.
