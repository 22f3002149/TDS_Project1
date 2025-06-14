## Local LLMs: Llamafile

You would have heard of Large Language Models (LLMs) like GPT-4, Claude, and Llama. Some of these models are available for free, but most of them are not.

An easy way to run LLMs locally is Mozilla's [Llamafile](https://github.com/Mozilla-Ocho/llamafile). It's a single executable file that works on Windows, Mac, and Linux. No installation or configuration needed - just download and run.

Watch this Llamafile Tutorial (6 min):

[Llamafile: Local LLMs Made Easy: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a promotional graphic for a software or tool called "Llamafile." It combines an illustration of a llama with computer-related imagery and text.\n\n**Visual Elements:**\n\n* **Llama:** A cartoon-style llama is the central figure. It\'s depicted in shades of gray and white, giving it a textured appearance. \n* **Computer Monitor:** To the left of the llama, there\'s an old-style computer monitor displaying code.\n* **Circuitry:** Underneath the monitor, a minimalist depiction of a circuit board is visible.\n* **Gear Icon:** Above the llama, a gear icon suggests a system, process or tool.\n* **Text:** \n * “Llamafile” is written in a red, bold font, indicating the name of the product.\n * Below “Llamafile” is the tagline “Local LLMs Made Easy” in a smaller, sans-serif font.\n\n**Color Palette:** The image primarily uses a grayscale palette, with the red “Llamafile” text providing a pop of color.\n\n**Style/Theme:** The graphic is clean and relatively simple, combining elements of technology and the animal theme (the llama). The image seems to be promoting a tool or software that makes it easier to work with "Local LLMs" (Large Language Models).](https://youtu.be_d1Fnfvat6nM)

Here's how to get started

1. [Download `Llama-3.2-1B-Instruct.Q6_K.llamafile` (1.11 GB)](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/Llama-3.2-1B-Instruct.Q6_K.llamafile?download=true).
2. From the command prompt or terminal, run `Llama-3.2-1B-Instruct.Q6_K.llamafile`.
3. Optional: For GPU acceleration, use `Llama-3.2-1B-Instruct.Q6_K.llamafile --n-gpu-layers 35`. (Increase or decrease the number of layers based on your GPU VRAM.)

You might see a message like this:

```text
██╗     ██╗      █████╗ ███╗   ███╗ █████╗ ███████╗██╗██╗     ███████╗
██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║██║     ██╔════╝
██║     ██║     ███████║██╔████╔██║███████║█████╗  ██║██║     █████╗
██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══╝  ██║██║     ██╔══╝
███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║     ██║███████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
software: llamafile 0.8.17
model:    Llama-3.2-1B-Instruct-Q8_0.gguf
compute:  13th Gen Intel Core i9-13900HX (alderlake)
server:   http://127.0.0.1:8080/

A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
```

You can now chat with the model. Type `/exit` or press `Ctrl+C` to stop.

You can also visit `http://127.0.0.1:8080/` in your browser to chat with the model.

LlamaFile exposes an OpenAI compatible API. Here's how to use it in Python:

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={"messages": [{"role": "user", "content": "Write a haiku about coding"}]}
)
print(response.json()["choices"][0]["message"]["content"])
```

Tools:

- [OpenAI API compatibility](https://platform.openai.com/docs/api-reference/chat): Use existing OpenAI code
- [Creating your own llamafiles](https://github.com/Mozilla-Ocho/llamafile#creating-llamafiles): Control output format
