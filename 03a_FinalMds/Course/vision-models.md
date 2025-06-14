## Vision Models

[LLM Vision Models: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a screenshot of a computer desktop displaying a cricket match. It shows a player celebrating, likely after scoring. \n\n**Key Elements:**\n\n* **Central Figure:** A cricket player in a blue India jersey is prominently featured. He has his arms raised in celebration, holding a cricket bat. The number "55" is visible on his jersey.\n* **Background:** The background is filled with a blurred crowd and stadium lights, indicating a live cricket match in progress.\n* **Computer Desktop:** The image is a screenshot of a computer desktop. The taskbar at the bottom indicates various open applications and the time. There are also multiple file explorer windows open, showing a directory structure related to "CricketPeople" and "Strategy".\n* **Additional Windows:** There\'s a small video call window in the bottom right corner displaying a person in a suit, suggesting a virtual meeting is also in progress.\n* **Interface:** The overall screen is a combination of various interfaces, including a file explorer, a browser window, and a video call interface. \n\n**In essence, the image is a snapshot of someone working on their computer while simultaneously watching and enjoying a cricket match.**](https://youtu.be_FgT_Mk_bakQ)

You'll learn how to use LLMs to interpret images and extract useful information, covering:

- **Setting Up Vision Models**: Integrate vision capabilities with LLMs using APIs like OpenAI's Chat Completion.
- **Sending Image URLs for Analysis**: Pass URLs or base64-encoded images to LLMs for processing.
- **Reading Image Responses**: Get detailed textual descriptions of images, from scenic landscapes to specific objects like cricketers or bank statements.
- **Extracting Data from Images**: Convert extracted image data to various formats like Markdown tables or JSON arrays.
- **Handling Model Hallucinations**: Address inaccuracies in extraction results, understanding how different prompts can affect output quality.
- **Cost Management for Vision Models**: Adjust detail settings (e.g., "detail: low") to balance cost and output precision.

Here are the links used in the video:

- [Jupyter Notebook](https://colab.research.google.com/drive/1bK0b1XMrZWImtw01T1w9NGraDkiVi8mS)
- [OpenAI Chat API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision)
- [Sample images used](https://drive.google.com/drive/folders/14MFc7XmGIUDU4-vbmF9305c1SSQrM-gR)

Here is an example of how to analyze an image using the OpenAI API.

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "detail": "low",
            "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png"}
          }
        ]
      }
    ]
  }'
```

Let's break down the request:

- `curl https://api.openai.com/v1/chat/completions`: The API endpoint for text generation.
- `-H "Content-Type: application/json"`: The content type of the request.
- `-H "Authorization: Bearer $OPENAI_API_KEY"`: The API key for authentication.
- `-d`: The request body.
  - `"model": "gpt-4o-mini"`: The model to use for text generation.
  - `"messages":`: The messages to send to the model.
    - `"role": "user"`: The role of the message.
    - `"content":`: The content of the message.
      - `{"type": "text", "text": "What is in this image?"}`: The text message.
      - `{"type": "image_url"}`: The image message.
        - `"detail": "low"`: The detail level of the image. `low` uses fewer tokens at lower detail. `high` uses more tokens for higher detail.
        - `"image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png"}`: The URL of the image.

You can send images in a [base64 encoded format](base64-image.md), too. For example:

```bash
# Download image and convert to base64 in one step
IMAGE_BASE64=$(curl -s "https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png" | base64 -w 0)

# Send to OpenAI API
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d @- << EOF
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,$IMAGE_BASE64" }
        }
      ]
    }
  ]
}
EOF
```
