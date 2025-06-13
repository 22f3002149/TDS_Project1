## LLM Sentiment Analysis

[OpenAI's API](https://platform.openai.com/) provides access to language models like GPT 4o, GPT 4o mini, etc.

For more details, read OpenAI's guide for:

- [Text Generation](https://platform.openai.com/docs/guides/text-generation)
- [Vision](https://platform.openai.com/docs/guides/vision)
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

Start with this quick tutorial:

[OpenAI API Quickstart: Send your First API Request: "Here's a detailed description of the image:\n\n**Overall Impression:**\n\nThe image displays a terminal window (likely on a macOS system) showcasing a `.zshrc` configuration file. This file is used to customize the Z shell (a popular command-line interpreter).\n\n**Detailed Breakdown:**\n\n* **Terminal Window:** The main focus is a dark-themed terminal window with white text.\n* **.zshrc File:** The content displayed is a snippet of the `.zshrc` file. It appears to be a series of `export` commands, which are used to set environment variables.\n* **Environment Variables:** Several environment variables are defined, including:\n * `PATH`: This variable specifies the directories the system searches for executable files. It includes paths related to Ruby, Android SDK, and the user's local bin directory.\n * `LANGUAGE` and `LC_ALL`: These variables are set to `en_US.UTF-8` for language and locale settings.\n * `ANDROID_HOME`: This variable points to the Android SDK directory.\n * `OPENAI_API_KEY`: A critical line revealing the configuration of an OpenAI API key.\n* **Comments:** There are a few comments denoted by the `#` symbol, providing context about the file and its creation date.\n* **File Editor:** The file is opened in a text editor with a dark theme.\n* **Toolbar:** At the bottom is a toolbar with icons, suggesting the text editor is part of a larger Integrated Development Environment (IDE).\n* **Code Highlighting:** The code appears to have syntax highlighting, enhancing readability.\n\n**In essence, the image depicts a developer’s customized shell configuration file, including setup for the OpenAI API, Ruby, and the Android SDK.**"](https://youtu.be_Xz4ORA0cOwQ)

Here's a minimal example using `curl` to generate text:

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{ "role": "user", "content": "Write a haiku about programming." }]
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
    - `"content": "Write a haiku about programming."`: The content of the message.

[LLM Sentiment Analysis: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a screenshot of a video call or online meeting interface. It appears to be a presentation or discussion about a show, potentially a documentary or series related to a prison. \n\n**Key Elements:**\n\n* **Screen Share:** A large portion of the screen displays text in different colors, likely subtitles or a script being discussed. The text mentions "Oz," "Security State Penitentiary," and details about the prison’s demographics (Aryans, Muslims, gangs, Italians, Irish). The text also highlights conflict and clandestine deals occurring within the prison.\n* **Participant Windows:** Several smaller windows show participants in the video call. \n * Top Right: A man with short dark hair and a beard is visible, seemingly the speaker. \n * Bottom Right: Initialed "AG" is a user\'s name.\n * Bottom Left: Initialed "P" is a user\'s name.\n* **Interface Elements:** The screenshot includes standard video conferencing UI elements: control buttons, participant lists, and a shared screen indicator.\n* **Color Scheme:** The text on the shared screen is colorful (red, green, blue) and appears to be code-like or a highlighted transcript.\n\n**Inferences:**\n\n* The group is likely discussing the TV show "Oz" (a prison drama).\n* The presentation is about the prison\'s complex social dynamics and internal conflicts.\n* The meeting is a discussion or analysis, potentially for a book club, film studies, or a related academic/entertainment purpose.\n\n\n\n](https://youtu.be__D46QrX-2iU)

This video explains how to use large language models (LLMs) for sentiment analysis and classification, covering:

- **Sentiment Analysis**: Use OpenAI API to identify the sentiment of movie reviews as positive or negative.
- **Prompt Engineering**: Learn how to craft effective prompts to get desired results from LLMs.
- **LLM Training**: Understand how to train LLMs by providing examples and feedback.
- **OpenAI API Integration**: Integrate OpenAI API into Python code to perform sentiment analysis.
- **Tokenization**: Learn about tokenization and its impact on LLM input and cost.
- **Zero-Shot, One-Shot, and Multi-Shot Learning**: Understand different approaches to using LLMs for learning.

Here are the links used in the video:

- [Jupyter Notebook](https://colab.research.google.com/drive/1tVZBD9PKto1kPmVJFNUt0tdzT5EmLLWs)
- [Movie reviews dataset](https://drive.google.com/file/d/1X33ao8_PE17c3htkQ-1p2dmW2xKmOq8Q/view)
- [OpenAI Playground](https://platform.openai.com/playground/chat)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/)
- [OpenAI Docs](https://platform.openai.com/docs/overview)
