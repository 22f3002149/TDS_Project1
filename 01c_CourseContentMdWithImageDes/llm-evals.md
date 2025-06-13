## LLM Evaluations with PromptFoo

Test-drive your prompts and models with automated, reliable evaluations.

[🚀 Test Driven Prompt Engineering with PromptFoo (12 min): "Here's a detailed description of the image:\n\n**Overall Impression:** The image is a promotional or advertising graphic with a vibrant and futuristic aesthetic, likely related to AI image generation or testing. \n\n**Visual Elements:**\n\n* **Text:** Prominently displayed text reads “FAST CHEAP ACCURATE” in a stacked format. Below this, “PROMPT TESTING” is written in larger, bolder letters.\n* **Color Scheme:** The primary colors are deep blues, golds, and neon pink/purple, creating a high-contrast and dynamic look.\n* **Background:** The background is a blurred, abstract mix of blue and golden streaks that appear like energy or digital light effects.\n* **Icons/Symbols:** Two circular icons are present. The icon on the left is crossed out in red, likely to symbolize a failed or less-desired option. The icon on the right is a circular, teal-colored design, possibly representing a logo or a successful/optimized option.\n* **Central Figure:** In the background, partially obscured, appears to be a person's face in profile.\n* **Style:** The image has a digital art/graphic design feel, with glowing effects and a sense of movement.\n\n**Possible Interpretation:** The image is likely promoting a service or tool for testing and optimizing AI image generation prompts. The “FAST CHEAP ACCURATE” text highlights the benefits, and the crossed-out icon suggests a comparison to less effective methods."](https://youtu.be_KhINc5XwhKs)

PromptFoo is a test-driven development framework for LLMs:

- **Developer-first**: Fast CLI with live reload & caching ([promptfoo.dev](https://promptfoo.dev))
- **Multi-provider**: Works with OpenAI, Anthropic, HuggingFace, Ollama & more ([GitHub](https://github.com/promptfoo/promptfoo))
- **Assertions**: Built‑in (`contains`, `equals`) & model‑graded (`llm-rubric`) ([docs](https://www.promptfoo.dev/docs/configuration/expected-outputs/))
- **CI/CD**: Integrate evals into pipelines for regression safety ([CI/CD guide](https://www.promptfoo.dev/docs/integrations/ci-cd/))

To run PromptFoo:

1. Install Node.js & npm ([nodejs.org](https://nodejs.org/))
2. Set up your [`OPENAI_API_KEY`](https://platform.openai.com/api-keys) environment variable
3. Configure `promptfooconfig.yaml`. Below is an example:

```yaml
prompts:
  - |
    Summarize this text: "{{text}}"
  - |
    Please write a concise summary of: "{{text}}"

providers:
  - openai:gpt-3.5-turbo
  - openai:gpt-4

tests:
  - name: summary_test
    vars:
      text: "PromptFoo is an open-source CLI and library for evaluating and testing LLMs with assertions, caching, and matrices."
    assertions:
      - contains-all:
          values:
            - "open-source"
            - "LLMs"
      - llm-rubric:
          instruction: |
            Score the summary from 1 to 5 for:
            - relevance: captures the main info?
            - clarity: wording is clear and concise?
          schema:
            type: object
            properties:
              relevance:
                type: number
                minimum: 1
                maximum: 5
              clarity:
                type: number
                minimum: 1
                maximum: 5
            required: [relevance, clarity]
            additionalProperties: false

commandLineOptions:
  cache: true
```

Now, you can run the evaluations and see the results.

```bash
# Execute all tests
npx -y promptfoo eval -c promptfooconfig.yaml

# List past evaluations
npx -y promptfoo list evals

# Launch interactive results viewer on port 8080
npx -y promptfoo view -p 8080
```

PromptFoo caches API responses by default (TTL 14 days). You can disable it with `--no-cache` or clear it.

```bash
# Disable cache for this run
echo y | promptfoo eval --no-cache -c promptfooconfig.yaml

# Clear all cache
echo y | promptfoo cache clear
```
