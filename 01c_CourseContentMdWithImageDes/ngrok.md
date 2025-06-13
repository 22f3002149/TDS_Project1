## Tunneling: ngrok

[Ngrok](https://ngrok.com/) is a tool that creates secure tunnels to your localhost, making your local development server accessible to the internet. It's essential for testing webhooks, sharing work in progress, or debugging applications in production-like environments.

[ngrok in 60 seconds: Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image shows a computer screen displaying a software interface with code and text overlays, likely related to a tutorial or demonstration.\n* **Background:** The background is a dark, blurred display of code, suggesting a development environment. The code appears to be in a light-colored font on a dark background.\n* **Text Overlay:** Large white text reads "NGROK" centered on the right side of the image.\n* **Additional Text:** Below "NGROK", smaller white text says "In 60 seconds".\n* **Left Panel:** A vertical, dark panel on the left side lists various options or commands with light-colored text.\n* **Color Scheme:** The overall color scheme is dark with white/light-colored text, typical of coding or development environments.\n\nIn short, this appears to be a screen capture from a software tutorial or demonstration related to the "ngrok" tool, likely presenting a quick introduction or setup guide.](https://youtu.be_dfMdLGZLXSg)

Run the command `uvx ngrok http 8000` to create a tunnel to your local server on port 8000. This generates a public URL that you can share with others.

To get started, log into `ngrok.com` and [get an authtoken from the dashboard](https://dashboard.ngrok.com/get-started/your-authtoken). Copy it. Then run:

```bash
ngrok config add-authtoken $YOUR_AUTHTOKEN
```

Now you can forward any local port to the internet. For example:

```bash
# Start a local server on port 8000
uv run -m http.server 8000

# Start HTTP tunnel
uvx ngrok http 8000
```

Here are useful things you can do with `ngrok http`:

- `ngrok http file://.` to serve local files
- `--response-header-add "Access-Control-Allow-Origin: *"` to enable CORS
- `--oauth google --oauth-client-id $CLIENT_ID --oauth-client-secret $SECRET --oauth-allow-domain example.com --oauth-allow-email user@example.org` to restrict users to @example.com and user@example.org using Google Auth
- `--ua-filter-deny ".*bot$"` to reject user agents ending with `bot`
