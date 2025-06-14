## JavaScript tools: npx

[npx](https://docs.npmjs.com/cli/v8/commands/npx) is a command-line tool that comes with npm (Node Package Manager) and allows you to execute npm package binaries and run one-off commands without installing them globally. It's essential for modern JavaScript development and data science workflows.

For data scientists, npx is useful when:

- Running JavaScript-based data visualization tools
- Converting notebooks and documents
- Testing and formatting code
- Running development servers

Here are common npx commands:

```bash
# Run a package without installing
npx http-server .                # Start a local web server
npx prettier --write .           # Format code or docs
npx eslint .                     # Lint JavaScript
npx typescript-node script.ts    # Run TypeScript directly
npx esbuild app.js               # Bundle JavaScript
npx jsdoc .                      # Generate JavaScript docs

# Run specific versions
npx prettier@3.2 --write .        # Use prettier 3.2

# Execute remote scripts (use with caution!)
npx github:user/repo            # Run from GitHub
```

Watch this introduction to npx (6 min):

[What you can do with npx (6 min): Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image is a graphic or promotional material featuring text and a logo.\n* **Background:** The background color is a dark, muted charcoal grey/black. There is a thin, horizontal red stripe across the bottom of the image.\n* **Text:** The text "what you can do with" is in a light grey, sans-serif font. It is positioned in the upper left and central portions of the image.\n* **Logo/Symbol:** To the right of "with" is a stylized logo or symbol consisting of the letters "п" and "X" in a bold, bright red color. The "X" has a brushstroke or distressed effect, giving it a slightly chaotic look. \n* **Color Scheme:** The primary colors are red, grey, and black. This color scheme suggests energy, boldness, and possibly technological or modern themes. \n* **Purpose:** The image is likely used to advertise or promote something related to the logo "пX." It is intended to convey the potential and capabilities associated with that brand or product.](https://youtu.be_55WaAoZV_tQ)
