## Documentation: Markdown

Markdown is a lightweight markup language for creating formatted text using a plain-text editor. It's the standard for documentation in software projects and data science notebooks.

Watch this introduction to Markdown (19 min):

[Markdown Crash Course (19 min): Here\'s a detailed description of the image:\n\n* **Composition:** The image features a graphic design with a dark gray background and a light gray rectangular area on the left side containing the letter "M" and an arrow pointing downwards. Adjacent to this, in the light gray section, are two lines of text.\n* **Text:** The text reads "Markdown" on the first line and "Crash Course" on the second line. Both lines are in a bold, sans-serif font.\n* **Symbol:** The "M" and the downward arrow seem to be combined as a stylized logo or symbol.\n* **Branding:** At the bottom left corner, the website address "TraversyMedia.com" is displayed in a small, light gray font.\n* **Overall Impression:** The graphic appears to be a promotional image for a "Markdown Crash Course" offered by TraversyMedia.com. Markdown is a lightweight markup language often used for formatting text on the web.](https://youtu.be_HUBNt18RFbo)

Common Markdown syntax:

````
# Heading 1
## Heading 2

**bold** and *italic*

- Bullet point
- Another point
  - Nested point

1. Numbered list
2. Second item

[Link text](https://url.com)


```python
# Code block
def hello():
    print("Hello")
```

> Blockquote
````

There is also a [GitHub Flavored Markdown](https://github.github.com/gfm/) standard which is popular. This includes extensions like:

```
- [ ] Incomplete task
- [x] Completed task

~~Strikethrough~~

Tables:

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |

```

Tools for working with Markdown:

- [markdown2](https://pypi.org/project/markdown2/): Python library to convert Markdown to HTML
- [markdownlint](https://github.com/DavidAnson/markdownlint): Linting
- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one): VS Code extension
- [pandoc](https://pandoc.org/): Convert between formats
