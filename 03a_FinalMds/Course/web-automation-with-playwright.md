## Web Scraping with Playwright in Python

Scrape JavaScript‑heavy sites effortlessly with Playwright.

[🤖 Playwright: Advanced Web Scraping in Python (14 min): Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a digital graphic promoting "Web Scraping with Playwright," with a surreal and artistic aesthetic.\n\n**Visual Elements:**\n\n* **Central Figure:** A large, straw figure resembling a scarecrow or a mythical creature is the primary focus. It\'s covered in dark, shaggy material, with a wide-brimmed hat and a green mask covering its face. The figure is wielding a rake, positioned as if it is "sweeping" a flowing, golden field.\n* **Background:** The background consists of a vast field of golden stalks, likely wheat or a similar grain. The field appears to be depicted with a sense of motion and flow, resembling waves or ripples. \n* **Text Overlay:** Large, white text reads "WEB SCRAPING WITH PLAYWRIGHT" and is positioned prominently in the upper left corner.\n* **Logos:** Two logos are displayed in the upper right corner:\n * The Python logo (a blue and yellow snake design)\n * The Playwright logo (a red and black theater mask)\n\n**Color Palette:** The image is dominated by golden yellows, browns, and greens, creating a natural and somewhat dramatic atmosphere.\n\n**Style & Composition:** The image uses a combination of realistic textures (the straw, field) with a slightly surreal and illustrative style. The composition emphasizes the idea of gathering or "scraping" data from a vast source (the field).\n\n\n\n](https://youtu.be/biFzRHk4xpY) ([youtube.com](https://www.youtube.com_watch?v=biFzRHk4xpY&utm_source=chatgpt.com))

Playwright offers:

- **JavaScript rendering**: Executes page scripts so you scrape only after content appears. ([playwright.dev](https://playwright.dev/python/docs/intro))
- **Headless & headed modes**: Run without UI or in a real browser for debugging. ([playwright.dev](https://playwright.dev/python/docs/intro))
- **Auto‑waiting & retry**: Built‑in locators reduce flakiness. ([playwright.dev](https://playwright.dev/python/docs/locators))
- **Multi‑browser support**: Chromium, Firefox, WebKit—all from one API. ([playwright.dev](https://playwright.dev/python/docs/intro))

### Example: Scraping a JS‑Rendered Site

We’ll scrape [Quotes to Scrape (JS)](https://quotes.toscrape.com/js/)—a site that loads quotes via JavaScript, so a simple `requests` call gets only an empty shell ([quotes.toscrape.com](https://quotes.toscrape.com/js/)). Playwright runs the scripts and gives us the real content:

```python
# /// script
# dependencies = ["playwright"]
# ///

from playwright.sync_api import sync_playwright

def scrape_quotes():
    with sync_playwright() as p:
        # Channel can be "chrome", "msedge", "chrome-beta", "msedge-beta" or "msedge-dev".
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page()
        page.goto("https://quotes.toscrape.com/js/")
        quotes = page.query_selector_all(".quote")
        for q in quotes:
            text = q.query_selector(".text").inner_text()
            author = q.query_selector(".author").inner_text()
            print(f"{text} — {author}")
        browser.close()

if __name__ == "__main__":
    scrape_quotes()
```

Save as `scraper.py` and run:

```bash
uv run scraper.py
```

You’ll see each quote plus author printed—fetched only after the JS executes.
