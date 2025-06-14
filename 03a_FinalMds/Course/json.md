## JSON

JSON (JavaScript Object Notation) is the de facto standard format for data exchange on the web and APIs. Its human-readable format and widespread support make it essential for data scientists working with web services, APIs, and configuration files.

For data scientists, JSON is essential when:

- Working with REST APIs and web services
- Storing configuration files and metadata
- Parsing semi-structured data from databases like MongoDB
- Creating data visualization specifications (e.g., Vega-Lite)

Watch this comprehensive introduction to JSON (15 min):

[JSON Crash Course: Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image is a promotional graphic likely for an educational course. It features text overlaid on a dark background, using a digital/coding aesthetic.\n* **Main Text:** Prominently displayed is the text "Crash Course" in large, white, sans-serif font.\n* **Coding/Data Element:** The word "JSON" is written in large, white, blocky font, resembling code syntax. Behind it, there\'s blurred code, indicating a JSON format with fields such as "name" and "rating".\n* **Background:** The background is a dark, muted green with lines of blurred code in lighter shades of green, creating a digital/tech-related vibe.\n* **Top Left Corner:** There\'s a small icon resembling a stylized bug or insect in the top left corner.\n* **Theme/Purpose:** The graphic suggests a course or tutorial on JSON (JavaScript Object Notation), a common data format used in web development and data exchange.](https://youtu.be_GpOO5iKzOmY)

Key concepts to understand in JSON:

- JSON only supports 6 data types: strings, numbers, booleans, null, arrays, and objects
- You can nest data. Arrays and objects can contain other data types, including other arrays and objects
- Always validate. Ensure JSON is well-formed. Comm errors: Trailing commas, missing quotes, and escape characters

[JSON Lines](https://jsonlines.org/) is a format that allows you to store multiple JSON objects in a single line.
It's useful for logging and streaming data.

Tools you could use with JSON:

- [JSONLint](https://jsonlint.com/): Validate and format JSON
- [JSON Editor Online](https://jsoneditoronline.org/): Visual JSON editor and formatter
- [JSON Schema](https://json-schema.org/): Define the structure of your JSON data
- [jq](https://stedolan.github.io/jq/): Command-line JSON processor

Common Python operations with JSON:

```python
import json

# Parse JSON string
json_str = '{"name": "Alice", "age": 30}'
data = json.loads(json_str)

# Convert to JSON string
json_str = json.dumps(data, indent=2)

# Read JSON from file
with open('data.json') as f:
    data = json.load(f)

# Write JSON to file
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read JSON data a Pandas DataFrame. JSON data is typically stored as an array of objects.
import pandas as pd
df = pd.read_json('data.json')

# Read JSON lines from file into a DataFrame. JSON lines are typically one line per object.
df = pd.read_json('data.jsonl', lines=True)
```

Practice JSON skills with these resources:

- [JSON Generator](https://json-generator.com/): Create sample JSON data
- [JSON Path Finder](https://jsonpathfinder.com/): Learn to navigate complex JSON structures
- [JSON Schema Validator](https://www.jsonschemavalidator.net/): Validate JSON against schemas
