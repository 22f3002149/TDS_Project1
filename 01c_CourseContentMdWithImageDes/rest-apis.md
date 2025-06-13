## REST APIs

REST (Representational State Transfer) APIs are the standard way to build web services that allow different systems to communicate over HTTP. They use standard HTTP methods and JSON for data exchange.

Watch this comprehensive introduction to REST APIs (52 min):

[REST API Crash Course - Introduction + Full Python API Tutorial (52): Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a promotional thumbnail or cover for a video tutorial on REST APIs.\n\n**Visual Elements:**\n\n* **Background:** A solid, dark blue background provides a high-contrast backdrop for the text and the person in the image.\n* **Text:** Large, white, bold text prominently displays "REST APIs" followed by "in 1 Hour" in a slightly smaller font. \n* **Person:** A man with curly brown hair and a beard is visible on the left side of the image. He is looking directly at the viewer and appears to be smiling. He is wearing a dark gray jacket over a lighter gray shirt.\n* **Composition:** The text is centered and takes up a significant portion of the image, while the man\'s face is positioned in the corner. This layout draws attention to the subject matter (REST APIs) while also featuring the instructor or presenter.\n\n**Overall, the image is clean, straightforward, and informative. It effectively communicates the topic and duration of the video tutorial.**](https://youtu.be_qbLc5a9jdXo)

Key Concepts:

1. **HTTP Methods**
   - `GET`: Retrieve data
   - `POST`: Create new data
   - `PUT/PATCH`: Update existing data
   - `DELETE`: Remove data
2. **Status Codes**
   - `2xx`: Success (200 OK, 201 Created)
   - `4xx`: Client errors (400 Bad Request, 404 Not Found)
   - `5xx`: Server errors (500 Internal Server Error)

Here's a minimal REST API using FastAPI. Run this `server.py` script via `uv run server.py`:

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
# ]
# ///
from fastapi import FastAPI, HTTPException
from typing import Dict, List

app = FastAPI()

# Create a list of items that will act like a database
items: List[Dict[str, float | int | str]] = []

# Create a GET endpoint that returns all items
@app.get("/items")
async def get_items() -> List[Dict[str, float | int | str]]:
    return items

# Create a GET endpoint that returns a specific item by ID
@app.get("/items/{item_id}")
async def get_item(item_id: int) -> Dict[str, float | int | str]:
    if item := next((i for i in items if i["id"] == item_id), None):
        return item
    raise HTTPException(status_code=404, detail="Item not found")

# Create a POST endpoint that creates a new item
@app.post("/items")
async def create_item(item: Dict[str, float | str]) -> Dict[str, float | int | str]:
    new_item = {"id": len(items) + 1, "name": item["name"], "price": float(item["price"])}
    items.append(new_item)
    return new_item

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Test the API with curl:

```bash
# Get all items
curl http://localhost:8000/items

# Create an item
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -d '{"name": "Book", "price": 29.99}'

# Get specific item
curl http://localhost:8000/items/1
```

Best Practices:

1. **Use Nouns for Resources**
   - Good: `/users`, `/posts`
   - Bad: `/getUsers`, `/createPost`
2. **Version Your API**
   ```
   /api/v1/users
   /api/v2/users
   ```
3. **Handle Errors Consistently**
   ```python
   {
     "error": "Not Found",
     "message": "User 123 not found",
     "status_code": 404
   }
   ```
4. **Use Query Parameters for Filtering**
   ```
   /api/posts?status=published&category=tech
   ```
5. **Implement Pagination**
   ```
   /api/posts?page=2&limit=10
   ```

Tools:

- [Postman](https://www.postman.com/): API testing and documentation
- [Swagger/OpenAPI](https://swagger.io/): API documentation
- [HTTPie](https://httpie.io/): Modern command-line HTTP client
- [JSON Schema](https://json-schema.org/): API request/response validation

Learn more about REST APIs:

- [REST API Design Best Practices](https://stackoverflow.blog/2020/03/02/best-practices-for-rest-api-design/)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
- [Google API Design Guide](https://cloud.google.com/apis/design)
