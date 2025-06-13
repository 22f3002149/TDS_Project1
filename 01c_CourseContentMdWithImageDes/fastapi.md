## Web Framework: FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is a modern Python web framework for building APIs with automatic interactive documentation. It's fast, easy to use, and designed for building production-ready REST APIs.

Here's a minimal FastAPI app, `app.py`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
# ]
# ///

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run this with `uv run app.py`.

1. **Handle errors by raising HTTPException**

   ```python
   from fastapi import HTTPException

   async def get_item(item_id: int):
       if not valid_item(item_id):
           raise HTTPException(
               status_code=404,
               detail=f"Item {item_id} not found"
           )
   ```

2. **Use middleware for logging**

   ```python
   from fastapi import Request
   import time

   @app.middleware("http")
   async def add_timing(request: Request, call_next):
       start = time.time()
       response = await call_next(request)
       response.headers["X-Process-Time"] = str(time.time() - start)
       return response
   ```

Tools:

- [FastAPI CLI](https://fastapi.tiangolo.com/tutorial/fastapi-cli/): Project scaffolding
- [Pydantic](https://pydantic-docs.helpmanual.io/): Data validation
- [SQLModel](https://sqlmodel.tiangolo.com/): SQL databases
- [FastAPI Users](https://fastapi-users.github.io/): Authentication

Watch this FastAPI Course for Beginners (64 min):

[FastAPI Course for Beginners (64 min): Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image features a promotional graphic for a "FastAPI Crash Course."\n* **Color Scheme:** It utilizes a dark blue background with text and a logo in shades of green and white.\n* **Logo:** A circular green logo with a white lightning bolt icon is prominently displayed.\n* **Text:**\n * "FastAPI" is written in large, bold, white text.\n * Below it, "Crash Course" is displayed in a smaller, light gray font.\n* **Top-Left Corner:** A small, abstract logo or graphic element is located in the upper left corner. It appears to be a stylized "M" or similar shape.\n* **Design:** The design is clean and minimalist, intended to be eye-catching and informative. It suggests a quick and efficient learning experience.\n\n\n\n](https://youtu.be_tLKKmouUams)
