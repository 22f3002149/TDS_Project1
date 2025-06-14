## CORS: Cross-Origin Resource Sharing

CORS (Cross-Origin Resource Sharing) is a security mechanism that controls how web browsers handle requests between different origins (domains, protocols, or ports). Data scientists need CORS for APIs serving data or analysis to a browser on a different domain.

Watch this practical explanation of CORS (3 min):

[CORS in 100 Seconds: Here is a detailed description of the image:\n\n* **Overall Layout**: The image is a graphic with a light gray background and red stripes at the top and bottom.\n* **Text**: The primary text is "CORS" in a stylized, flowing script font, colored red with a white outline. Below it are the words "CROSS-ORIGIN RESOURCE SHARING" in a bold, sans-serif font.\n* **Header**: At the very top of the image, in a smaller, sans-serif font, is the text "100 SECONDS OF". \n* **Color Scheme**: The color scheme consists of red, white, and gray.\n* **Purpose**: The graphic seems to be promoting or explaining the concept of Cross-Origin Resource Sharing (CORS). It likely comes from an educational or tutorial video.](https://youtu.be_4KHiSt0oLJ0)

Key CORS concepts:

- **Same-Origin Policy**: Browsers block requests between different origins by default
- **CORS Headers**: Server responses must include specific headers to allow cross-origin requests
- **Preflight Requests**: Browsers send OPTIONS requests to check if the actual request is allowed
- **Credentials**: Special handling required for requests with cookies or authentication

If you're exposing your API with a GET request publicly, the only thing you need to do is set the HTTP header `Access-Control-Allow-Origin: *`.

Here are other common CORS headers:

```http
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
```

To implement CORS in FastAPI, use the [`CORSMiddleware` middleware](https://fastapi.tiangolo.com/tutorial/cors/):

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)
```

Testing CORS with JavaScript:

```javascript
// Simple request
const response = await fetch("https://api.example.com/data", {
  method: "GET",
  headers: { "Content-Type": "application/json" },
});

// Request with credentials
const response = await fetch("https://api.example.com/data", {
  credentials: "include",
  headers: { "Content-Type": "application/json" },
});
```

Useful CORS debugging tools:

- [CORS Checker](https://cors-test.codehappy.dev/): Test CORS configurations
- Browser DevTools Network tab: Inspect CORS headers and preflight requests
- [cors-anywhere](https://github.com/Rob--W/cors-anywhere): CORS proxy for development

Common CORS errors and solutions:

- `No 'Access-Control-Allow-Origin' header`: Configure server to send proper CORS headers
- `Request header field not allowed`: Add required headers to `Access-Control-Allow-Headers`
- `Credentials flag`: Set both `credentials: 'include'` and `Access-Control-Allow-Credentials: true`
- `Wild card error`: Cannot use `*` with credentials; specify exact origins
