## Serverless hosting: Vercel

<!--

Why Vercel? I evaluated from https://survey.stackoverflow.co/2024/technology#2-cloud-platforms

- AWS, Azure, Google Cloud are too complex for beginners
- Cloudflare (next most popular, widely admired) Python support is in beta
- Hetzner (most admired), Supabase (next most admired) do not have a serverless platform
- Fly.io (next most admired) does not have a free tier
- Heroku (used in previous terms) is the least admired
- Vercel is both popular, admired, growing, has a free plan, and a simple API

-->

Serverless platforms let you rent a single function instead of an entire machine. They're perfect for small web tools that _don't need to run all the time_. Here are some common real-life uses:

- A contact form that emails you when someone wants to hire you (runs for 2-3 seconds, a few times per day)
- A tool that converts uploaded photos to black and white (runs for 5-10 seconds when someone uploads a photo)
- A chatbot that answers basic questions about your business hours (runs for 1-2 seconds per question)
- A newsletter sign-up that adds emails to your mailing list (runs for 1 second per sign-up)
- A webhook that posts your Etsy sales to Discord (runs for 1 second whenever you make a sale)

You only pay when someone uses your tool, and the platform automatically handles busy periods. For example, if 100 people fill out your contact form at once, the platform creates 100 temporary copies of your code to handle them all. When they're done, these copies disappear. It's cheaper than running a full-time server because you're not paying for the time when no one is using your tool - most tools are idle 95% of the time!

Rather than writing a full program, serverless platforms let you write functions. These functions are called via HTTP requests. They run in a cloud environment and are scaled up and down automatically. But this means you write programs in a different style. For example:

- You can't `pip install` packages - you have to use `requirements.txt`
- You can't read or write files from the file system - you can only use APIs.
- You can't run commands (e.g. `subprocess.run()`)

[Vercel](https://vercel.com/) is a cloud platform optimized for frontend frameworks and serverless functions. Vercel is tightly integrated with GitHub. Pushing to your repository automatically triggers new deployments.

Here's a [quickstart](https://vercel.com/docs/functions/runtimes/python). [Sign-up with Vercel](https://vercel.com/signup). Create an empty `git` repo with this `api/index.py` file.

To deploy a FastAPI app, add a `requirements.txt` file with `fastapi` as a dependency.

```text
fastapi
```

Add your FastAPI code to a file, e.g. `main.py`.

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
```

Add a `vercel.json` file to the root of your repository.

```json
{
  "builds": [{ "src": "main.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "main.py" }]
}
```

On the command line, run:

- `npx vercel` to deploy a test version
- `npx vercel --prod` to deploy to production

**Environment Variables**. Use `npx vercel env add` to add environment variables. In your code, use `os.environ.get('SECRET_KEY')` to access them.

### Videos

[Vercel Product Walkthrough: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a promotional or demonstration screenshot related to the Vercel platform, showcasing a user interface and a video presentation.\n\n**Key Elements:**\n\n1. **Video Presentation:** On the left side of the image is a video showcasing a man speaking. He has short brown hair, is wearing a dark grey sweater, and has a bright and engaged expression. \n\n2. **Vercel Interface:** The right side features a screenshot of the Vercel platform\'s user interface. \n * It includes a top navigation bar with the Vercel logo, a "Git Provider" option, "Feedback" and "Blog" links.\n * Below, there’s a section displaying “Acme - Enterprise”.\n * The main area shows tabs for "Projects," "Integrations," "Activity," "Domains," "Usage," and "Settings".\n * A search bar is visible, and below that are displayed a series of messages.\n * The messages indicate “update img src” and “updated docs on layout”.\n\n3. **Vercel Logo:** A prominent Vercel logo is situated at the bottom left of the screen, showcasing a triangular arrow pointing upwards with the "Vercel" text beside it.\n\n4. **Color Palette:** The color scheme is predominantly dark, with the background being black. The Vercel elements feature a combination of white, blue, and light grey colors.\n\n**In summary:** The image showcases the Vercel platform in use, potentially highlighting the platform’s update and change tracking functionality. The video portion likely demonstrates the use or explanation of the platform.](https://youtu.be_sPmat30SE4k)

[Deploy FastAPI on Vercel | Quick and Easy Tutorial: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a promotional or thumbnail image, likely for a tech-related video or course. It features a person and graphic elements.\n\n**Visual Elements:**\n\n* **Person:** A man with short dark hair and a beard is smiling at the camera. He is wearing a grey t-shirt. He is pointing with his right hand towards the right side of the frame.\n* **Text:** The words "DEPLOY FAST" are displayed in white, bold capital letters in the top-left corner of the image.\n* **Graphic:** A large black triangle dominates the left side of the image. Inside the triangle, there is a teal-colored shape that resembles a stylized "F."\n* **Background:** The background is a dark teal color with a pattern of repeating "0" and "1" digits (binary code) overlaid on it. This suggests a tech or programming context.\n\n**Composition:** The man is positioned on the right side, while the triangle and text are on the left, creating a visually balanced composition. \n\n**Overall Tone:** The image conveys a sense of energy and enthusiasm. The smiling man and the phrase "DEPLOY FAST" suggest a focus on speed and efficiency in a technological context.](https://youtu.be_8R-cetf_sZ4)
