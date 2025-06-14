## Images: Compression

Image compression is essential when deploying apps. Often, pages have dozens of images. Image analysis runs over thousands of images. The cost of storage and bandwidth can grow over time.

Here are things you should know when you're compressing images:

- **Image dimensions** are the width and height of the image in pixels. This impacts image size a lot
- **Lossless** compression (PNG, WebP) preserves exact data
- **Lossy** compression (JPEG, WebP) removes some data for smaller files
- **Vector** formats (SVG) scale without quality loss
- **WebP** is the modern standard, supporting both lossy and lossless

Here's a rule of thumb you can use as of 2025.

- Use SVG if you can (i.e. if it's vector graphics or you can convert it to one)
- Else, reduce the image to as small as you can, and save as (lossy or lossless) WebP

Common operations with Python:

```python
from pathlib import Path
from PIL import Image
import io

async def compress_image(input_path: Path, output_path: Path, quality: int = 85) -> None:
    """Compress an image while maintaining reasonable quality."""
    with Image.open(input_path) as img:
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Optimize for web
        img.save(output_path, 'WEBP', quality=quality, optimize=True)

# Batch process images
paths = Path('images').glob('*.jpg')
for p in paths:
    await compress_image(p, p.with_suffix('.webp'))
```

Command line tools include [cwebp](https://developers.google.com/speed/webp/docs/cwebp), [pngquant](https://pngquant.org/), [jpegoptim](https://github.com/tjko/jpegoptim), and [ImageMagick](https://imagemagick.org/).

```bash
# Convert to WebP
cwebp -q 85 input.png -o output.webp

# Optimize PNG
pngquant --quality=65-80 image.png

# Optimize JPEG
jpegoptim --strip-all --all-progressive --max=85 image.jpg

# Convert and resize
convert input.jpg -resize 800x600 output.jpg

# Batch convert
mogrify -format webp -quality 85 *.jpg
```

Watch this video on modern image formats and optimization (15 min):

[Modern Image Optimization (15 min): Here\'s a detailed description of the image:\n\n**Overall:** The image is a screenshot of a video call or webinar interface, showing two presenters in split-screen format along with a graphic and text on the left side.\n\n**Left Side:**\n* **Text:** "Image compression deep-dive" in a large, bold, sans-serif font.\n* **Graphic:** A colorful doodle-style illustration containing icons representing web development concepts: code snippets, a world globe, a heart, user profile icons, and more.\n* **Branding:** "web.dev LIVE" logo in a turquoise blue color at the top left corner.\n* **Background:** The background is a light grey color.\n\n**Right Side (Top):**\n* **Presenter:** A Caucasian man with short brown hair, wearing a dark blue t-shirt and a headset.\n* **Setting:** He\'s in a home office or living room with a large screen TV visible behind him and books on a shelf.\n\n**Right Side (Bottom):**\n* **Presenter:** A Caucasian man with short brown hair, wearing a dark blue t-shirt and a baseball cap with earphone.\n* **Setting:** He is indoors in a well lit room. There is a picture frame on the wall behind him.\n\n**Overall Impression:** The image suggests a presentation or webinar focused on the topic of image compression, likely targeted towards web developers. The split screen shows two people discussing this topic.](https://youtu.be_F1kYBnY6mwg)

Tools for image optimization:

- [squoosh.app](https://squoosh.app/): Browser-based compression
- [ImageOptim](https://imageoptim.com/): GUI tool for Mac
- [sharp](https://sharp.pixelplumbing.com/): Node.js image processing
- [Pillow](https://python-pillow.org/): Python imaging library
