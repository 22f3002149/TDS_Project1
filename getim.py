# write a python script that runs thorugh all json files in ext directory and downloads the images from the posts and saves them in a directory called images
import json
import os
import re
import requests
from glob import glob

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

def extract_image_urls(text):
    # Find all image URLs in the post content (cooked field)
    # This regex matches src="..." in img tags
    return re.findall(r'<img[^>]+src=["\'](.*?)["\']', text)

def download_images_from_ext():
    json_files = glob('ext/*.json')
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Check for post_stream/posts
        posts = data.get('post_stream', {}).get('posts', [])
        for post in posts:
            cooked = post.get('cooked', '')
            image_urls = extract_image_urls(cooked)
            for url in image_urls:
                # Handle relative URLs (Discourse may use /uploads/...)
                if url.startswith('/'):
                    url = 'https://discourse.onlinedegree.iitm.ac.in' + url
                filename = os.path.join('images', os.path.basename(url.split('?')[0]))
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(filename, 'wb') as img_file:
                            img_file.write(response.content)
                        print(f"Downloaded {url} -> {filename}")
                    else:
                        print(f"Failed to download {url}: Status {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    download_images_from_ext()
