# write a python script that runs through all .md files in the CC directory and extracts the image URLs, downloads them, and saves them in seperate directories with same name as the .md file.
import os
import re
import requests
from glob import glob
from urllib.parse import urljoin
def extract_image_urls(text):
    # Find all image URLs in the markdown content
    return re.findall(r'!\[.*?\]\((.*?)\)', text)

def download_images_from_md():
    md_files = glob('CC/*.md')
    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        image_urls = extract_image_urls(content)
        
        # Create a directory for the images based on the .md file name
        dir_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(f'CC_images/{dir_name}', exist_ok=True)
        
        for url in image_urls:
            # Handle relative URLs
            if not url.startswith('http'):
                url = urljoin('https://example.com/', url)  # Replace with actual base URL if needed
            
            # Save image with one-level-back folder from the URL path included in the filename
            url_path = url.split('?')[0]  # Remove query params if any
            parts = url_path.strip('/').split('/')
            if len(parts) > 1:
                folder_part = parts[-2]
                filename = os.path.join('CC_images', dir_name, f"{folder_part}_{os.path.basename(url_path)}")
            else:
                filename = os.path.join('CC_images', dir_name, os.path.basename(url_path))
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


download_images_from_md()

