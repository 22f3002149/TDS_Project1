# write python script that runs through all json files in the ext directory and extract 'cooked' field from each post, also if there are images in this 'cooked' field, download them and save them in seperate direcotries with the name of json files in ext2 directory, replace image url to local saved image, save extracted 'cooked' field as a text file with the same name as the json file in ext2 directory as text content without any HTML tags except for tags for the images
import json
import os
import re
import requests
from glob import glob
from bs4 import BeautifulSoup
def extract_image_urls(text):
    # Find all image URLs in the post content (cooked field)
    return re.findall(r'<img[^>]+src=["\'](.*?)["\']', text)
def download_images_and_extract_cooked():
    json_files = glob('ext/*.json')
    os.makedirs('ext2', exist_ok=True)  # Ensure ext2 directory exists
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check for post_stream/posts
        posts = data.get('post_stream', {}).get('posts', [])
        cooked_content = []
        image_urls = []

        for post in posts:
            cooked = post.get('cooked', '')
            if cooked:
                cooked_content.append(cooked)
                image_urls.extend(extract_image_urls(cooked))

        # Create a directory for images based on the JSON file name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        image_dir = os.path.join('ext2', base_name)
        os.makedirs(image_dir, exist_ok=True)

        # Download images and replace URLs in cooked content
        for url in image_urls:
            if url.startswith('/'):
                url = 'https://discourse.onlinedegree.iitm.ac.in' + url  # Handle relative URLs
            filename = os.path.join(image_dir, os.path.basename(url.split('?')[0]))
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(filename, 'wb') as img_file:
                        img_file.write(response.content)
                    print(f"Downloaded {url} -> {filename}")
                    # Replace URL in cooked content with local path
                    for i in range(len(cooked_content)):
                        cooked_content[i] = cooked_content[i].replace(url, filename)
                else:
                    print(f"Failed to download {url}: Status {response.status_code}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")

        # Save extracted cooked content as a text file without HTML tags except for images
        text_content = '\n\n'.join(cooked_content)
        soup = BeautifulSoup(text_content, 'html.parser')
        
        # Remove all tags except <img>
        for tag in soup.find_all(True):
            if tag.name != 'img':
                tag.unwrap()

        text_file_path = os.path.join('ext2', f"{base_name}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(str(soup))
        print(f"Extracted cooked content and saved to {text_file_path}")

download_images_and_extract_cooked()
# This script processes JSON files in the 'ext' directory, extracts the 'cooked' field from each post,
# downloads images found in the 'cooked' field, saves them in a directory named after the JSON file in 'ext2',
# replaces the image URLs in the 'cooked' content with local paths, and saves the cleaned 'cooked' content as a text file
# with the same name as the JSON file in the 'ext2' directory.

