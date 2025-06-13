# write python script that runs through all .md files in 'CC' directory and replace all image links with text saved in the corresponding .txt files same name of image in 'CC_images' directory, if the text file exists, otherwise remove the image link, print the number of links replaced and removed and save the modified .md files
import os
import re
def replace_image_links_in_md():
    md_files = [f for f in os.listdir('CC') if f.endswith('.md')]
    replaced_count = 0
    removed_count = 0

    for file_name in md_files:
        file_path = os.path.join('CC', file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Find all image links in the markdown content
        def replace_image_link(match):
            alt_text = match.group(1)
            image_url = match.group(2)
            image_name = os.path.basename(image_url).split('.')[0]  # Get the base name without extension
            text_file_path = os.path.join('CC_images', f"{image_name}.txt")

            if os.path.exists(text_file_path):
                with open(text_file_path, 'r', encoding='utf-8') as text_file:
                    description = text_file.read().strip()
                nonlocal replaced_count
                replaced_count += 1
                return f"{alt_text}: {description}"  # Replace with alt text and description
            else:
                nonlocal removed_count
                removed_count += 1
                return ''  # Remove the image link

        new_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_link, content)

        if new_content != content:  # Only write if changes were made
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Updated image links in {file_name}")

    print(f"Total links replaced: {replaced_count}")
    print(f"Total links removed: {removed_count}")
replace_image_links_in_md()