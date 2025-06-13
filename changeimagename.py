# write a python script that iterates through all .md files in the CC directory and renames all the image links by replacing last '/' with '_'.
import os
import re
def rename_image_links_in_md():
    md_files = [f for f in os.listdir('CC') if f.endswith('.md')]
    
    for file_name in md_files:
        file_path = os.path.join('CC', file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace the last '/' before the image name in the URL with '_'
        def replace_last_slash(match):
            url = match.group(2)
            if '/' in url:
                url = url.rsplit('/', 1)
                new_url = url[0] + '_' + url[1]
            else:
                new_url = url
            return f"![{match.group(1)}]({new_url})"

        new_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_last_slash, content)
        
        if new_content != content:  # Only write if changes were made
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Updated image links in {file_name}")

rename_image_links_in_md()
