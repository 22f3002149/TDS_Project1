# write a python script that runs through all files in the 'DisPosts' directory and convert the content of each file to Markdown format with .md extension. The content should be converted to Markdown format with appropriate headings, lists, and links. This Markdown should be ready for chunking and embedding using openai's API. The script should also handle any special characters or formatting issues that may arise during the conversion process.
import os
import re
def convert_to_markdown(posts_dir):
    for root, _, files in os.walk(posts_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Convert content to Markdown format
                markdown_content = convert_content_to_markdown(content)

                # Save the converted content to a new .md file
                md_file_path = os.path.splitext(file_path)[0] + '.md'
                with open(md_file_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(markdown_content)

                print(f"Converted {file_path} to {md_file_path}")
def convert_content_to_markdown(content):
    # Convert headings
    content = re.sub(r'^(#{1,6})\s*(.*)', lambda m: '#' * len(m.group(1)) + ' ' + m.group(2), content, flags=re.MULTILINE)

    # Convert lists
    content = re.sub(r'^\s*-\s+(.*)', r'* \1', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+(.*)', r'1. \1', content, flags=re.MULTILINE)

    # Convert links
    content = re.sub(r'\[(.*?)\]\((.*?)\)', r'[\1](\2)', content)

    # Convert images
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', r'![\1](\2)', content)

    # Handle special characters or formatting issues if needed
    # (This is a placeholder; you can add more specific handling as required)

    return content
# Example usage
posts_directory = 'DisPosts'  # Replace with your actual posts directory
convert_to_markdown(posts_directory)
# This script converts text files in the specified posts directory to Markdown format.
# It processes each text file, converting headings, lists, links, and images to Markdown syntax.