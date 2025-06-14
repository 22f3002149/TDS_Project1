# write a pyton script that checks all files in 'DisPosts' directory & replace all image links with the text saved in corresponding text files in the 'ImgData' directory. If no text file exists, remove the image link. provide a summary of each file as how many links were replaced and how many were removed. Also provide list of txt files that were not used from the 'ImgData' directory.
import os
import re
def replace_image_links(posts_dir, img_data_dir):
    summary = {
        'total_files': 0,
        'links_replaced': 0,
        'links_removed': 0,
        'unused_txt_files': [],
        'used_txt_files': []
    }

    # Get all text files in the ImgData directory
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_data_dir) if f.endswith('.txt')}

    # Iterate through all files in the posts directory
    for root, _, files in os.walk(posts_dir):
        for file in files:
            if file.endswith('.txt'):
                summary['total_files'] += 1
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                #print(len(content))

                # Find all Markdown image links (with full match)
                md_image_links = [(m.group(0), m.group(2)) for m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', content)]
                # Find all HTML <img ...> tags and extract src (with full match)
                html_image_links = [(m.group(0), m.group(2)) for m in re.finditer(r'(<img [^>]*src="([^"]+)"[^>]*>)', content)]
                image_links = md_image_links + html_image_links
                print(f"Processing file: {file_path}")
                print(f"Found {len(image_links)} image links.")

                for full_match, img_src in image_links:
                    base_name = os.path.splitext(os.path.basename(img_src))[0]
                    print(base_name)
                    if base_name in txt_files:
                        # Replace the image link with the corresponding text
                        txt_file_path = os.path.join(img_data_dir, txt_files[base_name])
                        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                            text_content = txt_file.read()
                            print(len(text_content))
                        content = content.replace(full_match, text_content)
                        summary['links_replaced'] += 1
                        summary['used_txt_files'].append(base_name)
                    else:
                        # Remove the image link if no corresponding text file exists
                        content = content.replace(full_match, '')
                        summary['links_removed'] += 1

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

    # Identify unused text files
    used_txt_files = {os.path.splitext(f)[0] for f in os.listdir(posts_dir) if f.endswith('.txt')}
    summary['unused_txt_files'] = [f for f in txt_files.values() if os.path.splitext(f)[0] not in used_txt_files]

    return summary
posts_directory = 'DisPosts'  # Change this to your posts directory
img_data_directory = 'ImgData'  # Change this to your image data directory
summary = replace_image_links(posts_directory, img_data_directory)

# Write the summary to a text file
summary_file_path = 'summary.txt'
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    summary_file.write(f"Total files processed: {summary['total_files']}\n")
    summary_file.write(f"Links replaced: {summary['links_replaced']}\n")
    summary_file.write(f"Links removed: {summary['links_removed']}\n")
    summary_file.write("Unused text files in ImgData directory:\n")
    for txt_file in summary['unused_txt_files']:
        summary_file.write(f"{txt_file}\n") 
    summary_file.write("Used text files in ImgData directory:\n\n\n")
    for txt_file in summary['used_txt_files']:
        summary_file.write(f"{txt_file}\n")
