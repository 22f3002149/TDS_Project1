# Write a python script that runs through all .txt files in 'CC_images' directory & remove all formatting, HTML tags, '\n' characters and make it human readable and save & replace .txt file
import os
import re
def clean_texts(directory):
    # Iterate through all .txt files in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove HTML tags
                content = re.sub(r'<[^>]+>', '', content)
                
                # Remove newlines and extra spaces
                content = re.sub(r'\n+', ' ', content)
                content = re.sub(r'\n\n+', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Write the cleaned text back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Processed {file_path}: Cleaned text.")
directory = 'Data'
clean_texts(directory)
# Write a python script that runs through all .txt files in 'CC_images' directory & remove all formatting, HTML tags, '\n' characters and make it human readable and save & replace .txt file
# import os
# import re
# def clean_texts(directory):