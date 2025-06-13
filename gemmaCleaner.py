# Write a python script that runs through all .txt files in 'CC_images' directory and extracts texts after '(role='assistant', content=' and before ', thinking=None,' and replaces the text in the file with the extracted text.
import os
import re
def extract_and_replace_texts(directory):
    # Iterate through all .txt files in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use regex to find the text after '(role='assistant', content=' and before ', thinking=None,'
                matches = re.findall(r"\(role='assistant', content='(.*?)', thinking=None,", content)
                
                if matches:
                    # Join all matches into a single string
                    extracted_text = ' '.join(matches)
                    
                    # Write the extracted text back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    
                    print(f"Processed {file_path}: Extracted and replaced text.")
                else:
                    print(f"No matching text found in {file_path}.")

directory = 'CC_images'
extract_and_replace_texts(directory)
