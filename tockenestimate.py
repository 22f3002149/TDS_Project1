# Calcualte tockens required for open ai embedding for all file contents in '01c_CourseContentMdWithImageDes' & '02f_DiscourseMegredPosts' together.
import os
import tiktoken
def count_tokens_in_directory(directory):
    total_tokens = 0
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tokens = encoding.encode(content)
                    total_tokens += len(tokens)
    
    return total_tokens
    
if __name__ == "__main__":
    directory = '01c_CourseContentMdWithImageDes'
    total_tokens = count_tokens_in_directory(directory)
    print(f"Total tokens in '{directory}': {total_tokens}")
    
    directory = '02f_DiscourseMegredPosts'
    total_tokens += count_tokens_in_directory(directory)
    print(f"Total tokens in '02f_DiscourseMegredPosts': {total_tokens}")
    
    print(f"Total tokens across both directories: {total_tokens}")
# This script calculates the total number of tokens required for OpenAI embeddings
# for all file contents in the specified directories. It uses the `tiktoken` library to encode the text and count the tokens.
# Make sure to install the tiktoken library if you haven't already: