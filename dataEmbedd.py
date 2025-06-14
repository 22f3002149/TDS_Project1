# create a python script that runs through all files in 'test' directory and extract content, then split it into chunks of 1000 characters with an overlap of 100 characters, and embedd each chunk using openai, then save both chunk and embedding to a compressed numpy array and save it to a file called 'embeddings.npz'.
import os
import numpy as np
from openai import OpenAI

client = OpenAI()
from glob import glob
from tqdm import tqdm
def extract_content_from_files(directory):
    content = []
    files = glob(os.path.join(directory, "*.md"))
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content.append(file.read())
    return content
def split_content_into_chunks(content, chunk_size=1000, overlap=100):
    chunks = []
    for text in content:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if len(chunk) < chunk_size:
                break
            chunks.append(chunk)
            start += chunk_size - overlap  # Move start forward by chunk_size - overlap
    return chunks
def embed_chunks(chunks):
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding chunks"):
        response = client.embeddings.create(input=chunk,
        model="text-embedding-3-small")
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return np.array(embeddings)
def save_embeddings(chunks, embeddings, filename='embeddings.npz'):
    np.savez_compressed(filename, chunks=chunks, embeddings=embeddings)
    print(f"Embeddings saved to {filename}")
def main():
    directory = 'test'
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    content = extract_content_from_files(directory)
    if not content:
        print("No content found in the specified directory.")
        return
    chunks = split_content_into_chunks(content)
    if not chunks:
        print("No chunks created from the content.")
        return
    embeddings = embed_chunks(chunks)
    save_embeddings(chunks, embeddings)
main()
# Note: Ensure you have set your OpenAI API key in the environment variable OPENAI_API_KEY
# or configure it in your OpenAI client before running this script.
# Make sure to install the required packages before running the script:
# pip install openai numpy tqdm