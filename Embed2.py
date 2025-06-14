import hashlib
import httpx
import json
import os
import time
import numpy as np
from  pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
#from google import genai
from semantic_text_splitter import MarkdownSplitter
from openai import OpenAI

# Get chunks from a file, splitting the content into manageable pieces.
def get_chunks(file_path: str, chunk_size: int = 1000):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    _overlap = 200
    splitter = MarkdownSplitter(chunk_size, overlap=_overlap)
    #splitter = MarkdownSplitter(chunk_size)
    chunks = splitter.chunks(content)
    return chunks

# Get embeddings for a given text using OpenAI API.
def get_embeddings(text: str) -> list:
    """ Get embeddings for a given text using OpenAI API. """
    client = OpenAI()
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding





file_path = "test/archive.md"
chunks = get_chunks(file_path)
em = get_embeddings(chunks[3])

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[3]}")
print(len(em))
print(f"Embedding for the chunk: {em}")  