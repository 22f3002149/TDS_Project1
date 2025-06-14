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

def get_chunks(file_path: str, chunk_size: int = 15000):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    _overlap = 500
    splitter = MarkdownSplitter(chunk_size, overlap=_overlap)
    #splitter = MarkdownSplitter(chunk_size)
    chunks = splitter.chunks(content)
    return chunks

file_path = "test/archive.md"
chunks = get_chunks(file_path)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[1]}")