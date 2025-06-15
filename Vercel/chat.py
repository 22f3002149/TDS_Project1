# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "argparse",
#   "fastapi",
#   "httpx",
#   "markdownify",
#   "numpy",
#   "semantic_text_splitter",
#   "tqdm",
#   "uvicorn",
#   "google-genai",
#   "google.generativeai ",
#   "openai",
#   "pydantic",
#   "pillow"
# ]
# ///


import argparse
import base64
import json
import numpy as np
import os
import re
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import google.generativeai as genai
import time
import openai
import requests
from openai import OpenAI

app = FastAPI()

@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        #print(data)
        return answer(data.get("question"), data.get("image"))
    except Exception as e:
        return {"error": str(e)}

def answer(question: str, image: str = None):
    loaded_chunks, loaded_embeddings = load_embeddings()
    if image:
        image_description = get_image_description(f"data:image/jpeg;base64,{image}")
        question += f" {image_description}"
    question_embedding = get_embedding(question)
    # Calculate cosine similarity
    similarities = np.dot(loaded_embeddings, question_embedding) / (
        np.linalg.norm(loaded_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    # Get the index of the 10 most similar chunks
    top_indices = np.argsort(similarities)[-10:][::-1]
    # Get the top chunks
    top_chunks = [loaded_chunks[i] for i in top_indices]

    response = generate_openai_response(question, "\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }

def load_embeddings():
    data = np.load("embeddings.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]

def get_image_description(image_path):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    my_file = client.files.upload(file=image_path)
    response = client.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, "Describe the content of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it."],
    )
    return response.text

def get_embedding(question):
    client = OpenAI()
    response = client.embeddings.create(input=question,
    model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding

# create a python function that take a questiona and context & generate a response using open ai https://api.openai.com/v1 API using 3-small model
def generate_openai_response(question: str, context: str) -> str:
    """Generate a response using OpenAI's text-embedding-3-small model via the API."""
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",  # 'text-embedding-3-small' is for embeddings, not chat. Use chat model for text generation.
        "messages": [
            {"role": "system", 
            "content": """You are a knowledgeable and concise teaching assistant. Use only the information provided in the context to answer the question.
            > * Format your response using **Markdown**.
            > * Use code blocks (```) for any code or command-line instructions.
            > * Use bullet points or numbered lists for clarity where appropriate.
            > * Always include a brief introduction or heading if needed.
            > * Always include links to reference context if available.
            > **Important:** If the question is not answerable with the provided context, respond with "I don't know".
            > Do not attempt to guess, fabricates, or add external information.
            """
            },
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


ans = answer("project1 for file not detecting after sending post request")
print(ans)

#project1 for file not detecting after sending post request