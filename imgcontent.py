import ollama
import os

def generate_image_description(image_path):
    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant who extract python codes from images, which should be ready to copy and paste'
        },        
        {
        'role': 'user',
        'content': 'What is in this image?',
        'images': [image_path]
    }]
    
    # Call the Ollama chat API
    response = ollama.chat(model='gemma3:27b', messages=messages)
    if response and 'content' in response['message']:
        return response['message']['content']
    else:
        return "No description generated."

print(generate_image_description('ImgData/01.jpg'))  # Replace with your actual image path