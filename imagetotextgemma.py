import ollama
import os

#iterate through all images in CC_images directory and subdirectories
image_dir = 'images'
if not os.path.exists(image_dir):
    print(f"Directory {image_dir} does not exist.")
    exit(1)
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            image_files.append(os.path.join(root, file))
if not image_files:
    print("No image files found in the directory.")
    exit(1)

# Iterate through each image file and generate a description
for image_path in image_files:
    # Prepare the message for the Ollama model
    print(f"Processing image: {image_path}")
    messages = [{
        'role': 'system',
        'content': 'You are a professional data analyst and image description generator. You will generate detailed point-by-point descriptions of images based on their content. Your descriptions should be concise, informative, and relevant to the image provided.'
        },        
        {
        'role': 'user',
        'content': 'What is in this image?',
        'images': [image_path]
    }]
    
    # Call the Ollama chat API
    response = ollama.chat(model='gemma3:27b', messages=messages)
    
    # Extract the description from the response
    # description = response['choices'][0]['message']['content']
    
    # Save the description to a text file with the same name as the image
    text_file_path = os.path.splitext(image_path)[0] + '.txt'
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        # Write the content string from the response
        if hasattr(response, 'content'):
            text_file.write(response.content)
        elif isinstance(response, dict) and 'content' in response:
            text_file.write(response['content'])
        else:
            text_file.write(str(response))
    
    print(f"Generated description for {image_path} and saved to {text_file_path}")
