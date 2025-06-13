import google.generativeai as genai

"""
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "Explain how AI works in a few words"
          }
        ]
      }
    ]
  }'
"""

def get_image_description(image_path):
    genai.configure(api_key='AIzaSyC5YkGur9cSxFSBotVm1q5JTYO-bpnNGGY')

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([
        {'text': 'What is in this image?'},
        {'inline_data': {'mime_type': 'image/png', 'data': image_data}}
    ])
    print(response)

get_image_description('CC_images/data-storytelling/aF93i6zVVQg_sddefault.webp')
