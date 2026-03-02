import ollama

def analyze_medical_image(image_path, prompt="Describe any abnormalities in this X-ray"):
    # moondream handles the image input natively
    res = ollama.chat(
        model="moondream",
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }]
    )
    return res['message']['content']