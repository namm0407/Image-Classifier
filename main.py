import ollama

def describe_image(image_path):
    try:
        with open(image_path, 'rb') as file:
            response = ollama.chat(
                model='llava:13b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in detail.',
                        'images': [file.read()],
                    }
                ]
            )
        return response['message']['content']
    except Exception as e:
        return f"Error processing image: {str(e)}"


# Example usage
if __name__ == "__main__":
    image_path = "cat.png"  
    description = describe_image(image_path)
    print("Image Description:", description)
