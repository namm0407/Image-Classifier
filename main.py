import cv2
import ollama
import numpy as np
import io
from PIL import Image

def describe_frame(frame):
    try:
        # Convert OpenCV frame (BGR) to RGB and then to JPEG
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Send frame to LLaVA model
        response = ollama.chat(
            model='llava:7b',
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [img_bytes],
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error processing frame: {str(e)}"

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Describe the current frame
            description = describe_frame(frame)
            print("Frame Description:", description)

            # Display the frame
            cv2.imshow('Webcam Feed', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: Add a delay to control frame rate (e.g., 1 second)
            cv2.waitKey(1000)

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()