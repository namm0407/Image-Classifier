import cv2
import numpy as np
import io
from PIL import Image
import pyrealsense2 as rs
import ollama

def describe_frame(frame):
    try:
        # Convert frame (BGR) to RGB and then to JPEG
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Send frame to LLaVA model
        response = ollama.chat(
            model='llava:13b',
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
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Color stream

    try:
        # Start streaming
        pipeline.start(config)
        print("RealSense camera started.")

        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Error: Could not read frame.")
                continue

            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Describe the current frame
            description = describe_frame(frame)
            print("Frame Description:", description)

            # Display the frame
            cv2.imshow('RealSense Webcam Feed', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: Add a delay to control frame rate (e.g., 1 second)
            cv2.waitKey(1000)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Stop streaming and release resources
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()