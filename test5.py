#live feed with object idenitifier

import cv2
import numpy as np
import pyrealsense2 as rs
import ollama
from collections import deque
import time

class ObjectDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 2  # seconds between full detections
        self.current_objects = []
        self.object_history = deque(maxlen=10)
        
    def detect_objects(self, frame):
        # Only do full detection every few seconds
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.current_objects
        
        try:
            # Convert frame to JPEG
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_bytes = cv2.imencode('.jpg', rgb_frame)[1].tobytes()

            # Get description from LLaVA
            response = ollama.chat(
                model='llava:13b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Briefly list the main objects in this image with their approximate positions (left, center, right). '
                                  'Respond with format: "object:position", one per line. Example:\n'
                                  'person:center\ncomputer:left\n',
                        'images': [img_bytes],
                    }
                ]
            )
            
            # Parse response
            description = response['message']['content']
            objects = []
            for line in description.split('\n'):
                if ':' in line:
                    obj, pos = line.split(':', 1)
                    obj = obj.strip().lower()
                    pos = pos.strip().lower()
                    objects.append((obj, pos))
            
            self.current_objects = objects
            self.object_history.append((current_time, objects))
            self.last_detection_time = current_time
            return objects
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            return self.current_objects

def position_to_coords(pos, frame_width, frame_height):
    """Convert position string to bounding box coordinates"""
    if pos == 'left':
        return (50, frame_height//2 - 50, 200, frame_height//2 + 50)
    elif pos == 'right':
        return (frame_width - 250, frame_height//2 - 50, frame_width - 100, frame_height//2 + 50)
    else:  # center or unknown
        return (frame_width//2 - 100, frame_height//2 - 50, frame_width//2 + 100, frame_height//2 + 50)

def main():
    # Initialize detector
    detector = ObjectDetector()
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("RealSense camera started.")

        while True:
            # Get frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_height, frame_width = frame.shape[:2]
            
            # Detect objects
            objects = detector.detect_objects(frame)
            
            # Draw bounding boxes and labels
            for obj, pos in objects:
                x1, y1, x2, y2 = position_to_coords(pos, frame_width, frame_height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, obj, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
