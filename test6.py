#live feed with detector (not laggy version) with comments in terminal

import cv2
import numpy as np
import pyrealsense2 as rs
import ollama
from collections import deque
import time
import threading
from queue import Queue
import io

class ObjectDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 0.5  # seconds between full detections
        self.current_objects = []
        self.object_history = deque(maxlen=5)  # Reduced history size
        self.result_queue = Queue()
        self.detection_thread = None
        self.running = False
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def stop(self):
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()

    def _detection_loop(self):
        while self.running:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                objects = self._detect_objects(frame)
                with self.lock:
                    self.current_objects = objects
                    self.object_history.append((time.time(), objects))
            time.sleep(0.01)  # Prevent thread from hogging CPU

    def detect_objects(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return self.current_objects

        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.current_objects

        self.last_detection_time = current_time
        
        # Only put new frame if queue is empty to prevent backlog
        if self.result_queue.qsize() < 1:
            # Resize frame before encoding to reduce processing time
            small_frame = cv2.resize(frame, (320, 240))
            self.result_queue.put(small_frame)
        
        with self.lock:
            return self.current_objects

    def _detect_objects(self, frame):
        try:
            # Convert to JPEG with lower quality to reduce processing time
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', rgb_frame, encode_param)
            img_bytes = buffer.tobytes()

            # Use lighter llava model
            response = ollama.chat(
                model='llava:7b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'List main objects with positions (left, center, right). Format: "object:position"',
                        'images': [img_bytes],
                    }
                ],
                options={
                    'temperature': 0.7,
                    'num_predict': 50  # Limit response length
                }
            )

            # Parse response
            description = response['message']['content']
            objects = []
            for line in description.split('\n'):
                if ':' in line:
                    obj, pos = line.split(':', 1)
                    obj = obj.strip().lower()
                    pos = pos.strip().lower()
                    if obj and pos in ['left', 'center', 'right']: 
                        objects.append((obj, pos))
                        print(f"Detected object: {obj} ({pos})")  # Print each object with position
            
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
    detector.start()
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("RealSense camera started.")

        while True:
            # Get frame
            frames = pipeline.wait_for_frames(timeout_ms=1000)
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
            
            # Exit on Enter or Space
            key = cv2.waitKey(1) & 0xFF
            if key in [13, 32]:  # 13 is Enter, 32 is Space
                break
 
    finally:
        detector.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
