import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import time
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

# Shared frame queue for multithreading
frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

# Initialize LLaVA model (load only once)
model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading LLaVA model on {device}...")
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("LLaVA model loaded!")

def analyze_frame(frame):
    """Use LLaVA to analyze the frame and generate a description"""
    prompt = "USER: <image>\nDescribe this scene in detail.\nASSISTANT:"
    inputs = processor(text=prompt, images=frame, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=100)
    description = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return description.split("ASSISTANT:")[-1].strip()

def camera_thread(stop_event):
    """Thread for continuous frame acquisition"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
   
    try:
        pipeline.start(config)
        print("Camera streaming started...")
       
        # Enable auto-exposure
        profile = pipeline.get_active_profile()
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
       
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                if frame_queue.empty():  # Only add if queue isn't full
                    frame_queue.put(frame)
           
    finally:
        pipeline.stop()
        print("Camera thread stopped")

def display_thread(stop_event):
    """Thread for display and LLaVA analysis"""
    last_update_time = time.time()
    description = "Waiting for analysis..."
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Process frame with LLaVA every 2 seconds (to reduce load)
            current_time = time.time()
            if current_time - last_update_time > 2.0:
                try:
                    # Convert BGR (OpenCV) to RGB (LLaVA expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    description = analyze_frame(rgb_frame)
                    last_update_time = current_time
                except Exception as e:
                    print(f"LLaVA error: {e}")
                    description = "Analysis failed"
            
            # Display frame with description
            display_frame = frame.copy()
            cv2.putText(display_frame, description, (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('RealSense + LLaVA', display_frame)
       
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def main():
    # Start camera thread
    cam_thread = threading.Thread(target=camera_thread, args=(stop_event,))
    cam_thread.start()
   
    # Start display thread
    disp_thread = threading.Thread(target=display_thread, args=(stop_event,))
    disp_thread.start()
   
    try:
        while not stop_event.is_set():
            time.sleep(0.1)  # Reduce CPU usage
           
    except KeyboardInterrupt:
        stop_event.set()
   
    # Cleanup
    cam_thread.join()
    disp_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
