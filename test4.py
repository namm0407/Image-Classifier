#live feed but no description

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
analysis_queue = queue.Queue(maxsize=1)  # For analysis results
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

def analysis_worker(stop_event):
    """Dedicated thread for LLaVA analysis"""
    last_frame_time = 0
    while not stop_event.is_set():
        if not frame_queue.empty() and time.time() - last_frame_time > 2.0:
            frame = frame_queue.get()
            try:
                # Convert BGR (OpenCV) to RGB (LLaVA expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                description = analyze_frame(rgb_frame)
                analysis_queue.put(description)
                last_frame_time = time.time()
            except Exception as e:
                print(f"LLaVA error: {e}")
                analysis_queue.put("Analysis failed")

def display_thread(stop_event):
    """Thread for display that maintains smooth video feed"""
    description = "Waiting for analysis..."
    last_description = ""
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Get latest analysis if available
            if not analysis_queue.empty():
                description = analysis_queue.get()
                last_description = description
            
            # Always show the latest frame with the latest description
            display_frame = frame.copy()
            
            # Split the description into multiple lines for better display
            y = 30
            for line in last_description.split('. '):  # Split at sentence endings
                if line.strip():
                    cv2.putText(display_frame, line.strip() + '.', (10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y += 25  # Move down for next line
            
            cv2.imshow('RealSense + LLaVA', display_frame)
       
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def main():
    # Start camera thread
    cam_thread = threading.Thread(target=camera_thread, args=(stop_event,))
    cam_thread.start()
   
    # Start analysis thread
    analysis_worker_thread = threading.Thread(target=analysis_worker, args=(stop_event,))
    analysis_worker_thread.start()
   
    # Start display thread (main thread)
    display_thread(stop_event)
   
    # Cleanup
    cam_thread.join()
    analysis_worker_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
