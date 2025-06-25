#have live feed but it runs in openCV

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import queue
import time

# Shared frame queue for multithreading
frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

def camera_thread(stop_event):
    """Thread for continuous frame acquisition"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Increased to 30fps
   
    try:
        pipeline.start(config)
        print("Camera streaming started...")
       
        # Enable auto-exposure for better motion handling
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
    """Thread for smooth display"""
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('RealSense Live Feed', frame)
       
        # Control display rate (1ms delay for smooth rendering)
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
