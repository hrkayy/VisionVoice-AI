import cv2
import time
import sys
import wave
import numpy as np
import sounddevice as sd
from vision import detect_objects
from gestures import detect_gesture
from utils import risk_level
from alerts import speak_async, play_siren
from speech import speech_to_text

# Define recording sample rate
SAMPLE_RATE = 16000
AUDIO_TEMP_FILE = "pi_audio_input.wav"

def record_and_transcribe(duration=4):
    """Records audio using sounddevice and transcribes it using Whisper."""
    print(f"\n🎤 [Speech Assistant] Recording for {duration} seconds... SPEAK NOW!")
    speak_async("Recording started. Please speak.")
    
    try:
        # Record 16-bit mono audio
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
        sd.wait()  # Wait for recording to complete
        print("✅ [Speech Assistant] Recording complete. Transcribing...")
        
        # Save to WAV file using standard python wave module
        with wave.open(AUDIO_TEMP_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
            
        # Transcribe audio using the speech module (Whisper tiny)
        class MockAudioFile:
            def read(self):
                with open(AUDIO_TEMP_FILE, 'rb') as f:
                    return f.read()
                    
        transcript = speech_to_text(MockAudioFile())
        print(f"\n💬 [Speech Subtitles]: \"{transcript.strip()}\"\n")
        speak_async(f"Captions: {transcript}")
        
    except Exception as e:
        print("❌ Recording error:", e)
        speak_async("Recording error.")

def run_pi_app():
    # Setup window
    window_name = "AccessHub AI - Standalone Pi deployment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Initialize camera
    print("📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)
    # Set buffer to 1 to avoid queue buildup and real-time lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the camera device.")
        sys.exit(1)
        
    mode = "VISION"  # Modes: "VISION" or "GESTURE"
    print("\n🟢 Standalone System Online!")
    print("==================================================")
    print("⌨️ Keyboard Shortcuts:")
    print("  [V] - Toggle Mode (Obstacle Detection <-> Sign Gestures)")
    print("  [S] - Record English Speech (Speech to Text)")
    print("  [Q] - Quit Application")
    print("==================================================")
    
    speak_async("System online. Vision mode active.")
    
    last_alert_time = 0.0
    last_alert_type = None
    cooldown = 6.0  # Cooldown for voice announcements
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break
                
            h, w, _ = frame.shape
            display_frame = frame.copy()
            
            if mode == "VISION":
                # YOLO Detection
                ret_vals = detect_objects(frame)
                if len(ret_vals) == 4:
                    annotated_frame, objects, risk, positions = ret_vals
                else:
                    annotated_frame, objects, risk = ret_vals
                    positions = []
                display_frame = annotated_frame
                
                # Compute risk level badge string
                lvl = risk_level(objects)
                
                # Draw risk overlay on screen
                color = (0, 255, 0)
                if lvl == "HIGH RISK":
                    color = (0, 0, 255)
                elif lvl == "MEDIUM RISK":
                    color = (0, 165, 255)
                    
                cv2.putText(display_frame, f"Risk: {lvl}", (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Non-spam Voice Alerts & Steering
                current_time = time.time()
                
                # Check for emergency close obstacles
                close_obstacles = [p for p in positions if p["y_height"] > 0.55]
                if close_obstacles:
                    alert_type = "emergency"
                    alert_text = f"Stop!"
                    play_siren()
                    speak_async(alert_text)
                    last_alert_time = current_time
                    last_alert_type = alert_type
                else:
                    # Spatial steering logic
                    center_obstacles = [p for p in positions if 0.35 <= p["x_center"] <= 0.65]
                    left_obstacles = [p for p in positions if p["x_center"] < 0.35]
                    right_obstacles = [p for p in positions if p["x_center"] > 0.65]
                    
                    if not objects:
                        alert_type = "safe"
                        alert_text = "Environment is safe."
                    elif center_obstacles:
                        closest_center_label = center_obstacles[0]["label"]
                        if not left_obstacles:
                            alert_type = "steer_left"
                            alert_text = "Obstacle ahead. Turn left."
                        elif not right_obstacles:
                            alert_type = "steer_right"
                            alert_text = "Obstacle ahead. Turn right."
                        else:
                            alert_type = "steer_stop"
                            alert_text = "Path blocked. Stop."
                    elif left_obstacles:
                        alert_type = "steer_right_caution"
                        alert_text = "Obstacle on left. Turn right."
                    elif right_obstacles:
                        alert_type = "steer_left_caution"
                        alert_text = "Obstacle on right. Turn left."
                    else:
                        alert_type = "general"
                        alert_text = f"{objects[0]} detected."
                        
                    # Handle Speech trigger
                    time_elapsed = current_time - last_alert_time
                    if alert_type != last_alert_type or time_elapsed > cooldown:
                        speak_async(alert_text)
                        last_alert_time = current_time
                        last_alert_type = alert_type
                        
            elif mode == "GESTURE":
                # MediaPipe Gesture Translation
                annotated_frame, gesture_text = detect_gesture(frame)
                
                # Mirror frame for natural tracking display
                display_frame = cv2.flip(annotated_frame, 1)
                
                # Draw output text overlay
                cv2.rectangle(display_frame, (5, h - 50), (w - 5, h - 5), (15, 23, 42), -1)
                cv2.putText(display_frame, f"Sign: {gesture_text}", (15, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2, cv2.LINE_AA)
            
            # Draw Mode Overlay
            cv2.putText(display_frame, f"Mode: {mode}", (w - 180, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 189, 56), 2, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Check keys
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('v'):
                # Toggle mode
                mode = "GESTURE" if mode == "VISION" else "VISION"
                print(f"🔄 Switched Mode to: {mode}")
                speak_async(f"Switched to {mode.lower()} mode.")
                last_alert_type = None
            elif key == ord('s'):
                # Stop camera temporarily to record audio
                cv2.destroyAllWindows()
                record_and_transcribe()
                # Recreate window
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 System shut down successfully.")

if __name__ == "__main__":
    run_pi_app()
