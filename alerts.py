import sys
import threading
import queue
import pyttsx3
import pythoncom

# Global queue to coordinate speech alerts safely
speech_queue = queue.Queue()

# Global speech rate configuration (WPM)
current_speech_rate = 150
rate_lock = threading.Lock()

def set_speech_rate(rate):
    """Sets the speech rate dynamically (Words per Minute)."""
    global current_speech_rate
    with rate_lock:
        current_speech_rate = rate

def _speech_worker():
    """Single background thread worker that handles speech sequentially to avoid COM conflicts."""
    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        
        while True:
            text = speech_queue.get()
            if text is None:  # Shutdown signal
                break
            try:
                with rate_lock:
                    rate = current_speech_rate
                engine.setProperty('rate', rate)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("Speech worker play error:", e)
            finally:
                speech_queue.task_done()
    except Exception as e:
        print("Speech worker initialization error:", e)
    finally:
        pythoncom.CoUninitialize()

# Start the speech worker thread immediately as a daemon thread
worker_thread = threading.Thread(target=_speech_worker, daemon=True)
worker_thread.start()

def speak_async(text):
    """
    Adds text to the speech queue. Clears older pending messages first 
    so safety warnings are spoken immediately without queue backup.
    """
    if not text:
        return
        
    # Clear any pending stale warnings in queue
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except (queue.Empty, ValueError):
            break
            
    # Queue the new alert
    speech_queue.put(text)

def play_siren():
    """
    Plays a high-priority dual-tone emergency siren in a background thread.
    Works on both Windows (winsound.Beep) and Linux/Raspberry Pi.
    """
    def run():
        if sys.platform == "win32":
            import winsound
            try:
                # Play high-pitch emergency sweep sound (dual-tone)
                for freq in [900, 1300, 900, 1300, 900, 1300]:
                    winsound.Beep(freq, 120)
            except Exception as e:
                print("Siren playback error:", e)
        else:
            # On Linux/Pi, trigger hardware terminal beeps and espeak
            print("\a")  # System terminal beep
            import os
            os.system("echo -en '\a'; sleep 0.1; echo -en '\a'; sleep 0.1; echo -en '\a'")
            os.system("espeak 'Emergency! Stop!' &")
            
    threading.Thread(target=run, daemon=True).start()

def generate_alert(objects):
    if "person" in objects:
        return "Person detected"
    if "car" in objects:
        return "Vehicle nearby"
    return None
