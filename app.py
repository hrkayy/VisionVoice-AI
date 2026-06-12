import streamlit as st
import cv2
import time
from vision import detect_objects
from alerts import speak_async
from utils import risk_level
from speech import speech_to_text
from gestures import detect_gesture

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="VisionVoice AI Assistant",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PREMIUM DARK HUD THEME CSS
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');

/* Global Font and Background overrides */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #0F172A, #020617);
    color: #F8FAFC;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0B0F19 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Titles and Headers */
.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.sub-title {
    font-size: 18px;
    color: #94A3B8;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 300;
}

/* Cards & Glassmorphism */
.glass-panel {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* Metrics and Status indicators */
.hud-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748B;
    margin-bottom: 4px;
}

.hud-value {
    font-size: 28px;
    font-weight: 700;
    color: #F1F5F9;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    text-align: center;
}
.badge-high {
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.badge-medium {
    background-color: rgba(245, 158, 11, 0.2);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.badge-low {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

/* Caption board */
.caption-box {
    background: rgba(15, 23, 42, 0.95);
    border-left: 5px solid #38BDF8;
    border-radius: 12px;
    padding: 25px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    text-align: center;
    color: #F8FAFC;
    font-weight: 600;
    box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.25);
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #475569;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR NAVIGATION & SYSTEM METRICS
# ==================================================
with st.sidebar:
    st.markdown("<div style='text-align: center; font-size: 60px;'>♿</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 0px;'>AccessHub AI</h2>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center;'>Multimodal Assistive System for Visually & Hearing Impaired</p>", unsafe_allow_html=True)
    st.divider()

    # Navigation Mode Select
    mode = st.selectbox(
        "Select System Mode",
        [
            "📷 Assistive Vision AI",
            "✋ Sign Language Translator",
            "🎤 Speech to Text (Captions)"
        ]
    )

    st.divider()
    st.markdown("### ⚙️ System Status")
    st.success("🟢 AI Models Loaded")
    st.info("📷 Camera Device Ready")
    
    st.divider()
    st.markdown("""
    **Technologies Engine:**
    - OpenCV & MediaPipe Hands
    - YOLOv8s Detection Model
    - Whisper AI Speech Engine
    - Multi-threaded SAPI5 TTS
    """)

# ==================================================
# HEADER SECTION
# ==================================================
st.markdown("<div class='main-title'>ACCESS-ASSIST MULTIMODAL AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>High-performance cognitive assistance and environmental awareness</div>", unsafe_allow_html=True)

# ==================================================
# VOICE ALERT HANDLER (PREVENTS SPAM)
# ==================================================
def handle_voice_alerts(objects, risk, positions):
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0.0
    if "last_alert_type" not in st.session_state:
        st.session_state.last_alert_type = None

    current_time = time.time()
    cooldown = 6.0  # seconds cooldown for identical announcements

    # New check: immediate alert when a person is detected
    if "person" in objects:
        alert_type = "person_detected"
        alert_text = "Person detected: unsafe environment."
        time_elapsed = current_time - st.session_state.last_alert_time
        if alert_type != st.session_state.last_alert_type or time_elapsed > cooldown:
            speak_async(alert_text)
            st.session_state.last_alert_time = current_time
            st.session_state.last_alert_type = alert_type
            st.session_state.current_voice_alert = alert_text
        # Skip further processing for this frame to avoid duplicate alerts
        return

    # 1. Check if empty space
    if not objects or not positions:
        alert_type = "safe"
        alert_text = "Environment is safe."
        
        time_elapsed = current_time - st.session_state.last_alert_time
        if alert_type != st.session_state.last_alert_type or time_elapsed > cooldown:
            speak_async(alert_text)
            st.session_state.last_alert_time = current_time
            st.session_state.last_alert_type = alert_type
            st.session_state.current_voice_alert = alert_text
        return

    # 2. Check for Immediate Emergency Escalation (extremely close obstacle)
    # y_height > 0.55 means the obstacle occupies more than 55% of the frame height
    close_obstacles = [p for p in positions if p["y_height"] > 0.55]
    if close_obstacles:
        alert_type = "emergency"
        alert_text = f"Emergency! Close {close_obstacles[0]['label']}. Stop!"
        
        # Trigger play_siren (non-blocking) and immediate speech
        from alerts import play_siren
        play_siren()
        speak_async("Warning! Stop!")
        
        st.session_state.last_alert_time = current_time
        st.session_state.last_alert_type = alert_type
        st.session_state.current_voice_alert = alert_text
        return

    # 3. Spatial Navigation Logic (Center, Left, Right)
    center_obstacles = [p for p in positions if 0.35 <= p["x_center"] <= 0.65]
    left_obstacles = [p for p in positions if p["x_center"] < 0.35]
    right_obstacles = [p for p in positions if p["x_center"] > 0.65]

    if center_obstacles:
        closest_center_label = center_obstacles[0]["label"]
        if not left_obstacles:
            alert_type = "steer_left"
            alert_text = f"{closest_center_label} ahead. Turn left."
        elif not right_obstacles:
            alert_type = "steer_right"
            alert_text = f"{closest_center_label} ahead. Turn right."
        else:
            alert_type = "steer_stop"
            alert_text = f"Path blocked by {closest_center_label}. Stop."
    elif left_obstacles:
        alert_type = "steer_right_caution"
        alert_text = f"Obstacle on left. Turn right."
    elif right_obstacles:
        alert_type = "steer_left_caution"
        alert_text = f"Obstacle on right. Turn left."
    else:
        alert_type = "general"
        alert_text = f"{objects[0].capitalize()} detected."

    # Speak if type changed OR if identical type has waited long enough
    time_elapsed = current_time - st.session_state.last_alert_time
    if alert_type != st.session_state.last_alert_type or time_elapsed > cooldown:
        speak_async(alert_text)
        st.session_state.last_alert_time = current_time
        st.session_state.last_alert_type = alert_type
        st.session_state.current_voice_alert = alert_text
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0.0
    if "last_alert_type" not in st.session_state:
        st.session_state.last_alert_type = None

    current_time = time.time()
    cooldown = 6.0  # seconds cooldown for identical announcements

    # 1. Check if empty space
    if not objects or not positions:
        alert_type = "safe"
        alert_text = "Environment is safe."
        
        time_elapsed = current_time - st.session_state.last_alert_time
        if alert_type != st.session_state.last_alert_type or time_elapsed > cooldown:
            speak_async(alert_text)
            st.session_state.last_alert_time = current_time
            st.session_state.last_alert_type = alert_type
            st.session_state.current_voice_alert = alert_text
        return

    # 2. Check for Immediate Emergency Escalation (extremely close obstacle)
    # y_height > 0.55 means the obstacle occupies more than 55% of the frame height
    close_obstacles = [p for p in positions if p["y_height"] > 0.55]
    if close_obstacles:
        alert_type = "emergency"
        alert_text = f"Emergency! Close {close_obstacles[0]['label']}. Stop!"
        
        # Trigger play_siren (non-blocking) and immediate speech
        from alerts import play_siren
        play_siren()
        speak_async("Warning! Stop!")
        
        st.session_state.last_alert_time = current_time
        st.session_state.last_alert_type = alert_type
        st.session_state.current_voice_alert = alert_text
        return

    # 3. Spatial Navigation Logic (Center, Left, Right)
    center_obstacles = [p for p in positions if 0.35 <= p["x_center"] <= 0.65]
    left_obstacles = [p for p in positions if p["x_center"] < 0.35]
    right_obstacles = [p for p in positions if p["x_center"] > 0.65]

    if center_obstacles:
        closest_center_label = center_obstacles[0]["label"]
        if not left_obstacles:
            alert_type = "steer_left"
            alert_text = f"{closest_center_label} ahead. Turn left."
        elif not right_obstacles:
            alert_type = "steer_right"
            alert_text = f"{closest_center_label} ahead. Turn right."
        else:
            alert_type = "steer_stop"
            alert_text = f"Path blocked by {closest_center_label}. Stop."
    elif left_obstacles:
        alert_type = "steer_right_caution"
        alert_text = f"Obstacle on left. Turn right."
    elif right_obstacles:
        alert_type = "steer_left_caution"
        alert_text = f"Obstacle on right. Turn left."
    else:
        alert_type = "general"
        alert_text = f"{objects[0].capitalize()} detected."

    # Speak if type changed OR if identical type has waited long enough
    time_elapsed = current_time - st.session_state.last_alert_time
    if alert_type != st.session_state.last_alert_type or time_elapsed > cooldown:
        speak_async(alert_text)
        st.session_state.last_alert_time = current_time
        st.session_state.last_alert_type = alert_type
        st.session_state.current_voice_alert = alert_text

# Initialize voice message status
if "current_voice_alert" not in st.session_state:
    st.session_state.current_voice_alert = "No active voice announcement"

# ==================================================
# FEATURE RUNNERS
# ==================================================

# --- MODE 1: ASSISTIVE VISION AI ---
if mode == "📷 Assistive Vision AI":
    st.markdown("### 📷 Live Camera Obstacle & Risk Detection")
    st.write("Identifies key objects (people, tables, chairs, etc.) in the camera path, computes situational risk levels, and reads warnings out loud.")

    # Two column layout: Left = Camera Feed, Right = HUD Details
    left_col, right_col = st.columns([5, 3])

    with right_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h4>🧠 AI Perception Metrics</h4>", unsafe_allow_html=True)
        st.divider()
        
        # Risk Badge Placeholder
        risk_badge_placeholder = st.empty()
        
        # Objects list Placeholder
        objects_placeholder = st.empty()
        
        # TTS Status
        st.markdown("<div class='hud-label'>Voice Assistant Status</div>", unsafe_allow_html=True)
        voice_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with left_col:
        run_camera = st.checkbox("Start Live Stream", value=True, key="vision_cam_toggle")
        frame_placeholder = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        # Limit buffer size to 1 to avoid queue buildup and real-time lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            st.error("❌ Unable to connect to camera device. Please ensure it's plugged in and not in use by another app.")
        else:
            try:
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame.")
                        break

                    # YOLO detection
                    ret_vals = detect_objects(frame)
                    if len(ret_vals) == 4:
                        annotated_frame, objects, _, positions = ret_vals
                    else:
                        annotated_frame, objects, _ = ret_vals
                        positions = []
                    
                    # Compute risk level based on the updated utils mapping rules
                    risk = risk_level(objects, positions)
                    
                    # Display camera feed
                    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                    # Update Risk Badge UI
                    if risk == "HIGH RISK":
                        badge_html = "<span class='badge badge-high'>🔴 HIGH RISK (Person)</span>"
                    elif risk == "MEDIUM RISK":
                        badge_html = "<span class='badge badge-medium'>⚠️ MEDIUM RISK (Obstacle)</span>"
                    else:
                        badge_html = "<span class='badge badge-low'>✅ LOW RISK (Empty Space)</span>"

                    risk_badge_placeholder.markdown(f"""
                        <div style='margin-bottom: 20px;'>
                            <div class='hud-label'>Situation Risk Level</div>
                            <div style='margin-top: 5px;'>{badge_html}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Update Detected Objects UI
                    if objects:
                        objs_html = "".join([f"<li style='font-size:16px; margin-bottom:5px;'>🔹 {obj.capitalize()}</li>" for obj in objects])
                        objects_placeholder.markdown(f"""
                            <div style='margin-bottom: 20px;'>
                                <div class='hud-label'>Obstacles Detected</div>
                                <ul style='list-style-type: none; padding-left: 0; margin-top: 5px;'>{objs_html}</ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        objects_placeholder.markdown("""
                            <div style='margin-bottom: 20px;'>
                                <div class='hud-label'>Obstacles Detected</div>
                                <p style='color: #64748B; font-style: italic; margin-top: 5px;'>No obstacles (Empty Space)</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Trigger non-spam voice alert
                    handle_voice_alerts(objects, risk, positions)
                    
                    # Update TTS status placeholder
                    voice_placeholder.markdown(f"<p style='color:#38BDF8; font-weight:600; font-size:16px;'>🔊 {st.session_state.current_voice_alert}</p>", unsafe_allow_html=True)

                    # Short yield to allow other parts of streamlit to run
                    time.sleep(0.01)

            finally:
                cap.release()

# --- MODE 2: SIGN LANGUAGE TRANSLATOR ---
elif mode == "✋ Sign Language Translator":
    st.markdown("### ✋ Sign Language to Text Translator")
    st.write("Displays the hand skeleton landmark tracking and translates hand configurations to text captions in real time.")

    left_col, right_col = st.columns([5, 3])

    with right_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<h4>✋ Sign Language Captions</h4>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='hud-label'>Translated Output Text</div>", unsafe_allow_html=True)
        
        caption_placeholder = st.empty()
        
        st.markdown("<br><p style='font-size: 13px; color: #64748B;'>Show one of these gestures to communicate:<br>"
                    "1. ✊ <b>Fist</b>: Stop / Emergency<br>"
                    "2. ✋ <b>Open Hand</b>: Hello / Help<br>"
                    "3. 👍 <b>Thumbs Up</b>: Yes / OK<br>"
                    "4. 👎 <b>Thumbs Down</b>: No / Not OK<br>"
                    "5. ✌️ <b>Victory Sign</b>: Peace / Success<br>"
                    "6. 👌 <b>OK Sign</b>: Perfect / Understood<br>"
                    "7. 🤟 <b>I Love You</b>: Love / Friendship<br>"
                    "8. 🤙 <b>Call Me</b>: Phone / Call Me<br>"
                    "9. ☝️ <b>Pointing</b>: Look / Attention</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with left_col:
        run_gesture_camera = st.checkbox("Start Live Stream", value=True, key="gesture_cam_toggle")
        frame_placeholder = st.empty()

    if run_gesture_camera:
        cap = cv2.VideoCapture(0)
        # Limit buffer size to 1 to avoid queue buildup and real-time lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            st.error("❌ Unable to connect to camera device. Please ensure it's plugged in and not in use by another app.")
        else:
            try:
                while run_gesture_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame.")
                        break

                    # Detect gesture
                    annotated_frame, gesture_text = detect_gesture(frame)
                    
                    # Mirror image for natural hand movement feel
                    annotated_frame = cv2.flip(annotated_frame, 1)

                    # Display frame
                    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                    # Update caption board UI
                    caption_placeholder.markdown(f"""
                        <div class='caption-box'>
                            {gesture_text}
                        </div>
                    """, unsafe_allow_html=True)

                    # Short yield
                    time.sleep(0.01)
            finally:
                cap.release()

# --- MODE 3: SPEECH TO TEXT ---
elif mode == "🎤 Speech to Text (Captions)":
    st.markdown("### 🎤 English Speech to Text Captioning")
    st.write("Designed for hearing-impaired individuals. Record spoken English words and read them clearly as large, accurate English subtitles.")

    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("<h4>🎤 Record Audio Input</h4>", unsafe_allow_html=True)
    
    # Audio recorder input widget
    audio_file = st.audio_input("Click to record audio (English voice)")

    if audio_file:
        st.success("🎤 Audio recorded successfully!")
        
        # Audio Player (Allows playing back the recorded voice)
        st.markdown("<div class='hud-label' style='margin-top: 15px;'>Recorded Voice Playback</div>", unsafe_allow_html=True)
        st.audio(audio_file)

        # Transcribe with spinner
        with st.spinner("🧠 Whisper AI transcribing speech to English..."):
            text = speech_to_text(audio_file)

        # Display proper English captions in a beautiful cinema subtitle box
        st.markdown("<div class='hud-label' style='margin-top: 25px;'>English Subtitles / Captions</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='caption-box'>
                💬 "{text.strip() if text.strip() else '[Unrecognized Speech]'}"
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# FOOTER SECTION
# ==================================================
st.markdown(
    """
    <div class="footer">
        AccessHub AI • Built with Streamlit, Ultralytics YOLOv8, MediaPipe Hands & OpenAI Whisper
    </div>
    """,
    unsafe_allow_html=True
)