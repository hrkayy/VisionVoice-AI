import whisper
import tempfile
import os

model = whisper.load_model("tiny")

def speech_to_text(audio_file):

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.read())
            temp_path = f.name

        result = model.transcribe(temp_path, language="en", temperature=0.0, fp16=False)

        os.remove(temp_path)

        return result["text"]

    except Exception as e:
        return f"Error: {e}"