import whisper
from dotenv import load_dotenv
import os

load_dotenv()

model = whisper.load_model("base")

def transcribe_and_translate(audio_path):
    """
    This function takes an audio file path, transcribes it, and translates the speech to text.
    """
    result = model.transcribe(audio_path, task="translate")
    return result["text"]
