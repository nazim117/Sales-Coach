import whisper

model = whisper.load_model("base")

def transcribe_and_translate(audio_path):
    result = model.transcribe(audio_path, task="translate")
    return result["text"]

audio_file = r"C:\Users\nahme\Downloads\test-audio-file.mp3"
translated_text = transcribe_and_translate(audio_file)
print("Translated Transcription:", translated_text)
