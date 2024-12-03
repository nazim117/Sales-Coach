import whisper
from textblob import TextBlob
import nltk
from dotenv import load_dotenv
import os

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

model = whisper.load_model("base")

def transcribe_and_translate(audio_path):
    result = model.transcribe(audio_path, task="translate")
    return result["text"]


audio_file = os.getenv('AUDIO_URL')
translated_text = transcribe_and_translate(audio_file)

print("Translated Transcription:", translated_text)

blob = TextBlob(translated_text)

print("POS Tags:", blob.tags)

print("Noun Phrases:", blob.noun_phrases)

for sentence in blob.sentences:
    print("Sentence Sentiment Polarity:", sentence.sentiment.polarity)
