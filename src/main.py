from transcription.transcriber import transcribe_and_translate
from sentiment_analysis.analyzer import generate_summary_and_suggestions
import os
from dotenv import load_dotenv
from data_loader import prepare_data

# load_dotenv()

# audio_file = os.getenv('AUDIO_URL')

# translated_text = transcribe_and_translate(audio_file)
# print("Translated Transcription:", translated_text)

# result = generate_summary_and_suggestions(translated_text)
# print(result)

def main():
    lines_file_path = 'data/archive/movie_lines.txt'
    conv_file_path = 'data/archive/movie_conversations.txt'
    
    lines_df, conversations_df = prepare_data(lines_file_path, conv_file_path)

    print(lines_df.head())
    print(conversations_df.head())

if __name__ == '__main__':
    main()
