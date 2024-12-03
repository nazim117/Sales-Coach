import pandas as pd

def load_movie_lines(file_path):
    lines = pd.read_csv(file_path, sep=' +++$+++ ', header=None, names=['line_id', 'character_id', 'movie_id', 'text'], engine='python')
    return lines

def load_movie_conversations(file_path):
    conversations = pd.read_csv(file_path, sep=' +++$+++ ', header=None, names=['character_1', 'character_2', 'movie_id', 'lines'], engine='python')
    return conversations

def prepare_data(lines_file_path, conv_file_path):
    lines_df = load_movie_lines(lines_file_path)
    conversations_df = load_movie_conversations(conv_file_path)
    return lines_df, conversations_df
