import string
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

top_words = 10000
max_review_length = 500

word_dict = imdb.get_word_index()
word_dict = {key: (value + 3) for key, value in word_dict.items()}
word_dict[''] = 0  # Padding
word_dict['&amp;amp;amp;gt;'] = 1  # Start
word_dict['?'] = 2  # Unknown word


def analyze(text):
    # Prepare the input by removing punctuation characters, converting
    # characters to lower case, and removing words containing numbers
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower().split(' ')
    text = [word for word in text if word.isalpha()]

    # Generate an input tensor
    input = [1]
    for word in text:
        if word in word_dict and word_dict[word] < top_words:
            input.append(word_dict[word])
        else:
            input.append(2)
    padded_input = sequence.pad_sequences([input], maxlen=max_review_length)
    return padded_input
