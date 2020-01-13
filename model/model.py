from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence

top_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length,
                    input_length=max_review_length))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=5, batch_size=128)

model.save("model.h5")
