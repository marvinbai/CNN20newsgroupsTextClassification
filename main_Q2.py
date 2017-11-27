'''
Author: Xiaoyu Bai
Date: Nov 25, 2017
Python version: 3.5.2
The word embedding is based on Global Vectors for Word Representation (GloVe).
This program requires the installation of scikit-learn, TensorFlow and Keras package.
'''

import numpy as np
import os
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dropout
from keras.models import Model


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

GLOVE_DIR = ''  # Directory where GloVe txt file is put.
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100 # Dimension of the word-vector after embedding. It can only be 50, 100, 200, 300. Corresponding GloVe file has to be accompanied.
MAX_NB_WORDS = 20000

# Fetch the text data.
newsgroups_train = fetch_20newsgroups(subset = 'train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset = 'test', shuffle = True)
newsgroups = fetch_20newsgroups(subset = 'all', shuffle = True)
print('Number of training data is ' + str(len(newsgroups_train.data)))
print('Number of test data is ' + str(len(newsgroups_test.data)))


# Text preprocessing using Keras.
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)    # Tokenization will be restricted to the top num_words most common words in the dataset.
tokenizer.fit_on_texts(newsgroups.data)   # Assign each token an index.
sequences_train = tokenizer.texts_to_sequences(newsgroups_train.data)  # Convert each text into a series of index sequences.
word_index = tokenizer.word_index
print('Number of unique token is ' + str(len(word_index)))
data_train = pad_sequences(sequences_train, maxlen = MAX_SEQUENCE_LENGTH)  # Pad the sequence of each text to make them of the same length.
label_train = to_categorical(newsgroups_train.target)
print('Shape of data_train tensor:', data_train.shape)
print('Shape of label_train tensor:', label_train.shape)
sequences_test = tokenizer.texts_to_sequences(newsgroups_test.data)
data_test = pad_sequences(sequences_test, maxlen = MAX_SEQUENCE_LENGTH)
label_test = to_categorical(newsgroups_test.target)


# Build a hashmap to map each token into a vector.
map = {}
file = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'), encoding = 'utf8')
# file = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt')) # For python version 2.
for line in file:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype = 'float32')
    map[word] = vec
print('Number of total vectors in GloVe is ' + str(len(map)))
file.close()

# Construct embedding matrix.
num_words = min(MAX_NB_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = map.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  # token not found will be all zeros.
print(embedding_matrix.sum())

# Build the neural network.
# Embedding layer.
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable = True)  # Set trainable to False to enable GloVe model.
# 1-D convnet.
sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH, ), dtype = 'int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.3)(embedded_sequences)
x = Conv1D(128, 5, activation = 'relu')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, 3, activation = 'relu')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, 5, activation = 'relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation = 'relu')(x)
output = Dense(20, activation = 'softmax')(x)
model = Model(sequence_input, output)
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['acc'])

# Train the neural network.
model.fit(data_train, label_train, verbose = 1, validation_data = (data_test, label_test),
          epochs = 50, batch_size = 128)
  
# Predict the test set.
predicted_test_matrix = model.predict(data_test)
predicted_test = np.zeros(predicted_test_matrix.shape[0], dtype = int)   # Change probability matrix into predicted target array.
for i in range(predicted_test.shape[0]):
    predicted_test[i] = np.argmax(predicted_test_matrix[i])

# Output F-1 score.
print('The F-1 score for test query is ' + str(metrics.f1_score(newsgroups_test.target, predicted_test, average = 'macro')))




