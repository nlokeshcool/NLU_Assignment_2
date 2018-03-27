from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import os
import numpy as np
import random
import sys
import pickle
import re

orig_stdout = sys.stdout
f = open('rnn_Output_word_RMSProp.txt', 'w')
sys.stdout = f

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
with open("train.pickle", "rb") as train_file:
    text = pickle.load(train_file)
print("Original text is of length : " , len(text))  
"""
text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry.\
         Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown\
         printer took a galley of type and scrambled it to make a type specimen book. It has survived not \
         only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. \
         It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, \
         nd more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\
        Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of \
        classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor\
        at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, \
        from a Lorem Ipsum passage, and going through the cites of the word in classical literature, \
        discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of \
        de Finibus Bonorum et Malorum (The Extremes of Good and Evil) by Cicero, written in 45 BC. \
        This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum,"
"""
text = text.lower()
words = re.sub("[^\w]", " ",  text).split()
print('corpus length in characters :', len(text))
print("corpus length in words : ", len(words))

words_unique = sorted(set(words))
print("Number of Unique words : ", len(words_unique))

word_indices = dict((w, i) for i,w in enumerate(words_unique))
indices_words = dict((i, w) for i, w in enumerate(words_unique))

maxlen = 40
step = 3
sentences = []
next_words = []

print("Creating sentence of length 80 each and the step size is 3")
for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_words.append(words[i + maxlen])
print('number of sequences:', len(sentences))

print('Vectorizing the sentences now')
print("the input sentences are not one hot encoded but in utput i force a prob distribution in the vocabulary")
print("creating an array of num(sentences) * ", maxlen)
x = np.zeros((len(sentences), maxlen, 1), dtype=np.int)
y = np.zeros((len(sentences), len(words_unique)), dtype=np.int)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t, 0] = word_indices[word]
    y[i, word_indices[next_words[i]]] = 1 


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, 1)))
model.add(Dense(len(words_unique)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print("saving the model after : ", epoch, " epoch")
    model_name = "./models/word_RMSProp/rnn_word_model_"
    model_name += str(epoch) + ".h5"
    model.save(model_name)
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(words) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = words[start_index: start_index + maxlen]
        generated += " ".join(str(x) for x in sentence)
        print('----- Generating with seed: ', sentence )
        sys.stdout.write(generated)

        for i in range(20):
            x_pred = np.zeros((1, maxlen, 1), dtype=np.int)
            for t, word in enumerate(sentence):
                x_pred[0, t, 0] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_words[next_index]

            generated += " " + next_word
            sentence[1:].append(next_word)

            sys.stdout.write(next_word + " " )
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=20,
callbacks=[print_callback])

model.save("Ass_2_rnn_char_final_model.h5")
f.close()

