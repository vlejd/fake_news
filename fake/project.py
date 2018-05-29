from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence

from qrnn import QRNN
import json
import numpy as np

VOCAB_SIZE = 10000
MAX_SEQ_LEN = 256
WORD_DIM = 128
BATCH_SIZE = 32

f = open('./data/dataset.json')
d = json.load(f)
split = int((50000*0.7)//BATCH_SIZE)*BATCH_SIZE
limit = 50000//BATCH_SIZE*BATCH_SIZE
print(split, limit)
X_train = np.array([x[0] for x in d[:split]])
X_test = np.array([x[0] for x in d[split:limit]])

y_train = np.array([x[1] for x in d[:split]])
y_test = np.array([x[1] for x in d[split:limit]])

X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LEN)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, WORD_DIM, input_length=MAX_SEQ_LEN, batch_input_shape=(BATCH_SIZE, MAX_SEQ_LEN)))
model.add(QRNN(128, window=3, 
                input_dim=WORD_DIM,
                input_length=MAX_SEQ_LEN, 
                batch_input_shape=(BATCH_SIZE, MAX_SEQ_LEN, WORD_DIM),
                ret_sequence=True))
model.add(QRNN(128, window=3, 
                input_dim=WORD_DIM,
                input_length=MAX_SEQ_LEN, 
                batch_input_shape=(BATCH_SIZE, MAX_SEQ_LEN, WORD_DIM),
                ret_sequence=True))
model.add(QRNN(128, window=2, 
                input_dim=WORD_DIM,
                input_length=MAX_SEQ_LEN, 
                batch_input_shape=(BATCH_SIZE, MAX_SEQ_LEN, WORD_DIM)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=BATCH_SIZE ,epochs = 10, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("%s: %.2f%%" % (model.metrics_names[1], acc*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")