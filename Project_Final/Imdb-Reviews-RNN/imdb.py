import numpy
from numpy import array
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import matplotlib.pyplot as plt

data_id= imdb.get_word_index()
print(data_id)
(X_train1, y_train1), (X_test, y_test)= imdb.load_data(num_words=1000)
word_and_index={i: word for word, i in data_id.items()}
review_words= [word_and_index.get(i, ' ') for i in X_train1[2]]
print(review_words)#third review words
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset

train,test = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
train_x,train_y = train
test_x,test_y = test

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in train_x[0] ))

numpy.random.seed(7)
# dataset  sadece tepedeki 5000 kelime geri kalanı 0'a set et
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Model create
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))#Long Short Term Memory
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#binary çünkü 0 or 1(good or bad)
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128)
#history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#yhat = model.predict(X_train, verbose=0) üzerinde çalış
#print(yhat)

bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
for review in [good,bad]:
    result = []
    for word in review.split(" "):
        result.append(word_to_id[word])#id sine göre review i bulup içine yerleştirdik
    tmp_padded = sequence.pad_sequences([result], maxlen=max_review_length)
    print("%s. Predict Score: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))#0'a ne kdar uzaksa o kadar pozitif
#https://www.kaggle.com/kredy10/simple-lstm-for-text-classification