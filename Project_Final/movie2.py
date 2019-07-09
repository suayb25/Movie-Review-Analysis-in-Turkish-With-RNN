import pandas as pd
import numpy as np
from numpy import array
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Adding dataset and Determinate  title of the dataset

column = ['yorum']
df = pd.read_csv('person.csv', encoding ='iso-8859-9', sep='"')
df.columns=column
df.info()
df.head()
#df.dropna(inplace=True)


#Removal of the Turkish stop-words in the data set
def remove_stopwords(df_fon):
    stopwords = open('turkce-stop-words.txt', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords], df_fon['yorum']))

remove_stopwords(df)

#Create a column called Positivity in the Data Set and initially assign 1 as a positive value to all values
df['Positivity'] = 1


#In our data set, data from the 394th data and after-data are negative data so we can change the Positivity values of these values to 0.
df.Positivity.iloc[394:] = 0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['yorum'], df['Positivity'], random_state = 0)
print(X_train.head())
print('\n\nX_train shape: ', X_train.shape)

tok = Tokenizer(num_words=5000)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=100)

top_words = 5000
def RNN():
    inputs = Input(name='inputs',shape=[100])
    layer = Embedding(5000,50,input_length=100)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history= model.fit(sequences_matrix,y_train,batch_size=16,epochs=100,
          validation_split=0.2)
#history= model.fit(sequences_matrix,y_train,batch_size=128,epochs=100, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=100)
accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#bad = "çok kötü bir film.iğrençti.beğenmedim.film gerçekten kötüydü."
#good = "çok güzel bir film mükemmeldi.iyi bir film harikaydı.acayip iyiydi."
bad = "this movie was terrible and bad"#I have never watched such a horrible movie in my life.
good = "i really liked the movie and had fun"#The best turkish film I ever watched ...
twt = ['The best turkish film I ever watched ...']
deneme = Tokenizer(num_words=5000)
deneme.fit_on_texts(twt)
deneme_sequences = deneme.texts_to_sequences(twt)
deneme_sequences_matrix = sequence.pad_sequences(deneme_sequences, maxlen=100)
#tmp_padded = sequence.pad_sequences([result], maxlen=100)
print("%s. Predict Score: %s" % (twt,model.predict(array([deneme_sequences_matrix][0]))[0][0]))#0'a ne kdar uzaksa o kadar pozitif

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'loss')
tokenizer = Tokenizer(num_words=5000)
#vectorizing the sentence by the pre-fitted tokenizer instance
tokenizer.fit_on_texts(twt)
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding` input
twt = sequence.pad_sequences(twt, maxlen=100, dtype='int32', value=0)
print(twt)
print("şimdi en son:")
print("%s. Predict Score: %s" % (twt,model.predict(array([twt][0]))[0][0]))#0'a ne kdar uzaksa o kadar pozitif
sentiment = model.predict(twt,batch_size=8,verbose = 2)[0]
print(sentiment)
#if(np.argmax(sentiment) == 0):
#   print("negative")
#elif (np.argmax(sentiment) == 1):
#    print("positive")

#if(sentiment<=0.7):
#    print("negative")
#elif (sentiment>0.7):
#    print("positive")