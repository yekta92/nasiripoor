# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:24:18 2021

@author: mashadservice.com
"""

#load Data
import pandas as pd
df=pd.read_csv('D:/data/data.csv')

X_train=df['text'][: 11522]
y_train=df['label'][: 11522]
x_val=df['text'][10239:11522]
y_val=df['label'][10239:11522]
X_test=df['text'][11522 :]
y_test=df['label'][11522 :]

import keras
y_train=keras.utils.to_categorical(y_train,num_classes=6)
y_test=keras.utils.to_categorical(y_test,num_classes=6)
y_val=keras.utils.to_categorical(y_val,num_classes=6)

###############################################################################
import numpy as np
import os
glove_dir='D:/data'
embedding_index={}
f=open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf8')
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embedding_index[word]=coefs
f.close()
print('found %s word vectors.' %len(embedding_index))

#Train
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

maxlen=200
max_words=10000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
sequence1=tokenizer.texts_to_sequences(X_train)
word_index1=tokenizer.word_index
X_train=pad_sequences(sequence1,maxlen=maxlen)
print('len(word_index1):',len(word_index1))

embedding_dim=100
embedding_matrix1=np.zeros((max_words,embedding_dim)) 
for word ,i in word_index1.items():
    embedding_vector=embedding_index.get(word)
    if i<max_words:
        if embedding_vector is not None:
            embedding_matrix1[i]=embedding_vector
       
#test    
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_test)
sequence2=tokenizer.texts_to_sequences(X_test)
word_index2=tokenizer.word_index
X_test=pad_sequences(sequence2,maxlen=maxlen)
print('len(word_index2):',len(word_index2))

embedding_dim=100
embedding_matrix2=np.zeros((max_words,embedding_dim)) 
for word ,i in word_index2.items():
    embedding_vector=embedding_index.get(word)
    if i<max_words:
        if embedding_vector is not None:
            embedding_matrix2[i]=embedding_vector
   
#val            
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_val)
sequence3=tokenizer.texts_to_sequences(x_val)
word_index3=tokenizer.word_index
x_val=pad_sequences(sequence3,maxlen=maxlen)
print('len(word_index3):',len(word_index3))

embedding_dim=100
embedding_matrix3=np.zeros((max_words,embedding_dim)) 
for word ,i in word_index3.items():
    embedding_vector=embedding_index.get(word)
    if i<max_words:
        if embedding_vector is not None:
            embedding_matrix3[i]=embedding_vector
       
##################################### model ###################################

from keras.layers import Flatten,Dense,Dropout,Conv1D,MaxPooling1D
from keras.models import Sequential
import keras

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same',input_shape=embedding_matrix1.shape))

model.add(MaxPooling1D(2,2)) 

model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same',input_shape=embedding_matrix1.shape))

model.add(MaxPooling1D(2,2))

model.add(Dropout(0.2))

model.add(Flatten()) 

model.add(Dense(120,activation='relu')) 
model.add(Dense(6,activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#problem:Error when checking input: expected conv1d_9_input to have 3 dimensions, but got array with shape (11522, 200)

history=model.fit(X_train, y_train, validation_data=(x_val,y_val),
                  epochs=40, batch_size=256)
