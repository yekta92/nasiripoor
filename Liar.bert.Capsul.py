# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:42:51 2021

@author: mashadservice.com
"""

import pandas as pd
df=pd.read_csv('D:/data/Data.csv')

################################### label ####################################
import numpy as np
from keras.utils import np_utils

y=df['label']
y=np_utils.to_categorical(y,num_classes=6)
y=np.reshape(y,(12788,6))
print(type(y))

y_train=y[: 11522]
y_val=y[10239:11522]
y_test=y[11522 :]
################################# Bert Encoding ##############################
import pickle
with open('D:/data/Embeddings1_10.txt', 'rb') as f:
    embeddings1 = pickle.load(f)

with open('D:/data/Embeddings11_18.txt', 'rb') as f:
    embeddings2 = pickle.load(f)

with open('D:/data/Embeddings19_26.txt', 'rb') as f:
    embeddings3 = pickle.load(f)

Embeddings=embeddings1+embeddings2+embeddings3

X_train=Embeddings[: 11522]
X_val=Embeddings[10239:11522]
X_test=Embeddings[11522 :]


################################### meta_data #################################
one_hot_metadata = pd.get_dummies(df['meta_data'])

X_train_meta_data=one_hot_metadata[: 11522]
X_val_meta_data=one_hot_metadata[10239:11522]
X_test_meta_data=one_hot_metadata[11522 :]
maxlength2=1854
#################################### Model ####################################

from keras.layers import Dense, Input, Dropout,Conv1D,MaxPooling1D,concatenate
from keras.layers import Flatten
from keras.models import Model

maxlen=150
max_words=10000
embedding_dim=768

# channel 1
inputs1 = Input(shape=(maxlen,embedding_dim))
conv1 = Conv1D(filters=128, kernel_size=4, activation='relu',input_shape = (maxlen,embedding_dim))(inputs1)
drop1 = Dropout(0.2)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(maxlen,embedding_dim))
conv2 = Conv1D(filters=128, kernel_size=6, activation='relu',input_shape=(maxlen,embedding_dim))(inputs2)
drop2 = Dropout(0.2)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(maxlen,embedding_dim))
conv3 = Conv1D(filters=128, kernel_size=8, activation='relu',input_shape=(maxlen,embedding_dim))(inputs3)
drop3 = Dropout(0.2)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(6, activation='softmax')(dense1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
print(model.summary())	


history=model.fit([X_train,X_train,X_train,X_train_meta_data], y_train, 
                  validation_data=([X_val,X_val,X_val,X_val_meta_data],y_val),epochs=10, batch_size=32)


model.save_weights('CNN_model_proposal.h5')
###################################### history ################################

history_dict=history.history
history_dict.keys()
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.clf()
acc_values=history_dict['acc']
val_acc_values=history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

                                     














