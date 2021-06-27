# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:44:06 2021

@author: mashadservice.com
"""

################################# import Dataset ##############################
import pandas as pd
df=pd.read_csv('D:/data/Data.csv')

################################# Bert Encoding ##############################
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

print("Max Sequence Length:", model.max_seq_length)

#Change the length to 200
model.max_seq_length = 200

print("Max Sequence Length:", model.max_seq_length)

#encoding_bert
sentences = df['bert'].astype('str') 
embeddings = model.encode(sentences)
#save embeddings
from numpy import save
bert_embeddings=save('D:/data/bert.npy', embeddings)
#load embeddings
from numpy import load
embeddings = load('D:/data/bert.npy')
#embeddings=embeddings.tolist()

X_train=embeddings[: 11522]
X_val=embeddings[10239:11522]
X_test=embeddings[11522 :]

############################## One_Hot Encoding(mata_data) ####################
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

df1=[]
for i in range(0,len(df['speaker']),1):
    df1.append(preprocess(str(df['speaker'][i])))
   
maxlength2=8
one_hot_speaker=[one_hot (d,maxlength2) for d in df1]

for review_number in range(len(one_hot_speaker)):
    numberofwords=len(one_hot_speaker[review_number])
    if (numberofwords) > (maxlength2):
        maxlength2=numberofwords
print('max length of word:',maxlength2)

#save one hot of speaker
from numpy import save
one_hot_speaker=save('D:/data/one_hot_speaker.npy', one_hot_speaker)

#load one hot of speaker
from numpy import load
one_hot_speaker = load('D:/data/one_hot_speaker.npy' , allow_pickle=True)

one_hot_speaker=one_hot_speaker.tolist()   

onehot_pad_speaker=pad_sequences(one_hot_speaker,maxlength2)
       
x_train_metadata=onehot_pad_speaker[: 11522]
y_train=df['label'][: 11522]
x_val_metadata=onehot_pad_speaker[10239:11522]
y_val=df['label'][10239:11522]
x_test_metadata=onehot_pad_speaker[11522 :]
y_test=df['label'][11522 :]

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,num_classes=6)
y_test=np_utils.to_categorical(y_test,num_classes=6)
y_val=np_utils.to_categorical(y_val,num_classes=6)
            
#####################################  model  #################################
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Conv1D,MaxPooling1D,concatenate
from keras.layers import Bidirectional,Flatten
from keras.models import Model

maxlen=384
max_words=1000
embedding_dim=384


inp = Input(shape=(maxlen,)) 
x1 = Embedding(max_words,embedding_dim,input_length=maxlen,weights=[embeddings], 
               trainable=False)(inp)
x1 = Conv1D(filters=128, kernel_size=2,activation='relu')(x1)
x1 = MaxPooling1D(pool_size=2)(x1)

x1 = Conv1D(filters=128, kernel_size=3,activation='relu')(x1)
x1 = MaxPooling1D(pool_size=2)(x1)

x1 = Conv1D(filters=128, kernel_size=4,activation='relu')(x1)
x1 = MaxPooling1D(pool_size=2)(x1)

x1 = (Flatten())(x1)
x1 = Dropout(0.8)(x1)

doc_model=Model(inp,x1)
output1=doc_model.output

Inp = Input(shape=(maxlength2,))
x2 = Embedding(maxlen,embedding_dim,input_length=maxlength2)(Inp)

x2 = Conv1D(filters=10,kernel_size=3,activation='relu')(x2)
x2 = MaxPooling1D(pool_size=2)(x2)

x2 = Conv1D(filters=10,kernel_size=8,activation='relu')(x2)
x2 = MaxPooling1D(pool_size=2)(x2)

x2 = Dropout(0.5)(x2)
x2 = Dropout(0.8)(x2)

x2 = Bidirectional(LSTM(120, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x2)
output2=x2

output=concatenate([output1,output2]) 
 
x = Dense(6, activation="softmax")(output)
model = Model(inputs=[doc_model.input,x2.Input], outputs=x)
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

'''
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(optimizer=adam, loss='mse', metrics=['mae', 'mape', 'acc'])
callbacks = [EarlyStopping('val_loss', patience=3)] '''


history=model.fit([X_train,x_train_metadata], y_train, 
                  validation_data=([X_val,x_val_metadata],y_val),epochs=10, batch_size=64)




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

                     