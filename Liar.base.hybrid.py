# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:20:20 2021

@author: yekta.com
"""

################################### load Data #################################
import pandas as pd
df=pd.read_csv('D:/data/Data.csv')

#################################### label ####################################
from keras.utils import np_utils
import numpy as np
y=df['label'][: 5000]
y=np_utils.to_categorical(y,num_classes=6)
y=np.reshape(y,(5000,6))
print(type(y))

y_train=y[: 4000]
y_val=y[3500:4000]
y_test=y[4000 :5000]

################################### meta_data #################################
print(df['speaker'].nunique())
print(df['job'].nunique())
print(df['state'].nunique())
print(df['party'].nunique())

one_hot_speaker=pd.get_dummies(df['speaker'][: 5000])
one_hot_job=pd.get_dummies(df['job'][: 5000])
one_hot_state=pd.get_dummies(df['state'][: 5000])
one_hot_party=pd.get_dummies(df['party'][: 5000])

metadata = pd.concat([one_hot_speaker, one_hot_job,one_hot_state,one_hot_party], axis=1)

X_train_metadata=metadata[: 4000]
X_val_metadata=metadata[3500:4000]
X_test_metadata=metadata[4000 :5000]

maxlength2=2707           
 
################################### Tokenizer ##################################
from keras.preprocessing.text import Tokenizer
import numpy as np

max_words=10000
maxlen=150

text=df['preprocess'][: 5000]

from keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)
word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=maxlen)
print('len(word_index):',len(word_index))

X_train=data[: 4000]
X_val=data[3500:4000]
X_test=data[4000 :5000]

################################ pre_trained W2V ##############################
import gensim
import numpy as np
wordembeddings=gensim.models.KeyedVectors.load_word2vec_format('D:/data/GoogleNews-vectors-negative300.bin',binary=True)

unique_words=len(word_index)
total_words=unique_words + 1
skipped_words=0
embedding_dim=300
embedding_vector=[0]

embedding_matrix=np.zeros((total_words,embedding_dim))
for word , index in tokenizer.word_index.items():
    try:
        embedding_vector=wordembeddings[word]
    except:
        skipped_words=skipped_words+1
        pass
    if embedding_vector is not None:
            embedding_matrix[index]=embedding_vector
print('embedding_matrix shape:', embedding_matrix.shape)            
            
            
#####################################  model  #################################
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Conv1D,MaxPooling1D,concatenate
from keras.layers import Bidirectional,Flatten
from keras.models import Model

# channel 1
inputs1 = Input(shape=(maxlen,))
emd1 = Embedding(total_words, embedding_dim, input_length=maxlen,weights=[embedding_matrix],trainable=False)(inputs1)
                    
conv1 = Conv1D(filters=128, kernel_size=2, activation='relu',input_shape = (maxlen,embedding_dim))(emd1)
drop1 = Dropout(0.8)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
out_put1 = Flatten()(pool1)

model1 = Model(inputs=inputs1, outputs=out_put1)

Inp = Input(shape=(maxlength2,))
x2 = Embedding(maxlen,output_dim=maxlength2,input_length=maxlength2)(Inp)

x2 = Conv1D(filters=10,kernel_size=3,activation='relu',padding='same')(x2)
x2 = MaxPooling1D(2,2)(x2)

x2 = Conv1D(filters=10,kernel_size=4,activation='relu',padding='same')(x2)
x2 = MaxPooling1D(2,2)(x2)

x2 = Dropout(0.5)(x2)
x2 = Dropout(0.8)(x2)

out_put2 = Bidirectional(LSTM(120,return_sequences=False,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x2)
final_output=concatenate([out_put1,out_put2])
# interpretation

outputs = Dense(6, activation='softmax')(final_output)

model2 = Model(inputs=[inputs1,Inp], outputs=outputs)
# summarize
model2.summary()
# compile
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model2.fit([X_train,X_train_metadata],y_train,validation_data=([X_val,X_val_metadata],y_val),epochs=10, batch_size=64)

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

  
###############################acc in test set ################################
'''score=model2.evaluate([X_test,X_test,X_test,X_test_metadata])
model2.metrics_names
score


y_pred=model2.predict([X_test,X_test,X_test,X_test_metadata])
y_pred=(y_pred > 0.3)
test_correct=y_test==y_pred
Acc=np.mean(test_correct)'''
                                   
