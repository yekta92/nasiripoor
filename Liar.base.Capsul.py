# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 09:30:39 2021

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
################################## Tokenizer ################################
from keras.preprocessing.text import Tokenizer
import numpy as np

max_words=10000
embedding_dim=300
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

EMBEDDING_FILE='D:/data/GoogleNews-vectors-negative300.bin'
from gensim.models import KeyedVectors
model_w2v = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


num_words=min(max_words, len(word_index) +1)
embedding_matrix=np.zeros((num_words,embedding_dim))
for word , i in tokenizer.word_index.items():
    if i>= max_words:
        continue
    if word in model_w2v.vocab:
        embedding_vector=model_w2v[word]
        embedding_vector=np.array(embedding_vector)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
            
print(embedding_matrix.shape)


len(model_w2v.index_to_key)
len(model_w2v.key_to_index)
len(model_w2v.get_vecattr)
len(model_w2v.set_vecattr)

################################### model #####################################
from keras.layers import Dropout,Flatten,Dense, Input, Embedding,Conv1D,MaxPooling1D,concatenate,AveragePooling1D
from keras.models import Model


# channel 1
inputs1 = Input(shape=(maxlen,))
EMD1 = Embedding(max_words,embedding_dim,input_length=maxlen,weights=[embedding_matrix],trainable=False)(inputs1)

conv1 = Conv1D(filters=128, kernel_size=4, activation='relu',input_shape = (maxlen,embedding_dim))(EMD1)
drop1 = Dropout(0.2)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(maxlen,))
EMD2 = Embedding(max_words,embedding_dim,input_length=maxlen,weights=[embedding_matrix],trainable=False)(inputs2)

conv2 = Conv1D(filters=128, kernel_size=6, activation='relu',input_shape=(maxlen,embedding_dim))(EMD2)
drop2 = Dropout(0.2)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(maxlen,))
EMD3 = Embedding(max_words,embedding_dim,input_length=maxlen,weights=[embedding_matrix],trainable=False)(inputs3)

conv3 = Conv1D(filters=128, kernel_size=8, activation='relu',input_shape=(maxlen,embedding_dim))(EMD3)
drop3 = Dropout(0.2)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
out_put = concatenate([flat1, flat2, flat3])

Avg_pooling = AveragePooling1D(pool_size=2)(out_put)

outputs = Dense(6, activation='softmax')(Avg_pooling)

model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

# summarize
model.summary()
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history=model.fit([X_train,X_train,X_train,X_train_metadata],y_train,
                   validation_data=([X_val,X_val_metadata],y_val),epochs=10, batch_size=32)


    
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
score=model.evaluate([X_test,X_test,X_test,X_test_meta_data])
model.metrics_names
score


y_pred=model.predict([X_test,X_test,X_test,X_test_meta_data])
y_pred=(y_pred > 0.3)
test_correct=y_test==y_pred
Acc=np.mean(test_correct)






