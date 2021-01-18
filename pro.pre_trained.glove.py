# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:46:00 2020

@author: mashadservice.com
"""

import pandas as pd
df= pd.read_csv('D:/data/DataFrame.csv')

df["label"]=df["label"].replace(["FAKE","REAL"],value=[1,0])
text=df['New_text']
labels=df['label']
###########################the most longest sentences##########################
res1 = max(len(ele) for ele in text)
res2 = min(len(ele) for ele in text)

print("Length of maximum string is : " + str(res1))
print("Length of manimum string is : " + str(res2))
############################train_test_split###################################
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
maxlen=150
max_words=100000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)
word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=maxlen)
print('len(word_index):',len(word_index))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2, random_state = 10)
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)

x_val=X_train[: 4000]
y_val=y_train[: 4000]
print('x_val:',x_val.shape)
print('y_val:',y_val.shape)

############################Pre_trained Glove################################## 
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

embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim)) 
for word ,i in word_index.items():
    embedding_vector=embedding_index.get(word)
    if i<max_words:
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector

print('embedding_matrix.shape:',embedding_matrix.shape)
################################### model #####################################            
from keras.models import Sequential
from keras.layers import Embedding, Flatten , Dense
from keras import layers

model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen,trainable=False)) 
model.add(Flatten())
model.add(layers.Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid')) 
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])         
 
history=model.fit(X_train,y_train,epochs=50,batch_size=512,validation_data=(x_val,y_val))

model.save_weights('pre_trained_glove_model.h5')
########################################plot###################################
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
################################## accuracy in X_test #########################
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sb
pred = model.predict_classes(X_test)
print(classification_report(y_test, pred, target_names = ['Fake','Real']))
cm = confusion_matrix(y_test,pred)
cm = pd.DataFrame(cm , index = ['Fake','Real'] , columns = ['Fake','Real'])
print('confusion_matrix:\n',cm)
accuracy = accuracy_score(y_test,pred)
print('accuracy:',accuracy)
sb.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True,
           fmt='' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
###############################################################################
import seaborn as sb
doc_len = df['New_text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)
sb.distplot(doc_len, hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length'); plt.legend()
plt.show()
