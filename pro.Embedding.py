# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:56:37 2020

@author: mashadservice.com
"""

import pandas as pd
df= pd.read_csv('D:/data/DataFrame.csv')

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
df["label"]=df["label"].replace(["FAKE","REAL"],value=[1,0])
text=df['New_text']
labels=df['label']

############################train_test_split###################################
maxlen=150
max_words=100000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)
word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=maxlen)
print('len(word_index):',len(word_index))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,
                                                 test_size=0.2, random_state = 10)
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)

x_val=X_train[: 4000]
y_val=y_train[: 4000]
print('x_val:',x_val.shape)
print('y_val:',y_val.shape)
##################################### model ###################################
import keras 
import tensorflow as tf
model=keras.Sequential()
model.add(keras.layers.Embedding(max_words,300))   
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(300,activation=tf.nn.relu))     
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit(X_train, y_train, 
                  validation_data=(x_val,y_val), epochs=40, batch_size=512)

###################################### plot ###################################
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
###############################accuracy in X_test##############################
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sb
pred = model.predict_classes(X_test)
print(classification_report(y_test, pred, target_names = ['Fake','Real']))
cm = confusion_matrix(y_test,pred)
cm = pd.DataFrame(cm , index = ['Fake','Real'] , columns = ['Fake','Real'])
print('confusion_matrix:\n',cm)
accuracy = accuracy_score(y_test,pred)
print('accuracy:',accuracy)
plt.figure(figsize = (5,5))
sb.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True,fmt='' , xticklabels = ['Fake','Real'] , yticklabels = ['Fake','Real'])
plt.xlabel("Predicted")
plt.ylabel("Actual")








