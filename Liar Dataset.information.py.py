# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:25:28 2021

@author: mashadservice.com
"""


################################# import Dataset ##############################
import pandas as pd
df=pd.read_csv('D:/data/Data.csv')

X_train=df['preprocess'][: 10239]
y_train=df['label'][: 10239]
x_val=df['preprocess'][10239:11522]
y_val=df['label'][10239:11522]
X_test=df['preprocess'][11522 :]
y_test=df['label'][11522 :]

###############################################################################
import matplotlib.pyplot as plt
import seaborn as sb

X_train['len_processTitle'] = [len(w) for w in X_train['text']]
plt.hist(X_train['len_processTitle'], bins=range(1,20, 1), 
              alpha=0.5, color="blue")
labels = ['Length of news title']
plt.xlabel("length of text")
plt.ylabel("proportion")
plt.legend(labels)
plt.title("Distribution words in text of news")
plt.savefig('title words distribution.jpg')

#length of string
res1 = max(len(ele) for ele in X_train)
res2 = min(len(ele) for ele in X_train)

print("Length of maximum string is : " + str(res1))
print("Length of manimum string is : " + str(res2))

##############################################################################
import matplotlib.pyplot as plt  
import numpy as np  
text=X_train
plt.hist([len(x) for x in text], bins=500)
plt.show()

nos = np.array([len(x) for x in text])
len(nos[nos  < 150])
print(len(nos[nos  < 150]),'news have less than 150 words')

#Average word length in a text#################################################
import numpy as np        
print(X_train.groupby(y_train).count())
sb.countplot(y_train)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=X_train[X_train['label']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('FALSE')
word=X_train[X_train['label']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('TRUE')
fig.suptitle('Average word length in each text')


fig,(ax3,ax4)=plt.subplots(3,4,figsize=(20,10))
word=X_train[X_train['label']==2]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='blue')
ax3.set_title('half-true')
word=X_train[X_train['label']==3]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax4,color='orange')
ax4.set_title('mostly-true')
fig.suptitle('Average word length in each text')


fig,(ax5,ax6)=plt.subplots(5,6,figsize=(20,10))
word=X_train[X_train['label']==4]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax5,color='gray')
ax5.set_title('barely-true')
word=X_train[X_train['label']==5]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax6,color='yellow')
ax6.set_title('pants-fire')
fig.suptitle('Average word length in each text')

###############################################################################
#length of word
X_train=X_train.astype(str)
import numpy as np
doc_len = X_train.apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)
sb.distplot(doc_len, hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length'); plt.legend()
plt.show()







