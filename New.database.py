# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:25:28 2021

@author: mashadservice.com
"""


################################# import Dataset ##############################
import pandas as pd
train=pd.read_table('D:/data/train2.txt')
test=pd.read_table('D:/data/test2.txt')
val=pd.read_table('D:/data/val2.txt')
train=train.drop('0',axis=1)
test=test.drop('0',axis=1)
val=val.drop('0',axis=1)
train=train.drop('2635.json',axis=1)
test=test.drop('11972.json',axis=1)
val=val.drop('12134.json',axis=1)

train.columns = ['label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']
test.columns = ['label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']
val.columns = ['label','statement','subject','speaker', 'job', 'state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue','extracted_justification']

train['text'] = train['statement'] + ' ' + train['extracted_justification'] 
test['text'] = test['statement'] + ' ' + test['extracted_justification'] 
val['text'] = val['statement'] + ' ' + val['extracted_justification'] 


train['label']=train['label'].replace(['FALSE','TRUE','half-true','mostly-true' ,'barely-true' ,'pants-fire'],value=[0,1,2,3,4,5])
test['label']=test['label'].replace(['FALSE','TRUE','half-true','mostly-true' ,'barely-true' ,'pants-fire'],value=[0,1,2,3,4,5])
val['label']=val['label'].replace(['FALSE','TRUE','half-true','mostly-true' ,'barely-true' ,'pants-fire'],value=[0,1,2,3,4,5])

####################################preprocessing##############################
import string as st

def preprocessing(new_text):
    def remove_punctuation(text):
        return ("".join([ch for ch in text if ch not in st.punctuation]))
    new_text=new_text.apply(lambda x: remove_punctuation(x))
    
    #tokenization
    import re
    def tokenize(text):
        text = re.split('\s+', text)
        return [x.lower() for x in text]
    new_text=new_text.apply(lambda msg:tokenize(msg))
    
    #lower    
    def rem_small_words(text):
        return [x for x in text if len(x)>2]
    new_text = new_text.apply(lambda x: rem_small_words(x))
    
    import nltk
    def rem_stopword(text):
        return[word for word in text if word not in nltk.corpus.stopwords.words('english')]
        
    new_text = new_text.apply(lambda x: rem_stopword(x))
    
    from nltk import  WordNetLemmatizer
    def lemmatizer(text):
        word_net = WordNetLemmatizer()
        return [word_net.lemmatize(word) for word in text]
    new_text =new_text.apply(lambda x: lemmatizer(x))
    
    import string 
    for punc in  string.punctuation:
        new_text=new_text.replace(punc,__)
            
    unwanted_digit= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for digit in unwanted_digit:
        new_text=new_text.replace(digit,__)
        
    unwanted_punc=['[', ']',',','.','آ','“','â€\u200c','â€','”','http','\n','\t','\r','"','\x80','\x9d','\x94','=','+','-','_',')','(','*','&','&','^','%','%','$','#','@','!','?','>','<',':','...','"']    
    for punc in  unwanted_punc:
        new_text=new_text.replace(punc,__)
    return new_text
        

#data=train+validation+test
data = pd.concat([train, val,test])
data['text'] = data['statement'] + ' ' + data['extracted_justification']
data.text=data.text.astype(str)
data['text']=preprocessing(data['text'])
df=data.to_csv('D:/data/data.csv')


X_train=df['text'][: 10239]
y_train=df['label'][: 10239]
x_val=df['text'][10239:11522]
y_val=df['label'][10239:11522]
X_test=df['text'][11522 :]
y_test=df['label'][11522 :]

###############################################################################
import matplotlib.pyplot as plt
import seaborn as sb

Train['len_processTitle'] = [len(w) for w in Train['text']]
plt.hist(Train['len_processTitle'], bins=range(1,20, 1), 
              alpha=0.5, color="blue")
labels = ['Length of news title']
plt.xlabel("length of text")
plt.ylabel("proportion")
plt.legend(labels)
plt.title("Distribution words in text of news")
plt.savefig('title words distribution.jpg')

#length of string
res1 = max(len(ele) for ele in Train['text'])
res2 = min(len(ele) for ele in Train['text'])

print("Length of maximum string is : " + str(res1))
print("Length of manimum string is : " + str(res2))

##############################################################################
import matplotlib.pyplot as plt  
import numpy as np  
text=Train['text']
plt.hist([len(x) for x in text], bins=500)
plt.show()

nos = np.array([len(x) for x in text])
len(nos[nos  < 150])
print(len(nos[nos  < 150]),'news have less than 150 words')

#Average word length in a text#################################################
import numpy as np        
print(Train.groupby(['label'])['text'].count())
sb.countplot(Train['label'])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=Train[Train['label']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('FALSE')
word=Train[Train['label']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('TRUE')
fig.suptitle('Average word length in each text')


fig,(ax3,ax4)=plt.subplots(3,4,figsize=(20,10))
word=Train[Train['label']==2]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='blue')
ax3.set_title('half-true')
word=Train[Train['label']==3]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax4,color='orange')
ax4.set_title('mostly-true')
fig.suptitle('Average word length in each text')


fig,(ax5,ax6)=plt.subplots(5,6,figsize=(20,10))
word=Train[Train['label']==4]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax5,color='gray')
ax5.set_title('barely-true')
word=Train[Train['label']==5]['text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax6,color='yellow')
ax6.set_title('pants-fire')
fig.suptitle('Average word length in each text')

###############################################################################
#length of word
Train.text=Train.text.astype(str)
import numpy as np
doc_len = Train['text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)
sb.distplot(doc_len, hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length'); plt.legend()
plt.show()







