# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:30:45 2020

@author: mashadservice.com
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 07:50:38 2020

@author: mashadservice.com
"""


#load Dataset
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('D:/data/New folder/fake_or_real_news.csv')
df=df.drop('Unnamed: 0',axis=1)
print(df['label'].value_counts())

#matplotlib inline
import seaborn as sb
plt.figure (figsize=(6,6))
p = sb.countplot(data=df,x = 'label',)

#clean punctuation
import string as st
def remove_punctuation(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))
df['New_text']=df['title']+df['text'].apply(lambda x: remove_punctuation(x))

#tokenization
import re
def tokenize(text):
    text = re.split('\s+', text)
    return [x.lower() for x in text]
df['New_text'] = df['New_text'].apply(lambda msg:tokenize(msg))

#lower    
def rem_small_words(text):
    return [x for x in text if len(x)>2]
df['New_text'] = df['New_text'].apply(lambda x: rem_small_words(x))

#clean stopwords
import nltk
def rem_stopword(text):
    return[word for word in text if word not in nltk.corpus.stopwords.words('english')]
    
df['New_text'] = df['New_text'].apply(lambda x: rem_stopword(x))

#Lemmatizer
from nltk import  WordNetLemmatizer
def lemmatizer(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]
df['New_text'] = df['New_text'].apply(lambda x: lemmatizer(x))
#######
import string 
for punc in  string.punctuation:
    df['New_text']=df['New_text'].replace(punc,__)
        
unwanted_digit= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for digit in unwanted_digit:
    df['New_text']=df['New_text'].replace(digit,__)
    
unwanted_punc=['[', ']',',','.','آ','“','â€\u200c','â€','”','http','\n','\t','\r','"','\x80','\x9d','\x94','=','+','-','_',')','(','*','&','&','^','%','%','$','#','@','!','?','>','<',':','...','"']    
for punc in  unwanted_punc:
    df['New_text']=df['New_text'].replace(punc,__)  
#######

'''sent tokenize
def return_setances(tokens):
    return " ".join([word for word in tokens])   
df['New_text'] = df['New_text'].apply(lambda x: return_setances(x))'''

###############################################################################
df= df.to_csv('D:/data/DataFrame.csv')

















