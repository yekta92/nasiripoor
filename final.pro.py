# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:02:16 2020

@author: mashadservice.com
"""
#load Data
import pandas as pd
df=pd.read_csv('D:/data/DataFrame.csv')

###############################################################################
#Information Data
from wordcloud import WordCloud
import matplotlib.pyplot as plt
fake_data = df[df["label"] == "FAKE"]
fake_text = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(fake_text)
plt.figure(figsize= [20,10])
plt.imshow(wordcloud)
plt.axis("off")
plt.title('FAKE NEWS WORDCLOUD',fontsize= 30)
plt.show()


real_data = df[df["label"] == "REAL"]
real_text = ' '.join([text for text in real_data.text])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(real_text)
plt.figure(figsize= [20,10])
plt.imshow(wordcloud)
plt.axis("off")
plt.title('REAL NEWS WORDCLOUD',fontsize= 30)
plt.show()


import seaborn as sb          
print(df.groupby(['label'])['text'].count())
sb.countplot(df['label'])

fig, (ax1,ax2)=plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Characters in News Title',fontsize=20)
news_len=df[df['label']=='REAL']['title'].str.len()
ax1.hist(news_len,color='orange',linewidth=2,edgecolor='black')
ax1.set_title('REAL news',fontsize=15)
news_len=df[df['label']=='FAKE']['title'].str.len()
ax2.hist(news_len,linewidth=2,edgecolor='black')
ax2.set_title('Fake news',fontsize=15)

#Unigram Analysis
from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
    

plt.figure(figsize = (16,9))
most_common_uni = get_top_text_ngrams(df.text,10,1)
most_common_uni = dict(most_common_uni)
sb.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))

#Bigram Analysis
plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(df.text,10,2)
most_common_bi = dict(most_common_bi)
sb.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))


#Trigram Analysis
plt.figure(figsize = (16,9))
most_common_tri = get_top_text_ngrams(df.text,10,3)
most_common_tri = dict(most_common_tri)
sb.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))

# selecting top 20 most frequent hashtags    
import nltk
import matplotlib.pyplot as plt

fake_text_vis =' '.join([str(x) for x in df[df['label']=='FAKE']['New_text']])
a = nltk.FreqDist(fake_text_vis.split())
d = pd.DataFrame({'Word': list(a.keys()),
                  'Count': list(a.values())})
d.sample(10) 
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sb.barplot(data=d, x= "Word", y = "Count")
ax.set_xticklabels(d["Word"], rotation=40, ha="right")
ax.set(ylabel = 'Count')
plt.show()
##############################################################################
#training model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df['New_text'],df['label'],test_size=0.2, random_state = 10)

df["label"]=df["label"].replace(["FAKE","REAL"],value=[1,0])

#LDA
from sklearn.preprocessing import StandardScaler  
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
    
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=10)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)
 

#Average word length in a text
import seaborn as sb
import numpy as np
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['label']==0]['New_text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('REAL')
word=df[df['label']==1]['New_text'].str.split().apply(lambda x : [len(i) for i in x])
sb.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Fake')
fig.suptitle('Average word length in each text')

  
#TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
print(X_train.shape)
print(X_test.shape)

#model1:RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

model1= RandomForestClassifier()
model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,pred1)
cm1 = confusion_matrix(y_test,pred1)
print("Accuracy score RandomForestClassifier : {}".format(accuracy1))
print("Confusion matrix : \n {}".format(cm1))
print(classification_report(y_test,pred1))
print(plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#model2:LogisticRegression
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(max_iter = 500)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,pred2)
cm2 = confusion_matrix(y_test,pred2)
print("Accuracy score LogisticRegression: {}".format(accuracy2))
print("Confusion matrix : \n {}".format(cm2))
print(classification_report(y_test,pred2))
print(plot_confusion_matrix(conf_mat=cm2,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#model3:PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
model3 = PassiveAggressiveClassifier(max_iter=50)
model3.fit(X_train,y_train)
pred3 = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,pred3)
cm3 = confusion_matrix(y_test,pred3)
print("Accuracy score PassiveAggressiveClassifier: {}".format(accuracy3))
print("Confusion matrix : \n {}".format(cm3))
print(classification_report(y_test,pred3))
print(plot_confusion_matrix(conf_mat=cm3,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#model4:SVC
from sklearn.svm import SVC
model4=SVC(gamma='auto')
model4.fit(X_train,y_train)
pred4 = model4.predict(X_test)
accuracy4 = accuracy_score(y_test,pred4)
cm4 = confusion_matrix(y_test,pred4)
print("Accuracy score SVC : {}".format(accuracy4))
print("Confusion matrix : \n {}".format(cm4))
print(classification_report(y_test,pred4))
print(plot_confusion_matrix(conf_mat=cm4,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#model5:DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model5=DecisionTreeClassifier()
model5.fit(X_train,y_train)
pred5 = model5.predict(X_test)
accuracy5 = accuracy_score(y_test,pred5)
cm5 = confusion_matrix(y_test,pred5)
print("Accuracy score DecisionTreeClassifier : {}".format(accuracy5))
print("Confusion matrix : \n {}".format(cm5))
print(classification_report(y_test,pred4))
print(plot_confusion_matrix(conf_mat=cm5,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#KFold
from sklearn.model_selection import KFold
KF1=KFold(n_splits=10)
def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)
print(get_score(PassiveAggressiveClassifier(),X_train,X_test,y_train,y_test))
print('\n\n\n')


KF2=KFold(n_splits=10)
def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)
print(get_score(LogisticRegression(),X_train,X_test,y_train,y_test))
print('\n\n\n')


KF3=KFold(n_splits=10)
def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)
print(get_score(DecisionTreeClassifier(),X_train,X_test,y_train,y_test))
print('\n\n\n')



#LSA
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=True,stop_words='english')
X =vectorizer.fit_transform(X_train)
from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=300,n_iter=100)
lsa.fit(X)
terms = vectorizer.get_feature_names()
for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")
   
#RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
params={'max_depth':[None,3],
        'max_features':randint(1,20),
        'min_samples_leaf':randint(1,20)}
tree=DecisionTreeClassifier()
tree_cv=RandomizedSearchCV(tree,params,cv=10)
tree_cv.fit(X_train,y_train)
print('tree_cv.best_params=',tree_cv.best_params_)
print('tree_cv.best_score=',tree_cv.best_score_)
print('score.tree_cv=',tree_cv.score(X_test,y_test))

#chi2
from scipy.stats import chi2_contingency
table=pd.crosstab(df['New_text'],df['label'])
chi2,p_value,dof,expected=chi2_contingency(table.values)
print('chi2=',chi2)
print('p_value.chi2=',p_value)
if p_value > 0.05:
    print('text and label not fail to reject Null hypothsis,Null is True')
else:
    print('text and label fail to reject Null hypothsis or Null is False')
print('\n\n\n')

#XGBoost
X_train.toarray()
from xgboost import XGBClassifier
model6=XGBClassifier()
model6.fit(X_train,y_train)
pred6=model6.predict(X_test)
accuracy6 = accuracy_score(y_test,pred6)
print('accuracy_score_XGBoost: {}'.format(accuracy6))
cm6 = confusion_matrix(y_test,pred6)
print("Confusion matrix : \n {}".format(cm6))
print(classification_report(y_test,pred6))
print(plot_confusion_matrix(conf_mat=cm4,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n')

from sklearn.model_selection import KFold
KF4=KFold(n_splits=10)
def get_score(model,x_train,X_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(X_test,y_test)
print('accuracy_score_XGBoost_10fold=',get_score(XGBClassifier(),X_train,X_test,y_train,y_test))
print('\n\n\n')

#Knn
from sklearn.neighbors import KNeighborsClassifier
model7=KNeighborsClassifier(n_neighbors=50)
model7.fit(X_train,y_train)
pred7=model7.predict(X_test)
accuracy7 = accuracy_score(y_test,pred7)
print('accuracy_score_KNN: {}'.format(accuracy6))
cm7 = confusion_matrix(y_test,pred7)
print("Confusion matrix : \n {}".format(cm7))
print(classification_report(y_test,pred7))
print(plot_confusion_matrix(conf_mat=cm7,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL']))
print('\n\n\n')

#GaussianNB
from sklearn.naive_bayes import GaussianNB
model8=GaussianNB()
pred8=model8.fit(X_train,y_train).predict(X_test)
print('accuracy_score.gnb=',model8.score(y_test,pred8))



    
#Creating the Dictionary with model name as key adn accuracy as key-value
labels={'RandomForestClassifier':accuracy1,'LogisticRegression':accuracy2,'PassiveAggressiveClassifier':accuracy3,
        'SVC':accuracy4,'DecisionTreeClassifier':accuracy5 ,'Xgboost':accuracy6 ,'knn':accuracy7}

#Plotting accuracy of all the models with Bar-Graphs
plt.figure(figsize=(15,8))
plt.title('Comparing Accuracy of ML Models',fontsize=20)
colors=['red','yellow','orange','magenta','cyan','blue','purple']
plt.xticks(fontsize=10,color='black')
plt.yticks(fontsize=20,color='black')
plt.ylabel('Accuracy',fontsize=20)
plt.xlabel('Models',fontsize=20)
plt.bar(labels.keys(),labels.values(),edgecolor='black',color=colors, linewidth=2,alpha=0.5)















