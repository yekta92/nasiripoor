# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:56:34 2021

@author: mashadservice.com
"""

#Collaborative filtering for movieLens dataset

#############################################################################
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

ratings=pd.read_csv("D:/data/movelens/ratings.csv")
moveis=pd.read_csv("D:/data/movelens/movies.csv")

print(ratings.head())

rating = pd.merge(moveis,ratings ,on='movieId').drop(['genres','timestamp'], axis=1)
movie_matrix=rating.pivot_table(index='userId' ,columns='title' , values='rating')
movie_matrix.head()



rate={}
rows_indexes={}
for i , row in movie_matrix.iterrows():
    rows=[x for x in range(0,len(movie_matrix.columns)) ]
    combine=list(zip(row.index, row.values, rows))
    rated=[(x,z) for x,y,z in combine if str(y) != 'nan']
    index=[i[1] for i in rated]
    row_names=[i[0] for i in rated]
    rows_indexes[i]=index
    rate[i]=row_names
    
import numpy as np 
pivot_table=rating.pivot_table(index='userId' ,columns='title' , values='rating').fillna(0)
pivot_table=pivot_table.apply(np.sign)


notrated={}
notrated_indexes={}

for i , row in pivot_table.iterrows():
    rows=[x for x in range(0,len(movie_matrix.columns)) ]
    combine=list(zip(row.index, row.values, rows))
    idx_row=[(idx,col) for idx,val,col  in combine if not val > 0]
    indices=[i[1] for i in idx_row]
    row_names=[i[0] for i in idx_row]
    notrated_indexes[i]=indices
    notrated[i]=row_names
    
#Unsupervised Nearest Neighbour Recommender
from sklearn.neighbors import NearestNeighbors
n=5
cosine_nn=NearestNeighbors(n_neighbors=n, algorithm='brute' , metric='cosine')
item_cosine_nn_fit=cosine_nn.fit(pivot_table.T.values)
item_distances, item_indices=item_cosine_nn_fit.kneighbors(pivot_table.T.values)


#Item_Based Recommender
import numpy as np

item_dict={}
for i in range(len(pivot_table.T.index)):
    item_idx=item_indices[i]
    col_names=pivot_table.T.index[item_idx].tolist()
    item_dict[pivot_table.T.index[i]]=col_names

topRecs={}
for k , v in rows_indexes.items():
    item_idx=[ j for i in item_indices[v] for j in i]
    item_dist=[j for i in item_distances[v] for j in i]
    combine=list(zip(item_dist, item_idx))
    diction={i:d for d,i in combine if i not in v}
    zipped=list(zip(diction.keys() , diction.values()))
    sort=sorted(zipped,key=lambda x:x[1] )
    recommendations=[(pivot_table.columns[i] , d) for i,d in sort]
    topRecs[k]=recommendations


def getrecommendations(user,number_of_recs=30):
    if user > len(pivot_table.index):
        print('out of range, there are only {} users, try again'.format(len(pivot_table.index)))
    else:
        print('there are all the movies you have viewed view in the past:\n\n{} '. format('\n'.join(rate[user])))
        print()
        print('we recommend to view these movies too:\n')
    
    for k , v in topRecs.items():
        if user == k :
            for i in v[:number_of_recs]:
                print('{} with similarity: {:.4f}'.format(i[0], 1 - i[1]))

#top recommendations

print(getrecommendations(601))

# make rating predictions for the movies users had not seen before!
item_distances=1-item_distances
predictions=item_distances.T.dot(pivot_table.T.values) / np.array([np.abs(item_distances.T).sum(axis=1)]).T
ground_truth=pivot_table.T.values[item_distances.argsort()[0]]


# Evaluating the recommenders predictions
def rmse(prediction,ground_truth):
    prediction=prediction[ground_truth.nonzero()].flatten()
    ground_truth=ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))


error_rate=rmse(predictions,ground_truth)
print("Accuracy:{:.3f}".format(100-error_rate))
print('RMSE :{: .5f}'.format(error_rate))

















