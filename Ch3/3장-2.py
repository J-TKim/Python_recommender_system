#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# 데이터 읽어 오기 
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/u.user', sep='|', names=u_cols, encoding='latin-1')
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 'unknown', 
          'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
          'Thriller', 'War', 'Western']
movies = pd.read_csv('../data/u.item', sep='|', names=i_cols, encoding='latin-1')
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('../data/u.data', sep='\t', names=r_cols, encoding='latin-1')


# In[3]:


# timestamp 제거 
ratings = ratings.drop('timestamp', axis=1)
# movie ID와 title 빼고 다른 데이터 제거
movies = movies[['movie_id', 'title']]


# In[4]:


# train, test 데이터 분리
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)


# In[5]:


# 정확도(RMSE)를 계산하는 함수 
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


# In[6]:


# 모델별 RMSE를 계산하는 함수 
def score(model, neighbor_size=0):
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    y_pred = np.array([model(user, movie, neighbor_size) for (user, movie) in id_pairs])
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)


# In[7]:


# train 데이터로 Full matrix 구하기 
rating_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')


# In[8]:


##### (1)
# train set의 모든 가능한 사용자 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)


# In[9]:


# Neighbor size를 정해서 예측치를 계산하는 함수
def cf_knn(user_id, movie_id, neighbor_size=0):
    if movie_id in rating_matrix:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_matrix[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        
        ##### (2) Neighbor size 가 지정되지 않은 경우
        if neighbor_size == 0:
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            
        ##### (3) Neighbor size 가 지정된 경우
        else:
            if len(sim_scores) > 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            else:
                mean_rating = 3.0
                
    else:
        mean_rating = 3.0
    return mean_rating

# 정확도 계산
score(cf_knn, neighbor_size=30)

