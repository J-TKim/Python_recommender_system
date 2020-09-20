#!/usr/bin/env python
# coding: utf-8

# ## 2.1 데이터 읽기

# In[1]:


# u.user 파일을 DataFrame로 읽기
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()


# In[2]:


# u.item 파일을 DataFrame으로 읽기
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies.head()


# In[3]:


# u.data 파일을 DataFrame으로 읽기
import pandas as pd
r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv("../data/u.data", sep="\t", names=r_cols,
    encoding="latin-1")
ratings = ratings.set_index("user_id")
ratings.head()


# ## 2.2 인기제품 방식

# In[4]:


# Best-seller 추천
def recom_movie1(n_items):
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
recom_movie1(5)


# In[5]:


def recom_moive2(n_items):
    return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]["title"]


# ## 2.3 추천 시스템의 정확도 측정

# In[6]:


# 정확도 계산
import numpy as np

def RMSE(y_ture, y_pred):
    return np.sqrt(np.mean((np.array(y_ture) - np.array(y_pred))**2))


# In[7]:


rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user]["rating"]
    y_pred = movie_mean[ratings.loc[user]["movie_id"]]
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))

