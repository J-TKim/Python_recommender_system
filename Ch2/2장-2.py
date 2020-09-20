#!/usr/bin/env python
# coding: utf-8

# ## 2.4 사용자 집단별 추천

# In[1]:


# 데이터 읽어오기
import pandas as pd
u_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
users = pd.read_csv("../data/u.user", sep="|", names=u_cols, encoding="latin-1")
i_cols = ["movie_id", "title", "release date", "video release date", "IMDB URL",
         "unknown", "Action",  "Adventure", "Animation", "CHildren\'s", "Comedy",
         "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
         "Mystery", "Romance", "Sci-Fi", "Thriller", "war", "western"]
movies = pd.read_csv("../data/u.item", sep="|", names=i_cols, encoding="latin-1")
r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv("../data/u.data", sep="|", names=r_cols, encoding="latin-1")

# timestamp 제거
ratings = ratings.drop("timestamp", axis=1)

# Movie ID와 title 빼고 다른 데이터 제거
movies = movies[["movie_id", "title"]]


# In[2]:


# train, test set 분리
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings["user_id"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[5]:


# 정확도(RMSE)를 계산하는 함수
import numpy as np

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


# In[ ]:




