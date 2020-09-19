# u.user 파일을 DataFrame로 읽기
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()

# u.item 파일을 DataFrame으로 읽기
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies.head()

# u.data 파일을 DataFrame으로 읽기
import pandas as pd
r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv("../data/u.data", sep="\t", names=r_cols,
    encoding="latin-1")
ratings = ratings.set_index("user_id")
ratings.head()