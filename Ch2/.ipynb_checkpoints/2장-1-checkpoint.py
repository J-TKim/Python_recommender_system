# u.user 파일을 DataFrame로 읽기
import pandas as pd
u_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
users = pd.read_csv("../data/u.user", sep="|", names=u_cols, encoding="latin-1")
users = users.set_index("user_id")
users.head()