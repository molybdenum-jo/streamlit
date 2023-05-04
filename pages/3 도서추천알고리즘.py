import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic
import streamlit as st

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book_Rating'] >= 4]

# 사용자-아이템 행렬 생성
pivot_data = train.pivot_table(index='User_ID', columns='Book-Title', values='Book_Rating', fill_value=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(pivot_data)

# 사용자 기반 협업 필터링 (KNNBasic) 모델 구축
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train[['User_ID', 'Book-Title', 'bookRating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True}
user_based_cf = KNNBasic(sim_options=sim_options)
user_based_cf.fit(trainset)

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_ratings = pivot_data[book_title]
    similar_books_index = cos_sim[np.argsort(book_ratings)][-6:-1]
    similar_books = list(pivot_data.index[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = user_based_cf.predict(uid=book, iid=book_title)
        if est >= 4.0:
            recommended_books.append(book)
    return recommended_books

# Streamlit 앱 구성
st.title('Book Recommender')
book_title = st.text_input('Enter a book title')
if book_title in pivot_data.columns:
    recommended_books = recommend_books(book_title)
    if len(recommended_books) > 0:
        st.write('Recommended books:')
        for book in recommended_books:
            st.write('- ' + book)
    else:
        st.write('No recommended books')
else:
    st.write('Enter a valid book title')

