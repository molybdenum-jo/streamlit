import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic
import streamlit as st
import matplotlib as plt
import plotly.express as px


st.header('📖도서추천알고리즘')


st.sidebar.markdown("""
    ## 도서 중심 분석
    - [part 1. 협업 필터링 기반의 추천 시스템](#part-1-book)
    - [part 2. 콘텐츠 기반 필터링 기반의 추천 시스템](#part-2-book)
    - [part 3. 행렬 인수분해  기반의 추천 시스템](#part-3-book)
    - [part 4. 딥 러닝 모델 기반의 추천 시스템](#part-4-book)
    - [part 5. 앙상블 기법을 사용한 추천 시스템](#part-5-book)
    - [part 6. 하이퍼마라미터 최적화를 통한 추천 시스템](#part-6-book)
""")
st.write('')
st.write("""
- 출판사나 도서 기업의 입장에서 도서 평점을 활용한 분석이다. 출판사/도서기업이 가지고 있는 고객 데이터와 평점 등을 통해 출판사/도서기업의 데이터가 고객의 평점에 어떠한 영향을 미치는지를 분석하여 고객의 선호도를 분석한다.
""")
st.write('')
st.write('')



js = "window.scrollTo(0, document.getElementById('part-1-book').offsetTop);"

    
st.markdown("<h3 id='part-1-book'>✅Part 1. 협업 필터링 기반의 추천 시스템</h3>", unsafe_allow_html=True)

st.write("""
✔ 사용자기반 협업필터링
""")

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# 사용자-아이템 행렬 생성
pivot_data = train.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating', fill_value=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(pivot_data)

# 사용자 기반 협업 필터링 (KNNBasic) 모델 구축
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train[['User-ID', 'Book-Title', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True}
user_based_cf = KNNBasic(sim_options=sim_options)
user_based_cf.fit(trainset)

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_rating = pivot_data[book_title]
    similar_books_index = np.unique(np.argsort(book_rating)[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
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
    
    
st.write("""
✔ 아이템기반 협업필터링
""")

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# 사용자-아이템 행렬 생성
pivot_data = train.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating', fill_value=0)

# 코사인 유사도 계산
cos_sim = cosine_similarity(pivot_data.T)

# 아이템 기반 협업 필터링 (KNNBasic) 모델 구축
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train[['User-ID', 'Book-Title', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
item_based_cf = KNNBasic(sim_options=sim_options)
item_based_cf.fit(trainset)

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_rating = pivot_data[book_title]
    similar_books_index = np.unique(np.argsort(book_rating)[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = user_based_cf.predict(uid=book, iid=book_title)
        if est >= 4.0:
            recommended_books.append(book)
    return recommended_books


# Streamlit 앱 구성
st.title('Book Recommender')
book_title = st.text_input('Enter a book title', key='book_input')
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

js = "window.scrollTo(0, document.getElementById('part-2-book').offsetTop);"

    
st.markdown("<h3 id='part-2-book'>✅Part 2. 콘텐츠 기반 필터링 기반의 추천 시스템</h3>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# 데이터 전처리: 책 제목의 중복 제거 및 TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
book_titles = train['Book-Title'].unique()
tfidf_matrix = tfidf_vectorizer.fit_transform(book_titles)
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_rating = pivot_data[book_title]
    similar_books_index = np.unique(np.argsort(book_rating)[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = user_based_cf.predict(uid=book, iid=book_title)
        if est >= 4.0:
            recommended_books.append(book)
    return recommended_books

# Streamlit 앱 구성
st.title('Book Recommender')
book_title = st.text_input('Enter a book title', key='book_title_input')
if book_title in book_titles:
    recommended_books = recommend_books(book_title)
    st.write('Recommended books:')
    for book in recommended_books:
        st.write('- ' + book)
else:
    st.write('Enter a valid book title')

js = "window.scrollTo(0, document.getElementById('part-3-book').offsetTop);"

st.markdown("<h3 id='part-3-book'>✅Part 3. 행렬 인수분해  기반의 추천 시스템</h3>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset
from surprise import SVD
import streamlit as st

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# 사용자-아이템 행렬 생성
pivot_data = train.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating', fill_value=0)

# SVD 모델 구축
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train[['User-ID', 'Book-Title', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
svd_model = SVD(n_factors=20, reg_all=0.02)
svd_model.fit(trainset)

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_rating = pivot_data[book_title]
    similar_books_index = np.unique(np.argsort(cos_sim[pivot_data.columns.get_loc(book_title)])[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = svd_model.predict(uid=book_title, iid=book)
        if est >= 4.0:
            recommended_books.append(book)
    return recommended_books

# Streamlit 앱 구성
st.title('Book Recommender')
book_title = st.text_input('Enter a book title', key='input')
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

js = "window.scrollTo(0, document.getElementById('part-4-book').offsetTop);"

st.markdown("<h3 id='part-4-book'>✅Part 4. 딥 러닝 모델 기반의 추천 시스템</h3>", unsafe_allow_html=True)

js = "window.scrollTo(0, document.getElementById('part-5-book').offsetTop);"

st.markdown("<h3 id='part-5-book'>✅Part 5. 앙상블 기법을 사용한 추천 시스템</h3>", unsafe_allow_html=True)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import streamlit as st
import random
import string

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# 사용자-아이템 행렬 생성
pivot_data = train.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating', fill_value=0)

# 훈련 데이터와 테스트 데이터로 분할
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train[['User-ID', 'Book-Title', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
testset = trainset.build_testset()

# SVD 모델 구축
svd_model = SVD(n_factors=20, reg_all=0.02)
svd_model.fit(trainset)

# Item-based 앙상블 모델 구축
# 책 제목 기반으로 벡터화
count_vect = CountVectorizer()
book_title_matrix = count_vect.fit_transform(train['Book-Title'])
book_title_sim = cosine_similarity(book_title_matrix)

svd_preds = svd_model.test(testset)
item_preds = item_based_recommendation(user_id, book_title_sim)
ensemble_preds = [(0.7 * svd_pred.est) + (0.3 * item_pred) for svd_pred, item_pred in zip(svd_preds, item_preds)]

# 사용자가 선택한 책과 유사한 책 5개 추천
def recommend_books(book_title):
    book_rating = pivot_data[book_title]
    similar_books = list(set(svd_similar_books + item_similar_books))
    similar_books_index = np.unique(np.argsort(cos_sim[pivot_data.columns.get_loc(book_title)])[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = svd_model.predict(uid=book_title, iid=book)
        if est >= 4.0:
            recommended_books.append(book)
    return recommended_books

# Streamlit 앱 구성
st.title('Book Recommender')
book_title = st.text_input('Enter a book title', key='input')
if book_title in pivot_data.columns:
    recommended_books = recommend_books(book_title)
    if len(recommended_books) > 0:
        st.write('Recommended books:')
        for book in recommended_books:
            st.write('- ' + book
