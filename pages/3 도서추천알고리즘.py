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
    similar_books_index = np.unique(np.argsort(cos_sim[pivot_data.columns.get_loc(book_title)])[-6:-1])
    similar_books = list(pivot_data.columns[similar_books_index])
    recommended_books = []
    for book in similar_books:
        _, _, _, est, _ = item_based_cf.predict(uid=book_title, iid=book)
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
    book_idx = np.where(book_titles == book_title)[0][0]
    similar_books_idx = np.argsort(cos_sim[book_idx])[:-6:-1]
    similar_books = book_titles[similar_books_idx]
    return similar_books

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


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
import streamlit as st

# 데이터 불러오기
train = pd.read_csv('data/TRAIN.csv')
train['Book-Rating'] = pd.to_numeric(train['Book-Rating'], errors='coerce').fillna(0).astype(int)

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# Book-ID에 고유한 정수 인덱스 부여
unique_books = list(set(train['Book-ID']))
book_to_idx = {book: i for i, book in enumerate(unique_books)}
idx_to_book = {i: book for book, i in book_to_idx.items()}
train['Book-ID'] = train['Book-ID'].map(book_to_idx)

# 사용자-아이템 행렬 생성
num_users = len(train['User-ID'].unique())
num_books = len(train['Book-ID'].unique())
ratings_matrix = np.zeros((num_users, num_books))
for row in train.itertuples():
    user_idx = int(row[1].split('_')[1]) - 1
    book_idx = row[3]
    rating = int(row[2])
    ratings_matrix[user_idx, book_idx] = rating


# 딥러닝 모델 구축
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 20)(user_input)
user_vec = Flatten()(user_embedding)

item_embedding = Embedding(num_books, 20)(item_input)
item_vec = Flatten()(item_embedding)

prod = Dot(name='Dot-Product', axes=1)([user_vec, item_vec])

dense1 = Dense(64, activation='relu')(prod)
dense2 = Dense(1)(dense1)

model = Model([user_input, item_input], dense2)
model.compile(loss='mse', optimizer='adam')

model.fit([ratings_matrix[:, 0], ratings_matrix[:, 1]], ratings_matrix[:, 2], batch_size=128, epochs=10, validation_split=0.1)

# 유사한 책 5개 추천
def recommend_books(book_id):
    book_idx = book_to_idx[book_id]
    book_vec = model.get_layer('Embedding_2')(np.array([book_idx]))
    sim_scores = cosine_similarity(book_vec, model.get_layer('Embedding_2').get_weights()[0])[0]
    sim_books_idx = np.argsort(sim_scores)[-6:-1]
    sim_books = [idx_to_book[i] for i in sim_books_idx]
    return sim_books

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

js = "window.scrollTo(0, document.getElementById('part-5-book').offsetTop);"

st.markdown("<h3 id='part-4-book'>✅Part 4. 앙상블 기법을 사용한 추천 시스템</h3>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 불러오기
import numpy as np
import pandas as pd

train_df = pd.read_csv('data/TRAIN.csv')

n_users = len(train_df['user_id'].unique())
n_books = len(train_df['book_id'].unique())

# ratings_matrix 초기화
ratings_matrix = np.zeros((n_users, n_books))

for row in train_df.itertuples():
    user_idx = int(row[1]) - 1
    book_idx = int(row[3])
    rating = row[2]
    ratings_matrix[user_idx, book_idx] = rating

train = train[train['Book-Rating'].str.isnumeric()]  # 숫자로만 이루어진 Book-Rating만 선택
train['Book-Rating'] = train['Book-Rating'].astype(int)  # Book-Rating 열을 정수형으로 변환

# 평점이 4점 이상인 데이터만 사용
train = train[train['Book-Rating'] >= 4]

# Book-ID에 고유한 정수 인덱스 부여
unique_books = list(set(train['Book-ID']))
book_to_idx = {book: i for i, book in enumerate(unique_books)}
idx_to_book = {i: book for book, i in book_to_idx.items()}
train['Book-ID'] = train['Book-ID'].map(book_to_idx)

# 사용자-아이템 행렬 생성
num_users = len(train['User-ID'].unique())
num_books = len(train['Book-ID'].unique())
ratings_matrix = np.zeros((num_users, num_books))
for row in train.itertuples():
    ratings_matrix[int(row[1])-1, int(row[3])] = int(row[2])


# 딥러닝 모델 구축
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 20)(user_input)
user_vec = Flatten()(user_embedding)

item_embedding = Embedding(num_books, 20)(item_input)
item_vec = Flatten()(item_embedding)

prod = Dot(name='Dot-Product', axes=1)([user_vec, item_vec])

dense1 = Dense(64, activation='relu')(prod)
dense2 = Dense(1)(dense1)

model = Model([user_input, item_input], dense2)
model.compile(loss='mse', optimizer='adam')

model.fit([ratings_matrix[:, 0], ratings_matrix[:, 1]], ratings_matrix[:, 2], batch_size=128, epochs=10, validation_split=0.1)

# 유사한 책 5개 추천
def recommend_books(book_id):
    book_idx = book_to_idx[book_id]
    book_vec = model.get_layer('Embedding_2')(np.array([book_idx]))
    sim_scores = cosine_similarity(book_vec, model.get_layer('Embedding_2').get_weights()[0])[0]
    sim_books_idx = np.argsort(sim_scores)[-6:-1]
    sim_books = [idx_to_book[i] for i in sim_books_idx]
    return sim_books

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

    
