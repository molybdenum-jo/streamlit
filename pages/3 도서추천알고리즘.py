import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
from surprise import KNNWithMeans, Reader, Dataset


st.header('📖도서평점 예측을 통한 도서추천알고리즘')


st.sidebar.markdown("""
    ## 도서추천알고리즘 분석
    - [part 1. 협업 필터링 기반의 추천시스템](#part-1-recommend)
    - [part 2. 콘텐츠 기반 필터링 추천시스템](#part-2-recommend)
    - [part 3. 행렬 인수분해 기반 추천시스템](#part-3-recommend)
    - [part 4. 딥러닝 모델 기반 추천시스템](#part-4-recommend)
    - [part 5. 앙상블 기법을 사용한 추천시스템](#part-5-recommend)
    - [part 6. 하이퍼파라미터 최적화를 통한 추천시스템](#part-6-recommend)
""")
st.write('')
st.write("""
- 도서 평점 예측을 통한 도서 추천 알고리즘은 사용자가 선호하는 도서를 추천하는 데 도움을 줄 수 있습니다. 이 알고리즘은 사용자의 평가 이력을 기반으로 작동하며, 이를 사용하여 사용자가 어떤 책을 좋아할지 예측합니다..
""")
st.write('')
st.write('')


js = "window.scrollTo(0, document.getElementById('part-1-recommend').offsetTop);"


st.markdown("<h3 id='part-1-recommend'>✅Part 1. 협업 필터링 기반의 추천시스템</h3>", unsafe_allow_html=True)

st.write('')
st.write('')
st.write("""
- 사용자 기반 협업 필터링 및 아이템 기반 협업 필터링 모델을 구현하고, 이들 모델의 평점 예측 성능을 평가한다. RMSE 값으로 성능을 비교하여 어떤 협업 필터링 방법이 더 나은 성능을 보이는지 결정한다.
""")

train = pd.read_csv("data/TRAIN.csv")

from sklearn.model_selection import train_test_split

# Train/Validation 데이터 분할
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)

# 사용자별 평균 평점 특성 생성
user_mean_rating = train_data.groupby('User-ID')['Book-Rating'].mean().reset_index()
user_mean_rating.columns = ['User-ID', 'User-Mean-Rating']

# 사용자별 평균 평점이 결측치인 경우, 전체 평균 평점으로 대체
mean_rating = train_data['Book-Rating'].mean()
user_mean_rating['User-Mean-Rating'] = user_mean_rating['User-Mean-Rating'].fillna(mean_rating)

# 사용자별 평가 횟수 특성 생성
user_rating_count = train_data.groupby('User-ID')['Book-Rating'].count().reset_index()
user_rating_count.columns = ['User-ID', 'User-Rating-Count']

# train_data와 val_data에 특성을 추가
train_data = train_data.merge(user_mean_rating, on='User-ID', how='left')
train_data = train_data.merge(user_rating_count, on='User-ID', how='left')

val_data = val_data.merge(user_mean_rating, on='User-ID', how='left')
val_data = val_data.merge(user_rating_count, on='User-ID', how='left')

# 사용자별 평균 평점과 책별 평균 평점을 포함한 새로운 데이터프레임 생성
user_item_rating_mean = train_data[["User-ID", "Book-ID", 'User-Mean-Rating']]

# 데이터를 Surprise 라이브러리 형식으로 변환
reader = Reader(rating_scale=(0, 10))
data_mean = Dataset.load_from_df(user_item_rating_mean, reader)

# 전체 데이터를 trainset으로 변환
trainset = data_mean.build_full_trainset()

# 사용자 기반 협업 필터링 (KNNWithMeans) 모델 구축
sim_options = {'name': 'pearson_baseline', 'user_based': False}
user_based_cf = KNNWithMeans(sim_options=sim_options)

# 샘플링된 데이터로 모델 학습
user_based_cf.fit(trainset)

# 사용자 기반 협업 필터링 예측
user_based_cf_preds = []
for _, row in val_data.iterrows():
    user_based_cf_preds.append(user_based_cf.predict(row['User-ID'], row['Book-ID']).est)
    
# 샘플링된 데이터로 모델 학습
user_based_cf.fit(trainset)



js = "window.scrollTo(0, document.getElementById('part-2-recommend').offsetTop);"

st.markdown("<h3 id='part-2-recommend'>✅Part 2. 콘텐츠 기반 필터링 추천시스템</h3>", unsafe_allow_html=True)


js = "window.scrollTo(0, document.getElementById('part-3-recommend').offsetTop);"

st.markdown("<h3 id='part-3-recommend'>✅Part 3. 행렬 인수분해 기반 추천시스템</h3>", unsafe_allow_html=True)

js = "window.scrollTo(0, document.getElementById('part-4-recommend').offsetTop);"

st.markdown("<h3 id='part-4-recommend'>✅Part 4. 딥러닝 모델 기반 추천시스템</h3>", unsafe_allow_html=True)

js = "window.scrollTo(0, document.getElementById('part-5-recommend').offsetTop);"

st.markdown("<h3 id='part-5-recommend'>✅Part 5. 앙상블 기법을 사용한 추천시스템</h3>", unsafe_allow_html=True)

js = "window.scrollTo(0, document.getElementById('part-6-recommend').offsetTop);"

st.markdown("<h3 id='part-6-recommend'>✅Part 6. 하이퍼파라미터 최적화를 통한 추천시스템</h3>", unsafe_allow_html=True)
