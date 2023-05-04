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
