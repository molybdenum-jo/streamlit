import streamlit as st
import nbconvert
from nbconvert import PythonExporter
import io

# ipynb 파일 업로드
uploaded_file = st.file_uploader("Choose an ipynb file", type="ipynb")

if uploaded_file is not None:
    # ipynb 파일을 파이썬 스크립트로 변환
    notebook = nbformat.read(io.BytesIO(uploaded_file.getbuffer()), nbformat.NO_CONVERT)
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(notebook)

    # 파이썬 스크립트 실행
    exec(source)
