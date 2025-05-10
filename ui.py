import streamlit as st
import os
import shutil
from main import main_logic

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Personal Assistant for answering your questions.")

uploaded_files = st.file_uploader()