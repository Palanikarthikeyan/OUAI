import streamlit as st
import pandas as pd

file = st.file_uploader('select your input file:')
st.write(file)

if file:
    st.success(f'Input file:  {file.name} is loaded')
    df = pd.read_csv(file)
    st.write(df)
    
