import streamlit as st

st.set_page_config(page_title="LB dashboard",layout="centered")
st.title("Load Balance Dashboard")
st.write("Welcome to streamlit this wide layout")

col1,col2 = st.columns(2)

with col1:
	st.write("Left")

with col2:
	st.write("Right")
